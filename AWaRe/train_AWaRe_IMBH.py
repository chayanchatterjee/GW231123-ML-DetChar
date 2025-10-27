import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import fnmatch
import scipy.signal

from ml4gw import augmentations, distributions, gw, transforms, waveforms
import torchaudio.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from peft import LoraConfig, get_peft_model
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WavUNetAttention(nn.Module):
    """
    U-Net variant tuned for heavy-mass BBH (few-cycle) signals,
    with smaller kernels, self-attention at the bottleneck,
    and heteroscedastic uncertainty (mean + log-variance output).
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        num_levels: int = 3,
        kernel_size: int = 7,
        device: str = 'cuda',
        attn_heads: int = 4
    ):
        super().__init__()
        self.device = device
        self.num_levels = num_levels
        padding = kernel_size // 2

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True)
                )
            )
        self.pool = nn.MaxPool1d(2, 2)

        # --- Bottleneck Attention ---
        bottleneck_ch = base_filters * (2 ** (num_levels - 1))
        self.attn = nn.MultiheadAttention(
            embed_dim=bottleneck_ch,
            num_heads=attn_heads,
            batch_first=True
        )

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, 0, -1):
            in_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** (i - 1))
            self.upconvs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True)
                )
            )

        # --- Final projection: 2 channels (mean and log-variance) ---
        self.final_conv = nn.Conv1d(base_filters, 2, kernel_size=1)
        self.to(device)

    def forward(self, x: torch.Tensor):
        # Encoder
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < self.num_levels - 1:
                x = self.pool(x)
        
        # Bottleneck + self-attention
        # x: [B, C, T]
        B, C, T = x.shape
        # to [B, T, C] for batch_first attention
        x_seq = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = attn_out.permute(0, 2, 1)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                x = F.pad(x, (diff // 2, diff - diff // 2))
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        out = self.final_conv(x)
        # Split into mean and log-variance
        mu, s = out.chunk(2, dim=1)

        # ensure var >= min_var
        min_var = 1e-2
        var = F.softplus(s) + min_var
        logvar = torch.log(var)
        
        return mu, logvar


class MultiSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[128,256,512], hop_sizes=[32,64,128]):
        super().__init__()
        self.fft_sizes, self.hop_sizes = fft_sizes, hop_sizes

    def forward(self, pred, targ):
        loss = 0
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            P = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True)
            T = torch.stft(targ.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True)
            loss += F.l1_loss(P.abs(), T.abs())
        return loss / len(self.fft_sizes)

# Gaussian NLL loss
import math

def gaussian_nll(mu, logvar, target, eps=1e-6):
    # var = exp(logvar), but clamp it away from 0 and ∞
    var = torch.exp(logvar).clamp(min=eps, max=1e3)
    # NLL per sample: 0.5*(logvar + (target-mu)^2/var + log(2π))
    nll = 0.5 * ( logvar + (target - mu)**2 / var + math.log(2 * math.pi) )
    return nll.mean()


def gaussian_nll_tfp_style(mu, s_raw, target, scale_factor, eps=1e-6):
    """
    TFP-style Gaussian NLL:
    - s_raw: unconstrained scale output (before softplus)
    - sigma = softplus(s_raw) + eps
    """
    sigma = (F.softplus(s_raw) + eps) * torch.exp(scale_factor)
    var = sigma ** 2

    nll = 0.5 * (torch.log(2 * math.pi * var) + (target - mu)**2 / var)
    return nll.mean()



# =============================================================================
# Ml4gw Reconstruction Model Wrapper (using ml4gw training framework)
# =============================================================================
class Ml4gwReconstructionModel(torch.nn.Module):
    """
    Ml4gwReconstructionModel
    A PyTorch-based model for gravitational wave signal reconstruction using a neural network architecture. 
    This model is designed to handle noise augmentation, waveform generation, and signal reconstruction 
    from interferometer data.
    Attributes:
        nn (torch.nn.Module): The neural network architecture used for reconstruction.
        device (str): The device to run the model on ("cuda" or "cpu").
        ifos (list): List of interferometers used in the analysis.
        kernel_length (float): Length of the kernel in seconds.
        fduration (float): Duration of the frequency domain whitening window in seconds.
        psd_length (float): Length of the PSD estimation window in seconds.
        sample_rate (float): Sampling rate of the data in Hz.
        fftlength (float): FFT length for PSD estimation in seconds.
        highpass (float): Highpass filter cutoff frequency in Hz.
        chunk_length (float): Length of data chunks in seconds.
        reads_per_chunk (int): Number of reads per chunk.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training and validation.
        waveform_prob (float): Probability of injecting a waveform into the data.
        waveform_duration (float): Duration of the waveform in seconds.
        inversion_prob (float): Probability of inverting the signal.
        reversal_prob (float): Probability of reversing the signal.
        f_min (float): Minimum frequency for waveform generation.
        f_max (float): Maximum frequency for waveform generation.
        f_ref (float): Reference frequency for waveform generation.
        min_snr (float): Minimum signal-to-noise ratio for waveform scaling.
        max_snr (float): Maximum signal-to-noise ratio for waveform scaling.
        max_epochs (int): Maximum number of training epochs.
        checkpoint_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory to save training logs.
        scale_factor (torch.nn.Parameter): Scale factor for loss computation.
        use_presaved (bool): Flag to indicate whether to use pre-saved datasets.
        param_dict (dict): Parameter distributions for waveform generation.
        approximant (callable): Waveform approximant for generating waveforms.
        spectral_density (ml4gw.transforms.SpectralDensity): PSD estimation transform.
        whitener (ml4gw.transforms.Whiten): Whitening transform.
        detector_tensors (torch.Tensor): Detector tensor geometry.
        detector_vertices (torch.Tensor): Detector vertex geometry.
        frequencies (torch.Tensor): Frequency bins for waveform generation.
        freq_mask (torch.Tensor): Mask for valid frequency bins.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        current_epoch (int): Current training epoch.
        global_step (int): Global training step.
        best_metric (float): Best validation metric achieved.
        train_losses (list): List of training losses.
        valid_metrics (list): List of validation metrics.
        steps (list): List of training steps.
        responses_file (str): Optional file path for test set injections.
    Methods:
        forward(X):
            Forward pass through the neural network.
        augment_noise2noise(X):
            Augments data using the Noise2Noise approach.
        augment_for_test(X):
            Augments data for testing, optionally using pre-saved responses.
        weighted_mse(pred, targ, eps=1e-3):
            Computes a weighted mean squared error loss.
        training_step(batch):
            Performs a single training step.
        validation_step(batch):
            Performs a single validation step.
        log_metrics():
            Logs training and validation metrics to a CSV file.
        fit():
            Trains the model for the specified number of epochs.
        test_and_save_on_test_set():
            Tests the model on the test set and saves the results.
        generate_waveforms(batch_size):
            Generates waveforms for a batch of data.
        project_waveforms(hc, hp):
            Projects waveforms onto detector tensors.
        rescale_snrs(responses, psd):
            Rescales waveforms to match target signal-to-noise ratios.
        sample_waveforms(responses):
            Samples waveforms for training or testing.
        create_train_dataloader():
            Creates a DataLoader for training data.
        create_val_dataloader():
            Creates a DataLoader for validation data.
        create_test_dataloader():
            Creates a DataLoader for test data.
        _create_test_dataset(num_samples):
            Creates a test dataset and saves it to disk.
        create_new_dataloader():
            Creates a new DataLoader for additional data.
        save_checkpoint(path):
            Saves the model checkpoint to the specified path.
        load_checkpoint(path):
            Loads the model checkpoint from the specified path.
    """
    def __init__(
        self,
        architecture: nn.Module,
        ifos: list = ["L1"],
        kernel_length: float = 1.0,
        fduration: float = 2,
        psd_length: float = 16,
        sample_rate: float = 1024,
        fftlength: float = 2,
        highpass: float = 20,
        chunk_length: float = 128,
        reads_per_chunk: int = 40,
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        waveform_prob: float = 1.0,
        approximant: callable = None,
        param_dict: dict = None,
        waveform_duration: float = 8,
        f_min: float = 20,
        f_max: float = None,
        f_ref: float = 20,
        min_snr: float = 10,
        max_snr: float = 35,
        inversion_prob: float = 0.5,
        reversal_prob: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_epochs: int = 150,
        checkpoint_dir: str = "checkpoints_wavunet",
        log_dir: str = "logs",
        scale_factor = nn.Parameter(torch.tensor(0.0))
    ) -> None:
        super().__init__()
        self.nn = architecture
        self.device = device
        
        # Save hyperparameters.
        self.ifos = ifos
        self.kernel_length = kernel_length
        self.fduration = fduration
        self.psd_length = psd_length
        self.sample_rate = sample_rate
        self.fftlength = fftlength
        self.highpass = highpass
        self.chunk_length = chunk_length
        self.reads_per_chunk = reads_per_chunk
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.waveform_prob = waveform_prob
        self.waveform_duration = waveform_duration
        self.inversion_prob = inversion_prob
        self.reversal_prob = reversal_prob
        self.f_min = f_min
        self.f_max = f_max or (sample_rate / 2)
        self.f_ref = f_ref
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.scale_factor = nn.Parameter(torch.tensor(0.0))
        
        self.use_presaved = False
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create augmentations.
        self.inverter = augmentations.SignalInverter(prob=inversion_prob)
        self.reverser = augmentations.SignalReverser(prob=reversal_prob)
        
        # Use ml4gw transforms for PSD estimation and whitening.
        from ml4gw import transforms
        self.spectral_density = transforms.SpectralDensity(sample_rate, fftlength, average="median", fast=False).to(device)
        self.whitener = transforms.Whiten(fduration, sample_rate, highpass=highpass).to(device)
        
        # Get interferometer geometry.
        from ml4gw import gw
        detector_tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("detector_tensors", detector_tensors.to(device))
        self.register_buffer("detector_vertices", vertices.to(device))
        
        # Frequency setup for waveform generation.
        nyquist = sample_rate / 2
        num_samples = int(waveform_duration * sample_rate)
        num_freqs = num_samples // 2 + 1
        frequencies = torch.linspace(0, nyquist, num_freqs)
        freq_mask = (frequencies >= f_min) * (frequencies < self.f_max)
        self.register_buffer("frequencies", frequencies.to(device))
        self.register_buffer("freq_mask", freq_mask.to(device))
        
        # Parameter distributions for simulated injections.
        if param_dict is None:
            from ml4gw.distributions import PowerLaw, Sine, DeltaFunction
            from torch.distributions import Uniform
          
            param_dict = {
                "chirp_mass": Uniform(10.0, 100.0),
                "mass_ratio": Uniform(0.25, 0.999),
                "chi1": Uniform(-0.999, 0.999),
                "chi2": Uniform(-0.999, 0.999),
                "distance": PowerLaw(100, 1000, 2),
                "phic": DeltaFunction(0),
                "inclination": Sine(),
            }
            
        self.param_dict = param_dict
        
        # For waveform generation.
        from ml4gw import waveforms
        if approximant is None:
            approximant = waveforms.cbc.IMRPhenomD
        self.approximant = approximant().to(device)
        self.psi = torch.distributions.Uniform(0, torch.pi)
        self.phi = torch.distributions.Uniform(-torch.pi, torch.pi)
        
        from ml4gw import distributions
        self.snr = torch.distributions.Uniform(min_snr, max_snr)
        
        self.kernel_size = int(kernel_length * sample_rate)
        self.window_size = self.kernel_size + int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        
        self.optimizer = torch.optim.AdamW(self.nn.parameters(), self.learning_rate)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.train_losses = []
        self.valid_metrics = []
        self.steps = []
        
        # Optional: responses file for test set injections.
        self.responses_file = None
    
    def forward(self, X):
        return self.nn(X)
    
    def augment_noise2noise(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augments input data for a Noise2Noise training setup by injecting synthetic signals 
        into noise and generating corresponding labels. This function handles both cases 
        where precomputed responses are provided or where waveforms are generated on-the-fly.
        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, channels, window_size + psd_size].
                              The tensor contains noise data with an additional segment for 
                              computing the Power Spectral Density (PSD).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - X1 (torch.Tensor): The first augmented noisy input tensor of shape 
                                     [batch_size, 1, window_size].
                - X2 (torch.Tensor): The second augmented noisy input tensor of shape 
                                     [batch_size, 1, window_size].
                - labels (torch.Tensor): The corresponding clean signal labels of shape 
                                         [batch_size, 1, window_size].
        Raises:
            ValueError: If the dimensionality of the precomputed responses data is unsupported.
        Notes:
            - If a precomputed responses file is provided, the function uses it to inject 
              signals into the noise. The responses can be either 1D or 2D arrays.
            - If no responses file is provided, synthetic waveforms are generated, projected, 
              rescaled, and sampled to create the injected signals.
            - The function applies whitening to the noisy inputs and the labels using the PSD.
            - During the first global step, random samples from the batch are plotted and 
              saved to visualize the augmentation process.
        """
        background, X = torch.split(X,
                                    [self.psd_size, self.window_size],
                                    dim=-1)
        psd = self.spectral_density(background.double())
        batch_size = X.size(0)
        
        
        if hasattr(self, "responses_file") and self.responses_file is not None:
            with h5py.File(self.responses_file, "r") as f:
                # If there's a 2D L1 dataset, use that; otherwise use the 1D "data"
                if "injection_parameters" in f and "l1_signal_whitened" in f["injection_parameters"]:
                    responses_data = f["injection_parameters/l1_signal_whitened"][()]
                else:
                    responses_data = f["data"][:]
        
            # 1D case: exactly the original behavior
            if responses_data.ndim == 1:
                responses = torch.tensor(responses_data, device=self.device, dtype=torch.float32)
                L = responses.shape[0]
                window = 2048
                if L > window:
                    responses = responses[-window:]
                elif L < window:
                    pad_size = window - L
                    responses = F.pad(responses, (0, pad_size))
                responses = responses.unsqueeze(0).unsqueeze(0)  # [1,1,window_size]
                responses = responses.repeat(batch_size, 1, 1)   # [batch_size,1,window_size]
                mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            # 2D case: (n_samples, timesteps) → random row + window per batch
            elif responses_data.ndim == 2:
                num_samples, total_timesteps = responses_data.shape
                window = 2048
                segments = []
                sample_idxs = torch.randint(0, num_samples, (batch_size,), device=self.device)

                if total_timesteps >= window:
                    max_start = total_timesteps - window
                    start_idxs = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
                    for s_idx, start in zip(sample_idxs, start_idxs):
                        raw = responses_data[s_idx.item(), start.item(): start.item() + window]
                        seg = torch.from_numpy(raw).to(device=self.device)
                        segments.append(seg)
                else:
                    pad_size = window - total_timesteps
                    for s_idx in sample_idxs:
                        raw = responses_data[s_idx.item(), :]
                        padded = np.pad(raw, (0, pad_size), mode="constant")
                        seg = torch.from_numpy(padded).to(device=self.device)
                        segments.append(seg)

                responses = torch.stack(segments, dim=0).unsqueeze(1)  # [batch_size,1,window_size]
                mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                
                
                # Use the noise portion from the batch directly for the first copy.
                X1 = X.clone()
                indices = torch.randperm(batch_size)
                X2 = X[indices].clone()
        
                X1 = X1[:, 0:1, :]
                X2 = X2[:, 0:1, :]
            
                X1 = self.whitener(X1, psd)
                X2 = self.whitener(X2, psd)
            
                responses = responses/(2048**0.5)
            
                X1[mask] += responses.float()
                X2[mask] += responses.float()
    
                labels = torch.zeros_like(X1)
                labels[mask] = responses.float()
                
                
            else:
                raise ValueError(f"Unsupported responses_data.ndim = {responses_data.ndim}")

        
        else:
            
            # Generate the injection (clean signal) for the batch.
            hc, hp, mask = self.generate_waveforms(batch_size)
        
            responses = self.project_waveforms(hc, hp)
            responses = self.rescale_snrs(responses, psd[mask])
            responses = self.sample_waveforms(responses)
        
        
            # Use the noise portion from the batch directly for the first copy.
            X1 = X.clone()
            indices = torch.randperm(batch_size)
            X2 = X[indices].clone()
        
            X1[mask] += responses.float()
            X2[mask] += responses.float()
        
            X1 = X1[:, 0:1, :]
            X1 = self.whitener(X1, psd)
            X2 = X2[:, 0:1, :]
            X2 = self.whitener(X2, psd)
    
            labels = torch.zeros_like(X1)
            labels[mask] = self.whitener(responses, psd[mask])


        if self.global_step == 0:
            num_samples_to_plot = min(10, X1.size(0))
            idxs = np.random.choice(X1.size(0), num_samples_to_plot, replace=False)

            # Build a common time axis spanning 0–1.5 s for every trace
            time_axis = np.linspace(0.0, 1.0, X1.size(-1))

            os.makedirs("sample_plots_wavunet", exist_ok=True)
            for i in idxs:
                fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

                # ── Copy 1 ────────────────────────────────────────────────────────────────
                axs[0].plot(time_axis, X1[i, 0].cpu().numpy(), label="Noisy Input Copy 1")
                axs[0].plot(time_axis, labels[i, 0].cpu().numpy(),
                        label="Pure Waveform", linestyle="--")
                axs[0].set_ylabel("Amplitude")
#                axs[0].set_title(
#                    f"Noise2Noise Sample {i} – Copy 1  (Epoch {self.current_epoch}, Step {self.global_step})"
#                )
                axs[0].legend()
                axs[0].set_xlim(0.0, 1.0)

                # ── Copy 2 ────────────────────────────────────────────────────────────────
                axs[1].plot(time_axis, X2[i, 0].cpu().numpy(), label="Noisy Input Copy 2")
                axs[1].plot(time_axis, labels[i, 0].cpu().numpy(),
                            label="Pure Waveform", linestyle="--")
                axs[1].set_xlabel("Time (secs)")
                axs[1].set_ylabel("Amplitude")
#                axs[1].set_title(
#                    f"Noise2Noise Sample {i} – Copy 2  (Epoch {self.current_epoch}, Step {self.global_step})"
#                )
                axs[1].legend()
                axs[1].set_xlim(0.0, 1.0)

                fig.tight_layout()
                plot_path = os.path.join(
                    "sample_plots_wavunet",
                    f"n2n_sample_epoch{self.current_epoch}_step{self.global_step}_idx{i}.png",
                )
                plt.savefig(plot_path)
                plt.close(fig)

            print("Saved random Noise2Noise sample plots to 'sample_plots_wavunet/'")

        return X1, X2, labels


    
    # Updated augment_for_test: If a responses file is provided, load a single 1D responses array,
    # adjust its dimensions, and replicate it across the batch.
    def augment_for_test(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        background, X_signal = torch.split(X, [self.psd_size, self.window_size], dim=-1)
        psd = self.spectral_density(background.double())
        batch_size = X_signal.size(0)
    
        if hasattr(self, "responses_file") and self.responses_file is not None:
            with h5py.File(self.responses_file, "r") as f:
                # If there's a 2D L1 dataset, use that; otherwise use the 1D "data"
                if "injection_parameters" in f and "l1_signal_whitened" in f["injection_parameters"]:
                    responses_data = f["injection_parameters/l1_signal_whitened"][()]
                else:
                    responses_data = f["data"][:]
        
            # 1D case: exactly your original behavior
            if responses_data.ndim == 1:
                responses = torch.tensor(responses_data, device=self.device, dtype=torch.float32)
                L = responses.shape[0]
                window = 2048
                if L > window:
                    responses = responses[-window:]
                elif L < window:
                    pad_size = window - L
                    responses = F.pad(responses, (0, pad_size))
                responses = responses.unsqueeze(0).unsqueeze(0)  # [1,1,window_size]
                responses = responses.repeat(batch_size, 1, 1)   # [batch_size,1,window_size]
                mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            # 2D case: (n_samples, timesteps) → random row + window per batch
            elif responses_data.ndim == 2:
                num_samples, total_timesteps = responses_data.shape
                window = 2048
#                segments = []
                sample_idxs = torch.randint(0, num_samples, (batch_size,), device=self.device)

                if total_timesteps >= window:
                    max_start = total_timesteps - window
                    start_idxs = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
                    for s_idx, start in zip(sample_idxs, start_idxs):
                        raw = responses_data[s_idx.item(), start.item(): start.item() + window]
                        seg = torch.from_numpy(raw).to(device=self.device)
                        segments.append(seg)
                else:
                    pad_size = window - total_timesteps
                    for s_idx in sample_idxs:
                        raw = responses_data[s_idx.item(), :]
                        padded = np.pad(raw, (0, pad_size), mode="constant")
                        seg = torch.from_numpy(padded).to(device=self.device)
                        segments.append(seg)

                responses = torch.stack(segments, dim=0).unsqueeze(1)  # [batch_size,1,window_size]
                mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                
            else:
                raise ValueError(f"Unsupported responses_data.ndim = {responses_data.ndim}")
            
            responses = responses/(2048**0.5)
            
            X_signal_aug = X.clone()
            X_signal_aug = X_signal_aug[:, 0:1, :]
            X_whiten = self.whitener(X_signal_aug, psd)
            X_whiten[mask] += responses.float()
            labels = torch.zeros_like(X_whiten)
            labels[mask] = responses.float()
        
            
        elif hasattr(self, "polarizations_file") and self.polarizations_file is not None:
                
            with h5py.File(self.polarizations_file, "r") as f:
                if "hp" in f and "hc" in f:
                    hp = f["hp"][()]
                    hc = f["hc"][()]        
                        
                
            hp = torch.tensor(hp, device=self.device, dtype=torch.float32)
            hc = torch.tensor(hc, device=self.device, dtype=torch.float32)
                
            # pad to at least window_size
            pad_amount = self.window_size - hp.shape[-1]
            if pad_amount > 0:
                hp = F.pad(hp, (pad_amount, 0))
                hc = F.pad(hc, (pad_amount, 0))
                
            # repeat across the batch
            hp = hp.unsqueeze(0).repeat(batch_size, 1)   # [batch_size, window_size]
            hc = hc.unsqueeze(0).repeat(batch_size, 1)   # [batch_size, window_size]
            
            responses_scaled = self.rescale_snrs(responses, psd[mask])
            
            responses_sampled = self.sample_waveforms(responses_scaled)
            X_signal_aug = X_signal.clone()
            X_signal_aug[mask] += responses_sampled.float()
            X_signal_aug = X_signal_aug[:, 0:1, :]
            X_whiten = self.whitener(X_signal_aug, psd)
            labels = torch.zeros_like(X_whiten)
            labels[mask] = self.whitener(responses_sampled, psd[mask])
            

        else:
            hc, hp, mask = self.generate_waveforms(batch_size)
            responses = self.project_waveforms(hc, hp)    
            
            responses_scaled = self.rescale_snrs(responses, psd[mask])
            
            responses_sampled = self.sample_waveforms(responses_scaled)
            
            X_signal_aug = X_signal.clone()
            X_signal_aug[mask] += responses_sampled.float()
            X_signal_aug = X_signal_aug[:, 0:1, :]
            X_whiten = self.whitener(X_signal_aug, psd)
            labels = torch.zeros_like(X_whiten)
            labels[mask] = self.whitener(responses_sampled, psd[mask])
        
        return X_whiten, labels
    
    
    
    def weighted_mse(self, pred, targ, eps=1e-3):
        # normalize weights to [0,1]
        w = targ.abs()
        w = w / (w.max(dim=-1, keepdim=True)[0] + eps)
        # add a small floor so you still learn the small oscillations
        w = 0.1 + 0.9 * w  
    
        return ((w * (pred - targ)**2).mean())
    
    
    def training_step(self, batch):
        if isinstance(batch, (list, tuple)):
            X, y = batch
        else:
            X, y = batch, None

        if self.use_presaved:
            # Directly use provided data (X = noisy, y = clean)
            X1 = X.to(self.device)
            labels = y.to(self.device)
        else:
            # Use Noise2Noise augmentation
            X1, _, labels = self.augment_noise2noise(X.to(self.device))

        self.optimizer.zero_grad()
        
        mu, s_raw = self(X1)
        loss = gaussian_nll_tfp_style(mu, s_raw, labels, scale_factor=self.scale_factor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    
    def validation_step(self, batch):
        if isinstance(batch, (list, tuple)):
            X, y = batch
        else:
            X, y = batch, None

        if self.use_presaved:
            X1 = X.to(self.device)
            labels = y.to(self.device)
        else:
            X1, _, labels = self.augment_noise2noise(X.to(self.device))

        with torch.no_grad():
            
            mu, s_raw = self(X1)
            loss = gaussian_nll_tfp_style(mu, s_raw, labels, scale_factor=self.scale_factor)

        return loss.item()


    
    def log_metrics(self):
        with open(os.path.join(self.log_dir, 'metrics.csv'), 'a') as f:
            for step, loss in zip(self.steps, self.train_losses):
                f.write(f"{self.current_epoch},{step},{loss},,\n")
            for metric_val in self.valid_metrics:
                f.write(f"{self.current_epoch},{self.global_step},,,{metric_val}\n")
        self.steps = []
        self.train_losses = []
        self.valid_metrics = []
    
    def fit(self):
        if not os.path.exists(os.path.join(self.log_dir, 'metrics.csv')):
            with open(os.path.join(self.log_dir, 'metrics.csv'), 'w') as f:
                f.write("epoch,step,train_loss, ,valid_loss\n")
        train_dataloader = self.create_train_dataloader()
        val_dataloader = self.create_train_dataloader()
        total_steps = self.max_epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            pct_start=0.1,
            total_steps=total_steps
        )
        self.to(self.device)
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            self.nn.train()
            train_loop = tqdm(train_dataloader, desc="Training")
            for batch in train_loop:
                loss = self.training_step(batch)
                scheduler.step()
                self.global_step += 1
                self.steps.append(self.global_step)
                self.train_losses.append(loss)
                if self.global_step % 5 == 0:
                    train_loop.set_postfix(loss=f"{loss:.4f}")
            self.nn.eval()
            val_losses = []
            val_loop = tqdm(val_dataloader, desc="Validation")
            for batch in val_loop:
                loss = self.validation_step(batch)
                val_losses.append(loss)
            avg_val_loss = np.mean(val_losses)
            self.valid_metrics.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < self.best_metric:
                self.best_metric = avg_val_loss
                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_model_wavunet_new_3_150_epochs_1e-4_lr.pt'))
                print(f"New best model saved with Loss: {avg_val_loss:.4f}")
#            self.save_checkpoint(os.path.join(self.checkpoint_dir, f'epoch_{epoch+1}.pt'))
            self.log_metrics()
    
    
    def test_and_save_on_test_set(self):
        test_dataloader = self.create_test_dataloader()
        all_noisy, all_clean, all_recon, all_std, all_lower90, all_upper90 = [], [], [], [], [], []
        self.nn.eval()

        for noisy, clean in tqdm(test_dataloader, desc="Testing on Test Set"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            with torch.no_grad():
                mu, logvar = self.nn(noisy)
                var = torch.exp(logvar)
                std = torch.sqrt(var)

            lower90 = mu - 1.645 * std
            upper90 = mu + 1.645 * std

            all_noisy.append(noisy.cpu())
            all_clean.append(clean.cpu())
            all_recon.append(mu.cpu())
            all_std.append(std.cpu())
            all_lower90.append(lower90.cpu())
            all_upper90.append(upper90.cpu())

        # concatenate across batches
        noisy_all   = torch.cat(all_noisy,   dim=0)
        clean_all   = torch.cat(all_clean,   dim=0)
        recon_all   = torch.cat(all_recon,   dim=0)
        std_all     = torch.cat(all_std,     dim=0)
        l90_all     = torch.cat(all_lower90, dim=0)
        u90_all     = torch.cat(all_upper90, dim=0)

        # save everything, including the 90% intervals
        torch.save({
            'noisy':    noisy_all,
            'clean':    clean_all,
            'recon':    recon_all,
            'std':      std_all,
            'lower90':  l90_all,
            'upper90':  u90_all,
        }, "test_reconstruction_results_wavunet.pt")
        print("Test results saved to test_reconstruction_results_wavunet.pt")

        # plot first 10 examples with 90% band
        os.makedirs("test_plots_wavunet", exist_ok=True)
        num_to_plot = min(10, noisy_all.size(0))
        for idx in range(num_to_plot):
            fig, ax = plt.subplots()
            x = np.arange(recon_all.shape[-1])

#            ax.plot(noisy_all[idx,0].numpy(), label="Noisy Input", linewidth=1)
            ax.plot(clean_all[idx,0].numpy(), label="Clean", linestyle="--", linewidth=1)
            ax.plot(recon_all[idx,0].numpy(), label="Reconstruction Mean", linestyle=":", linewidth=1)

            # fill 90% uncertainty
            ax.fill_between(
                x,
                l90_all[idx,0].numpy(),
                u90_all[idx,0].numpy(),
                alpha=0.5,
                label="90% Uncertainty"
            )

            ax.legend()
            ax.set_title(f"Test Sample {idx}")
            plt.tight_layout()
            fig.savefig(os.path.join("test_plots_wavunet", f"test_sample_{idx}.png"))
            plt.close(fig)

        print("Saved first 10 test sample plots (with 90% bands) to 'test_plots_wavunet/'")

    
    
    def generate_waveforms(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rvs = torch.rand(size=(batch_size,), device=self.device)
        mask = rvs < self.waveform_prob
        num_injections = mask.sum().item()
        params = {k: v.sample((num_injections,)).to(self.device) for k, v in self.param_dict.items()}
        hc, hp = self.approximant(
            f=self.frequencies[self.freq_mask],
            f_ref=self.f_ref,
            **params
        )
        shape = (hc.shape[0], len(self.frequencies))
        hc_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hp_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hc_spectrum[:, self.freq_mask] = hc
        hp_spectrum[:, self.freq_mask] = hp
        hc = torch.fft.irfft(hc_spectrum) * self.sample_rate
        hp = torch.fft.irfft(hp_spectrum) * self.sample_rate
        ringdown_duration = 0.5
        ringdown_size = int(ringdown_duration * self.sample_rate)
        hc = torch.roll(hc, -ringdown_size, dims=-1)
        hp = torch.roll(hp, -ringdown_size, dims=-1)
        return hc, hp, mask
    
    def project_waveforms(self, hc: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
        N = len(hc)
        dec = torch.distributions.Uniform(-1, 1).sample((N,)).to(hc.device)
        psi = self.psi.sample((N,)).to(hc.device)
        phi = self.phi.sample((N,)).to(hc.device)
        from ml4gw import gw
        return gw.compute_observed_strain(
            dec=dec,
            psi=psi,
            phi=phi,
            detector_tensors=self.detector_tensors,
            detector_vertices=self.detector_vertices,
            sample_rate=self.sample_rate,
            cross=hc,
            plus=hp
        )
    
    def rescale_snrs(self, responses: torch.Tensor, psd: torch.Tensor) -> torch.Tensor:
        num_freqs = int(responses.size(-1) // 2) + 1
        
        # --- add a dummy channel so interpolate is happy --------------------
        added_channel = False
        if psd.dim() == 2:                     # [N, F] → [N, 1, F]
            psd = psd.unsqueeze(1)
            added_channel = True

        # interpolate only if necessary
        if psd.size(-1) != num_freqs:
            psd = torch.nn.functional.interpolate(
                psd, size=num_freqs, mode="linear", align_corners=False
            )

        # remove the channel we added, so downstream code sees [N, F]
        if added_channel:
            psd = psd.squeeze(1)
        # --------------------------------------------------------------------

        N = len(responses)
        target_snrs = self.snr.sample((N,)).to(responses.device)
        from ml4gw import gw
        return gw.reweight_snrs(
            responses=responses.double(), 
            target_snrs=target_snrs,
            psd=psd,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
        )
    
    def sample_waveforms(self, responses: torch.Tensor) -> torch.Tensor:
        responses = responses[:, :, -self.window_size:]
        pad = [0, int(self.window_size // 2)]
        responses = torch.nn.functional.pad(responses, pad)
        from ml4gw.utils.slicing import sample_kernels
        return sample_kernels(responses, self.window_size, coincident=True)
    
    
    def create_train_dataloader(self):
        sig = 22.116413
        # If pre-saved training dataset provided
        if hasattr(self, "train_hdf") and self.train_hdf is not None:
            with h5py.File(self.train_hdf, "r") as f:
                X = torch.Tensor(f["injection_samples"]["l1_strain"][:, ::2]/sig)
                y = torch.Tensor(f["injection_parameters"]["l1_signal_whitened"][:, ::2]/sig)
            X = X[:, None, :]
            y = y[:, None, :]
            dataset = TensorDataset(X, y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # Otherwise, fall back to sample generation
        from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
        samples_per_epoch = 3000
        batches_per_epoch = int((samples_per_epoch - 1) // self.batch_size) + 1
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1
        fnames = list(Path("/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data").iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )
        return ChunkedTimeSeriesDataset(
            dataset,
            kernel_size=self.window_size + self.psd_size,
            batch_size=self.batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=False
        )

       
    def create_val_dataloader(self):
        # Validation should always mirror test loader
        return self.create_test_dataloader()
    
    
    
    def create_test_dataloader(self):
        sig = 22.116413
        # If pre-saved test dataset provided
        if hasattr(self, "test_hdf") and self.test_hdf is not None:
            with h5py.File(self.test_hdf, "r") as f:
                X = torch.Tensor(f["injection_samples"]["l1_strain"][:, ::2]/sig)
                y = torch.Tensor(f["injection_parameters"]["l1_signal_whitened"][:, ::2]/sig)
                
            X = X[:, None, :]
            y = y[:, None, :]
            dataset = TensorDataset(X, y)
            return DataLoader(dataset, batch_size=self.batch_size * 4, shuffle=False, pin_memory=True)

        # Otherwise, fall back to generated test dataset
        if not os.path.exists("test_dataset_wavunet.hdf5"):
            print("Creating test dataset...")
            self._create_test_dataset(num_samples=100)

        with h5py.File("test_dataset_wavunet.hdf5", "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])
        X = X[:, 0:1, :]
        y = y[:, 0:1, :]
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size * 4, shuffle=False, pin_memory=True)

    def _create_test_dataset(self, num_samples=50):
        new_dataloader = self.create_new_dataloader()
        X_list, y_list = [], []
        total_samples = 0
        for batch in new_dataloader:
            batch = batch.to(self.device)
            X_whiten, labels = self.augment_for_test(batch)
            X_list.append(X_whiten.cpu())
            y_list.append(labels.cpu())
            total_samples += X_whiten.size(0)
            if total_samples >= num_samples:
                break
        X_test = torch.cat(X_list, dim=0)[:num_samples]
        y_test = torch.cat(y_list, dim=0)[:num_samples]
        with h5py.File("test_dataset_wavunet.hdf5", "w") as f:
            f.create_dataset("X", data=X_test.numpy())
            f.create_dataset("y", data=y_test.numpy())
        print("Test dataset saved to test_dataset_wavunet.hdf5")
    
    
    def create_new_dataloader(self):
        from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
        samples_per_epoch = 3000
        batches_per_epoch = int((samples_per_epoch - 1) // self.batch_size) + 1
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1
        fnames = list(Path("/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data_test").iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )
        from ml4gw.dataloading import ChunkedTimeSeriesDataset
        return ChunkedTimeSeriesDataset(
            dataset,
            kernel_size=self.window_size + self.psd_size,
            batch_size=self.batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=False
        )
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']


# =============================================================================
# Main Script with argparse
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train_hdf",
        type=str,
        default=None,
        help="Path to HDF5 file containing pre-saved training data (X,y)."
    )
    
    parser.add_argument(
        "--test_hdf",
        type=str,
        default=None,
        help="Path to HDF5 file containing pre-saved test data (X,y). Will also be used for validation."
    )

    parser.add_argument(
        "--responses_file",
        type=str,
        default=None,
        help="Path to an HDF file containing the 'responses' data for generating the test set. "
             "This file should contain a single 1D numpy array stored under key 'responses'."
    )
    parser.add_argument(
        "--polarizations_file",
        type=str,
        default=None,
        help="Path to an HDF file containing the polarizations data for generating the test set. "
             "This file should contain a single 1D numpy array stored under key 'hp' and 'hc'."
    )
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint (e.g., checkpoints_wavunet/best_model_wavunet.pt)")
    
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    architecture = WavUNetAttention(
        in_channels=1,
        base_filters=64,
        num_levels=3,
        device=device,
    ).to(device)


    model = Ml4gwReconstructionModel(
        architecture=architecture,
        device=device,
        kernel_length=1.0,           # seconds per training window
        fduration=2,
        psd_length=16,
        sample_rate=1024,
        highpass=20,
        min_snr=8, max_snr=40,      
    ).to(device)
    
    
    if args.train_hdf is not None:
        model.train_hdf = args.train_hdf
        model.use_presaved = True
        print(f"Using pre-saved training dataset: {args.train_hdf}")

    if args.test_hdf is not None:
        model.test_hdf = args.test_hdf
        model.use_presaved = True
        print(f"Using pre-saved test/validation dataset: {args.test_hdf}")


    # If a responses file was provided, assign it.
    if args.responses_file is not None:
        model.responses_file = args.responses_file
        print(f"Using provided responses file: {args.responses_file}")
        
    # If a responses file was provided, assign it.
    if args.polarizations_file is not None:
        model.polarizations_file = args.polarizations_file
        print(f"Using provided polarizations file: {args.polarizations_file}")
        

    # ALWAYS create test dataset if it does not exist.
    if not os.path.exists("test_dataset_wavunet.hdf5"):
        print("Creating test dataset...")
        def create_test_dataset(model, num_samples=50):
            new_dataloader = model.create_new_dataloader()
            X_list, y_list = [], []
            total_samples = 0
            for batch in new_dataloader:
                batch = batch.to(model.device)
                X_whiten, labels = model.augment_for_test(batch)
                X_list.append(X_whiten.cpu())
                y_list.append(labels.cpu())
                total_samples += X_whiten.size(0)
                if total_samples >= num_samples:
                    break
            X_test = torch.cat(X_list, dim=0)[:num_samples]
            y_test = torch.cat(y_list, dim=0)[:num_samples]
            with h5py.File("test_dataset_wavunet.hdf5", "w") as f:
                f.create_dataset("X", data=X_test.numpy())
                f.create_dataset("y", data=y_test.numpy())
            print("Test dataset saved to test_dataset_wavunet.hdf5")
        create_test_dataset(model, num_samples=100)
        
        
    # ---- Load checkpoint if given ----
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)
    else:
        print("No checkpoint provided. Training from scratch.")

    model.fit()
    model.test_and_save_on_test_set()
