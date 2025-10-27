# --- Environment (keep if HF/SSL issues crop up) --------------------------------
import os, certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# --- Imports --------------------------------------------------------------------
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
from pycbc.filter import overlap
from pycbc.types import TimeSeries

from train_AWaRe_IMBH import WavUNetAttention, Ml4gwReconstructionModel

# --- Constants ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 22.116413
SAMPLE_RATE = 1024

# Input files
PATH_1 = "/home/chattec/samplegen/output/GW231123_injection_test_random_glitches.hdf"
PATH_2 = "/home/chattec/LIGO/Wav2Vec2/Waveform_reconstruction/Eccentricity_data/O3b_test_EccentricTD_convergence_mass-30_SNR-12.hdf"
PATH_3 = "/data/p_dsi/ligo/IMBH_data/IMBH_val_GW231123_injection_test.hdf"

# Checkpoint
CKPT = "checkpoints_wavunet/best_model_wavunet_new_3_150_epochs_1e-4_lr.pt"

# --- Model ----------------------------------------------------------------------
def build_model(device=DEVICE):
    # Swap architecture by commenting the one you don't use
    arch = WavUNetAttention(
        in_channels=1, base_filters=64, num_levels=3, device=device
    ).to(device)

    model = Ml4gwReconstructionModel(architecture=arch, device=device, sample_rate=SAMPLE_RATE).to(device)
    checkpoint = torch.load(CKPT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {CKPT}")
    return model

# --- Data helpers ----------------------------------------------------------------
def load_strain_and_signal(path, det="l1", halve=False, scale=SCALE):
    """Return (strain, signal) as np.float64 arrays, scaled by 'scale'."""
    with h5py.File(path, "r") as f:
        x = f["injection_samples"][f"{det}_strain"][()]
        y = f["injection_parameters"][f"{det}_signal_whitened"][()]
        if halve:
            x = x[:, ::2]
            y = y[:, ::2]
    return x / scale, y / scale

def make_loader_from_arrays(x_np, y_np, batch_size=32, device=DEVICE):
    X = torch.tensor(x_np, dtype=torch.float32)[:, None, :]  # [N, 1, T]
    Y = torch.tensor(y_np, dtype=torch.float32)[:, None, :]
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)

def forward_pass(model, loader, device=DEVICE):
    mus, stds = [], []
    with torch.no_grad():
        for noisy, _ in loader:
            noisy = noisy.to(device)
            mu, s_raw = model.nn(noisy)  # heteroscedastic head
            sigma = torch.nn.functional.softplus(s_raw) * torch.exp(model.scale_factor)
            mus.append(mu.cpu())
            stds.append(sigma.cpu())
    mu_all = torch.cat(mus, dim=0).squeeze(1)   # [N, T]
    std_all = torch.cat(stds, dim=0).squeeze(1) # [N, T]
    return mu_all.numpy(), std_all.numpy()

def read_snr(path, det):
    with h5py.File(path, "r") as f:
        snr = f["injection_parameters"][f"{det}_snr"][()]
        scale_factor = f["injection_parameters"]["scale_factor"][()]
    return snr * scale_factor

# --- Coverage utilities ----------------------------------------------------------
def coverage_vs_nominal(mu, sigma, y_true, coverages=(10, 50, 90), eps=1e-9):
    mu = np.asarray(mu)
    sigma = np.maximum(np.asarray(sigma), eps)
    y_true = np.asarray(y_true)

    nominals, empirical = [], []
    for c in coverages:
        z = norm.ppf((1 + c/100.0) / 2.0)
        inside = (y_true >= (mu - z*sigma)) & (y_true <= (mu + z*sigma))
        nominals.append(c/100.0)
        empirical.append(inside.mean())
    return np.array(nominals), np.array(empirical)

def plot_cov_curve(nominals, empirical, outfile="Coverage_plot.png"):
    plt.figure(figsize=(8, 6))
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, xs, linestyle="--")
    plt.scatter(nominals, empirical)
    for x, y in zip(nominals, empirical):
        plt.text(x, y, f" {int(round(x*100))}%", va="bottom", ha="left", fontsize=14)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=400)
    plt.close()

def inflation_lambda(y, mu, sigma, c=90, eps=1e-9):
    z = np.abs((y - mu) / np.maximum(sigma, eps))
    z_emp = np.quantile(z, c/100.0)
    z_th = norm.ppf((1 + c/100.0) / 2.0)
    return float(z_emp / z_th)

# --- Overlap utilities -----------------------------------------------------------
def batch_overlap(all_recons, pure_waveforms, delta_t=1.0/1024, f_low=20, psd=None):
    N = all_recons.shape[0]
    ovs = np.zeros(N, dtype=np.float32)
    for i in range(N):
        rec_ts  = TimeSeries(all_recons[i].astype(np.float32), delta_t=delta_t)
        pure_ts = TimeSeries(pure_waveforms[i].astype(np.float32), delta_t=delta_t)
        ovs[i] = overlap(rec_ts, pure_ts, psd=psd, low_frequency_cutoff=f_low)
    return ovs

# --- Main ------------------------------------------------------------------------
def main():
    model = build_model()

    # ----------------- Coverage computation -----------------------------
    x1_l1, y1_l1 = load_strain_and_signal(PATH_1, det="l1", halve=False)
    x2_l1, y2_l1 = load_strain_and_signal(PATH_2, det="l1", halve=True)   # ::2 in original
    x3_l1, y3_l1 = load_strain_and_signal(PATH_3, det="l1", halve=False)

    y_true_cov = np.concatenate([y1_l1, y2_l1, y3_l1], axis=0)

    loader1 = make_loader_from_arrays(x1_l1, y1_l1, batch_size=32)
    loader2 = make_loader_from_arrays(x2_l1, y2_l1, batch_size=32)
    loader3 = make_loader_from_arrays(x3_l1, y3_l1, batch_size=128)

    mu1, std1 = forward_pass(model, loader1)
    mu2, std2 = forward_pass(model, loader2)
    mu3, std3 = forward_pass(model, loader3)

    mu_cov = np.concatenate([mu1, mu2, mu3], axis=0)
    std_cov = np.concatenate([std1, std2, std3], axis=0)

    lam = inflation_lambda(y_true_cov, mu_cov, std_cov, c=80)
    std_cov_infl = lam * std_cov

    nom, emp = coverage_vs_nominal(mu_cov, std_cov_infl, y_true_cov, coverages=[10, 50, 90])
    plot_cov_curve(nom, emp, outfile="Coverage_plot.png")

    
    # H1
    x1_h1, y1_h1 = load_strain_and_signal(PATH_1, det="h1", halve=False)
    x2_h1, y2_h1 = load_strain_and_signal(PATH_2, det="h1", halve=False)
    x3_h1, y3_h1 = load_strain_and_signal(PATH_3, det="h1", halve=False)

    loader1_h1 = make_loader_from_arrays(x1_h1, y1_h1, batch_size=32)
    loader2_h1 = make_loader_from_arrays(x2_h1, y2_h1, batch_size=32)
    loader3_h1 = make_loader_from_arrays(x3_h1, y3_h1, batch_size=128)

    mu1_h1, _ = forward_pass(model, loader1_h1)
    mu2_h1, _ = forward_pass(model, loader2_h1)
    mu3_h1, _ = forward_pass(model, loader3_h1)


    y_true_h1 = np.concatenate([y1_h1, y2_h1, y3_h1], axis=0)
    mu_h1     = np.concatenate([mu1_h1, mu2_h1, mu3_h1], axis=0)
    snr_h1    = np.concatenate([read_snr(PATH_1, "h1"), read_snr(PATH_2, "h1"), read_snr(PATH_3, "h1")], axis=0)
    data_h1   = np.concatenate([x1_h1, x2_h1, x3_h1], axis=0)

    ovs_h1 = batch_overlap(mu_h1, y_true_h1)

    # L1
    x1_l1, y1_l1 = load_strain_and_signal(PATH_1, det="h1", halve=False)
    x2_l1, y2_l1 = load_strain_and_signal(PATH_2, det="h1", halve=False)
    x3_l1, y3_l1 = load_strain_and_signal(PATH_3, det="h1", halve=False)

    loader1_l1 = make_loader_from_arrays(x1_l1, y1_l1, batch_size=32)
    loader2_l1 = make_loader_from_arrays(x2_l1, y2_l1, batch_size=32)
    loader3_l1 = make_loader_from_arrays(x3_l1, y3_l1, batch_size=128)

    mu1_l1, _ = forward_pass(model, loader1_l1)
    mu2_l1, _ = forward_pass(model, loader2_l1)
    mu3_l1, _ = forward_pass(model, loader3_l1)

    y_true_l1 = np.concatenate([y1_l1, y2_l1, y3_l1], axis=0)
    mu_l1     = np.concatenate([mu1_l1, mu2_l1, mu3_l1], axis=0)
    snr_l1    = np.concatenate([read_snr(PATH_1, "l1"), read_snr(PATH_2, "l1"), read_snr(PATH_3, "l1")], axis=0)
    data_l1   = np.concatenate([x1_l1, x2_l1, x3_l1], axis=0)

    ovs_l1 = batch_overlap(mu_l1, y_true_l1)


    # Save
    out_hdf = "Overlaps_injection_test_IMBH_WavUNet.hdf"
    with h5py.File(out_hdf, "w") as f:
        f.create_dataset('overlaps_H1', data=ovs_h1)
        f.create_dataset('overlaps_L1', data=ovs_l1)
        f.create_dataset('snr_H1', data=snr_h1)
        f.create_dataset('snr_L1', data=snr_l1)
        f.create_dataset('y_true_H1', data=y_true_h1)
        f.create_dataset('y_true_L1', data=y_true_l1)
        f.create_dataset('mu_cal_H1', data=mu_h1)
        f.create_dataset('mu_cal_L1', data=mu_l1)
        f.create_dataset('data_H1', data=data_h1)
        f.create_dataset('data_L1', data=data_l1)

    print(f"Saved overlaps and metadata to {out_hdf}")

if __name__ == "__main__":
    main()
