#!/usr/bin/env python
# Copyright 2022 Ondřej Zelenka
# Licensed under the Apache License, Version 2.0

import os
import sys
import time as t
import logging
import multiprocessing as mp
from argparse import ArgumentParser
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import pycbc
from pycbc.types import TimeSeries, FrequencySeries

from ml4gw.transforms import QScan

# Local imports
from GW_Whisper_ml4gw_train import GWWhisperMultiDetector, GWWhisperModel

# Optional: fix SSL roots on clusters
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# ----------------------- Torch / device -----------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

DTYPE = torch.float32

# ----------------------- Whitening ----------------------------------------------
def whiten(
    strain: np.ndarray,
    delta_t: float = 1.0 / 2048.0,
    segment_duration: float = 0.5,
    max_filter_duration: float = 0.25,
    trunc_method: str = "hann",
    remove_corrupted: bool = True,
    low_frequency_cutoff: float | None = None,
    psd=None,
    return_psd: bool = False,
    **kwargs,
):
    """Whiten a 1D or 2D numpy strain array using PyCBC."""
    if strain.ndim == 1:
        from pycbc.psd import inverse_spectrum_truncation, interpolate

        colored_ts = TimeSeries(strain, delta_t=delta_t)
        if psd is None:
            psd = colored_ts.psd(segment_duration, **kwargs)
        elif isinstance(psd, np.ndarray):
            # Infer delta_f from length; assumes even length
            assumed_duration = delta_t * (2 * len(psd) - 2)
            psd = FrequencySeries(psd, delta_f=1.0 / assumed_duration)
        elif isinstance(psd, FrequencySeries):
            pass
        else:
            raise ValueError("Unknown PSD format.")

        unprocessed_psd = psd
        psd = interpolate(psd, colored_ts.delta_f)
        max_filter_len = int(max_filter_duration * colored_ts.sample_rate)

        psd = inverse_spectrum_truncation(
            psd,
            max_filter_len=max_filter_len,
            low_frequency_cutoff=low_frequency_cutoff,
            trunc_method=trunc_method,
        )

        inv_psd = 1.0 / psd
        white_ts = (colored_ts.to_frequencyseries() * inv_psd ** 0.5).to_timeseries().numpy()

        if remove_corrupted:
            white_ts = white_ts[max_filter_len // 2 : (len(colored_ts) - max_filter_len // 2)]

        return (white_ts, unprocessed_psd) if return_psd else white_ts

    if strain.ndim == 2:
        # Broadcast PSD appropriately across channels
        if isinstance(psd, np.ndarray) and psd.ndim == 1:
            psds = [psd for _ in strain]
        elif psd is None or isinstance(psd, FrequencySeries):
            psds = [psd for _ in strain]
        else:
            assert len(psd) == len(strain)
            psds = psd

        pieces = [
            whiten(
                sd,
                delta_t=delta_t,
                segment_duration=segment_duration,
                max_filter_duration=max_filter_duration,
                trunc_method=trunc_method,
                remove_corrupted=remove_corrupted,
                low_frequency_cutoff=low_frequency_cutoff,
                psd=p_i,
                return_psd=return_psd,
                **kwargs,
            )
            for sd, p_i in zip(strain, psds)
        ]
        if return_psd:
            psd_list = [p[1] for p in pieces]
            w = np.stack([p[0] for p in pieces], axis=0)
            return w, psd_list
        return np.stack(pieces, axis=0)

    raise ValueError("Strain must be 1D or 2D numpy array.")


# ----------------------- Data slicing -------------------------------------------
class SegmentSlicer:
    def __init__(
        self,
        infile: h5py.File,
        key: str,
        step_size: float = 0.1,
        peak_offset: float = 0.6,
        slice_length: int = 2048,
        detectors: list[str] | None = None,
        white: bool = False,
        whitened_file: str | None = None,
        save_psd: bool = False,
        low_frequency_cutoff: float | None = None,
        segment_duration: float = 0.5,
        max_filter_duration: float = 0.25,
    ):
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors
        self.white = white
        self.whitened_file = whitened_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        self.key = key

        # delta_t stored in dataset attrs
        self.delta_t = float(infile[self.detectors[0]][key].attrs["delta_t"])
        self.index_step_size = int(self.step_size / self.delta_t)
        self.time_step_size = self.delta_t * self.index_step_size

        self.dss = [infile[det][self.key] for det in self.detectors]
        self.start_time = self.dss[0].attrs["start_time"]
        for ds in self.dss:
            assert ds.attrs["start_time"] == self.start_time

        logging.debug(
            "SegmentSlicer initialized with index_step_size=%i, time_step_size=%.6f, key=%s dtypes=%s",
            self.index_step_size,
            self.time_step_size,
            self.key,
            self.dss[0].dtype,
        )
        self.process(save_psd)

    def process(self, save_psd: bool):
        whitened = []
        self.psds = []
        for ds, det in zip(self.dss, self.detectors):
            if self.white:
                new_ds = ds[()]
            else:
                res = whiten(
                    ds,
                    delta_t=self.delta_t,
                    low_frequency_cutoff=self.low_frequency_cutoff,
                    segment_duration=self.segment_duration,
                    max_filter_duration=self.max_filter_duration,
                    return_psd=save_psd,
                )
                if save_psd:
                    new_ds, psd = res
                    self.psds.append(psd)
                else:
                    new_ds = res
            whitened.append(new_ds)
            if self.whitened_file is not None:
                with h5py.File(self.whitened_file, "a") as wfile:
                    wfile[det].create_dataset(self.key, data=new_ds)

        self.dss = np.stack(whitened, axis=0)
        if not self.white:
            # accounts for whitening filter group delay
            self.start_time += 0.125
        self.white = True

    def __len__(self):
        full_len = self.slice_length if self.white else (512 + self.slice_length)
        return 1 + ((self.dss.shape[1] - full_len) // self.index_step_size)

    def __iter__(self):
        self.current_index = 0
        self.current_time = self.start_time
        return self

    def get_next_slice(self):
        if self.current_index + self.slice_length > self.dss.shape[1]:
            raise StopIteration
        this_slice = self.dss[:, self.current_index : self.current_index + self.slice_length]
        this_time = self.current_time + self.peak_offset
        self.current_index += self.index_step_size
        self.current_time += self.time_step_size
        return this_slice, this_time

    def __next__(self):
        return self.get_next_slice()

    def split_and_pop(self, ds_dict, size: int):
        data = self.dss
        self.dss = None
        max_split_index = int(np.floor(data.shape[1] / size)) * size
        split_indices = list(range(size, max_split_index + 1, size))
        self.ds_dict_keys = []
        for i, ds in enumerate(np.split(data, split_indices, axis=1)):
            new_key = f"{self.key}_{i}"
            self.ds_dict_keys.append(new_key)
            logging.debug("Saving data %s to shared dictionary", new_key)
            ds_dict[new_key] = ds
        logging.debug("Worker %s finished!", self.key)

    def stack_and_load(self, ds_dict):
        dss = [ds_dict.pop(k) for k in self.ds_dict_keys]
        self.dss = np.concatenate(dss, axis=1)


class TorchSegmentSlicer(SegmentSlicer, torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        torch.utils.data.IterableDataset.__init__(self)
        SegmentSlicer.__init__(self, *args, **kwargs)

    def __next__(self):
        sl, tm = self.get_next_slice()
        sl = torch.from_numpy(sl).contiguous().to(dtype=torch.float32, device="cpu")
        tm = torch.as_tensor(tm, dtype=torch.float32, device="cpu")
        return sl, tm


# ----------------------- Q-transform -> Whisper adapter --------------------------
class QTransformAdapter(nn.Module):
    """Makes Q-scans Whisper-compatible and adds simple FiLM per-detector."""

    def __init__(
        self,
        kernel_length: float = 1.0,
        sample_rate: int = 2048,
        q_range: List[int] = [4, 128],
        spectrogram_shape: List[int] = [512, 512],
        target_shape: Tuple[int, int] = (80, 3000),
        n_detectors: int = 2,
    ):
        super().__init__()
        self.n_detectors = n_detectors
        self.q_transform = QScan(
            duration=kernel_length, sample_rate=sample_rate, spectrogram_shape=spectrogram_shape, qrange=q_range
        ).to(DEVICE)

        self.freq_adapter = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )
        self.final_pool = nn.AdaptiveAvgPool2d(target_shape)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.film_gamma = nn.Parameter(torch.ones(self.n_detectors))
        self.film_beta = nn.Parameter(torch.zeros(self.n_detectors))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        B, D, _ = x.shape
        out = []
        for i in range(D):
            qspec = self.q_transform(x[:, i]).unsqueeze(1)  # [B,1,F,T]
            y = self.freq_adapter(qspec)
            y = self.final_pool(y).squeeze(1)  # [B,F',T']
            y = self.scale * y + self.bias
            y = y * self.film_gamma[i] + self.film_beta[i]
            out.append(y)
        return torch.stack(out, dim=1)  # [B,D,F',T']


# ----------------------- GW-Whisper inference head (USR) -------------------------
def build_gw_whisper_from_checkpoint_usr(
    checkpoint_path: str,
    device: str = DEVICE,
    n_detectors: int = 2,
    whisper_model_name: str = "openai/whisper-tiny",
    kernel_length: float = 1.0,
    sample_rate: int = 1024,
    q_range=(4, 128),
    spectrogram_shape=(128, 128),
):
    """Return a module that emits USR scores (logit(p0) - logit(p1))."""
    arch = GWWhisperMultiDetector(
        n_detectors=n_detectors,
        whisper_model_name=whisper_model_name,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        q_range=list(q_range),
        spectrogram_shape=list(spectrogram_shape),
        output_dim=2,
    ).to(device)

    model = GWWhisperModel(
        architecture=arch,
        ifos=["H1", "L1"][:n_detectors],
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        learning_rate=1e-3,
        batch_size=128,
        max_epochs=1,
        checkpoint_dir="checkpoints_gw_whisper",
        log_dir="logs_gw_whisper",
        device=device,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class USRHead(nn.Module):
        def __init__(self, base, eps: float = 1e-6):
            super().__init__()
            self.base = base
            self.eps = eps

        def forward(self, x):
            with torch.no_grad():
                p = self.base(x).float()  # [B,2], probabilities
                p = torch.clamp(p, self.eps, 1.0 - self.eps)
                logit = torch.log(p) - torch.log(1.0 - p)
                s_usr = logit[:, 0] - logit[:, 1]  # class-0 minus class-1
                return s_usr.unsqueeze(1)  # [B,1]

    return USRHead(model).to(device)


# ----------------------- Worker / evaluation ------------------------------------
def worker(inp):
    fpath = inp.pop("fpath")
    key = inp.pop("key")
    wdata_dict = inp.pop("wdata_dict", None)
    with h5py.File(fpath, "r") as infile:
        slicer = TorchSegmentSlicer(infile, key, **inp)
    if wdata_dict is not None:
        slicer.split_and_pop(wdata_dict, 10**6)
    return slicer


def evaluate_slices(
    slicer: TorchSegmentSlicer,
    Network: nn.Module,
    device: str = DEVICE,
    trigger_threshold: float = 0.2,
    verbose: bool = False,
):
    """Run the model over all slices and collect triggers + raw scores."""
    new_triggers = []
    all_vals = []

    dl = DataLoader(
        slicer,
        batch_size=256,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
        shuffle=False,
    )

    batch_size = 256
    num_batches = (len(slicer) + batch_size - 1) // batch_size
    logging.info("Starting evaluation for %s (%d slices → %d batches)", slicer.key, len(slicer), num_batches)

    with torch.no_grad():
        for slice_batch, slice_times in tqdm(
            dl,
            desc=f"Evaluating {slicer.key}",
            total=num_batches,
            leave=False,
            disable=not verbose,
        ):
            slice_batch = slice_batch.to(device, dtype=DTYPE)

            # downsample (every other sample) and normalize
            slice_batch = slice_batch[:, :, ::2] / 22.116413

            output_values = Network(slice_batch)[:, 0]  # [B]
            all_vals.append(output_values.cpu().numpy())

            trig = output_values > trigger_threshold
            for t0, is_trig, val in zip(slice_times, trig, output_values):
                if is_trig.item():
                    new_triggers.append([float(t0), float(val)])

    return new_triggers, all_vals


# ----------------------- Clustering ---------------------------------------------
def get_clusters(triggers: dict[str, list], cluster_threshold: float = 0.35):
    """Cluster triggers by proximity; return maxima per cluster."""
    # Flatten per-key lists
    all_clusters = []
    for lst in triggers.values():
        clusters = []
        for tt, val in lst:
            if not clusters or (tt - clusters[-1][-1][0]) > cluster_threshold:
                clusters.append([[tt, val]])
            else:
                clusters[-1].append([tt, val])
        all_clusters.extend(clusters)

    logging.info(
        "Clustering produced %d independent triggers. Centering triggers at maxima.", len(all_clusters)
    )

    times, values, timevars = [], [], []
    for cluster in all_clusters:
        arr = np.asarray(cluster, dtype=float)  # [n,2] -> (time, val)
        i_max = int(np.argmax(arr[:, 1]))
        times.append(arr[i_max, 0])
        values.append(arr[i_max, 1])
        timevars.append(0.2)  # fixed timing uncertainty

    return np.asarray(times), np.asarray(values), np.asarray(timevars)


# ----------------------- Orchestration ------------------------------------------
def get_triggers(
    checkpoint_path: str,
    coincident: bool,  # kept for CLI compatibility
    inputfile: str,
    step_size: float = 0.1,
    trigger_threshold: float = 0.0,
    device: str = DEVICE,
    verbose: bool = False,
    white: bool = False,
    whitened_file: str | None = None,
    low_frequency_cutoff: float = 15.0,
    num_workers: int = -1,
    n_detectors: int = 2,
    sample_rate: int = 1024,
):
    if num_workers < 0:
        num_workers = mp.cpu_count()

    if whitened_file is not None:
        with h5py.File(whitened_file, "w") as wfile:
            for det in ("H1", "L1"):
                wfile.create_group(det)

    logging.debug("Initializing GW-Whisper USR model from checkpoint.")
    Network = build_gw_whisper_from_checkpoint_usr(
        checkpoint_path=checkpoint_path,
        device=device,
        n_detectors=n_detectors,
        whisper_model_name="openai/whisper-tiny",
        kernel_length=1.0,
        sample_rate=sample_rate,
        q_range=(4, 128),
        spectrogram_shape=(128, 128),
    ).to(dtype=DTYPE, device=device)
    Network.eval()

    arguments = []
    with h5py.File(inputfile, "r") as infile:
        det_grp = next(iter(infile.values()))
        for key in det_grp.keys():
            arguments.append(
                dict(
                    fpath=inputfile,
                    key=key,
                    step_size=step_size,
                    low_frequency_cutoff=low_frequency_cutoff,
                    white=white,
                    whitened_file=whitened_file,
                    detectors=["H1", "L1"],
                )
            )
        # process longest first
        arguments.sort(key=lambda x: len(infile[x["detectors"][0]][x["key"]]), reverse=True)

    triggers = {}

    if num_workers > 0:
        try:
            mp.set_start_method("forkserver")
        except RuntimeError:
            pass  # already set

        wdata_dict = None
        if whitened_file is not None:
            m = mp.Manager()
            wdata_dict = m.dict()
            for tmp in arguments:
                tmp["wdata_dict"] = wdata_dict

        with mp.Pool(num_workers) as pool:
            for slicer in tqdm(
                pool.imap_unordered(worker, arguments),
                disable=not verbose,
                ascii=True,
                total=len(arguments),
            ):
                if wdata_dict is not None:
                    slicer.stack_and_load(wdata_dict)

                triggers[slicer.key], all_vals = evaluate_slices(
                    slicer,
                    Network,
                    device=device,
                    trigger_threshold=trigger_threshold,
                    verbose=verbose,
                )

    else:
        for kwargs in tqdm(arguments, disable=not verbose, ascii=True):
            slicer = worker(kwargs)
            triggers[slicer.key], all_vals = evaluate_slices(
                slicer,
                Network,
                device=device,
                trigger_threshold=trigger_threshold,
                verbose=verbose,
            )

    triggers = dict(sorted(triggers.items(), key=lambda kv: kv[0]))
    return triggers, all_vals


# ----------------------- CLI -----------------------------------------------------
def main():
    start_time = t.time()

    p = ArgumentParser(description="GW search with GW-Whisper USR scoring.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--inputfile", type=str, required=True)
    p.add_argument("--outputfile", type=str, required=True)
    p.add_argument("--white", action="store_true")
    p.add_argument("--softmax", action="store_true")  # kept for CLI compat; ignored
    p.add_argument("--coincident", action="store_true")
    p.add_argument("--coinc-window", type=float, default=0.1)

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n-detectors", type=int, default=2)
    p.add_argument("--sample-rate", type=int, default=1024)
    p.add_argument("-t", "--trigger-threshold", type=float, default=12.4)

    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--cluster-threshold", type=float, default=0.35)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--debug-triggers-file", type=str, default=None)
    p.add_argument("--debug-whitened-file", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=8)

    args = p.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=log_level,
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    print("Starting the process…", flush=True)

    # Output sanity checks
    if os.path.isfile(args.outputfile) and not args.force:
        raise RuntimeError("Output file exists.")

    for path in (args.debug_triggers_file, args.debug_whitened_file):
        if path is not None and os.path.isfile(path) and not args.force:
            raise RuntimeError(f"Debug file exists: {path}")

    triggers, all_vals = get_triggers(
        checkpoint_path=args.checkpoint,
        coincident=args.coincident,
        inputfile=args.inputfile,
        step_size=args.step_size,
        trigger_threshold=args.trigger_threshold,
        device=args.device,
        verbose=args.verbose,
        white=args.white,
        whitened_file=args.debug_whitened_file,
        num_workers=args.num_workers,
        low_frequency_cutoff=20.0,
        n_detectors=args.n_detectors,
        sample_rate=args.sample_rate,
    )

    n_exceeded = sum(len(v) for v in triggers.values())
    logging.info("A total of %i samples exceeded the threshold of %.3f", n_exceeded, args.trigger_threshold)

    if args.debug_triggers_file is not None:
        with h5py.File(args.debug_triggers_file, "w") as f:
            for key, lst in triggers.items():
                f.create_dataset(key, data=np.asarray(lst, dtype=np.float32))

    times, stats, vars_ = get_clusters(triggers, args.cluster_threshold)
    all_vals_flat = np.concatenate(all_vals).astype("float32")

    with h5py.File(args.outputfile, "w") as outfile:
        logging.debug("Saving clustered triggers into %s.", args.outputfile)
        outfile.create_dataset("time", data=times)
        outfile.create_dataset("stat", data=stats)
        outfile.create_dataset("var", data=vars_)
        outfile.create_dataset("all_vals", data=all_vals_flat)
        logging.debug("Triggers saved, closing file.")

    total = t.time() - start_time
    print(f"Total execution time: {total:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
