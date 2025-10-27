import torch, h5py, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from torch import special

import pylab
import seaborn as sns
sns.set_context('talk') 


sns.set_theme(font_scale=2)
sns.set_palette('colorblind')
sns.set_style('ticks')

pylab.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'axes.grid' : True,
        'grid.linestyle' : '--',
        'grid.color' : '#bbbbbb'

    }
)

pylab.rcParams['axes.linewidth'] = 1


# -----------------------------------------------------------------------------
# 1. Import architecture
# -----------------------------------------------------------------------------
from GW_Whisper_ml4gw_train import GWWhisperMultiDetector, GWWhisperModel  # adjust if needed

# -----------------------------------------------------------------------------
# 2. Load model for inference 
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

arch = GWWhisperMultiDetector(
    n_detectors=2,
    whisper_model_name="openai/whisper-tiny",
    kernel_length=1.0,
    sample_rate=1024,
    q_range=[4,128],
    spectrogram_shape=[128,128],
    output_dim=2,
).to(device)


model = GWWhisperModel(
        architecture=arch,
        ifos=["H1", "L1"],
        kernel_length=1.0,
        sample_rate=1024,
        learning_rate=1e-3,
        batch_size=128,
        max_epochs=1,
        checkpoint_dir="checkpoints_gw_whisper",
        log_dir="logs_gw_whisper",
        device=device,
    ).to(device)

ckpt = torch.load("checkpoints/Detection/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -----------------------------------------------------------------------------
# 3. Load test data (simplified version of your HDF5 loader)
# -----------------------------------------------------------------------------
def load_hdf5_pair(path, ifos=("H1","L1"), scale=22.116413):
    with h5py.File(path, "r") as f:
        inj, noi = [], []
        for ifo in ifos:
            inj.append(f["injection_samples"][f"{ifo.lower()}_strain"][()] / scale)
            noi.append(f["noise_samples"][f"{ifo.lower()}_strain"][()] / scale)
        return np.stack(inj, axis=1), np.stack(noi, axis=1)


X1_inj, X1_noi = load_hdf5_pair('/home/chattec/samplegen/output/GW231123_injection_test_random_glitches.hdf')
X2_inj, X2_noi = load_hdf5_pair('/home/chattec/LIGO/Wav2Vec2/Waveform_reconstruction/Eccentricity_data/O3b_test_EccentricTD_convergence_mass-30_SNR-12.hdf')
X3_inj, X3_noi = load_hdf5_pair('/data/p_dsi/ligo/IMBH_data/IMBH_val_GW231123_injection_test.hdf')

# Combine (add others as needed)
X_inj = np.concatenate([X1_inj, X2_inj[:,:,::2], X3_inj], axis=0)
X_noi = np.concatenate([X1_noi, X2_noi[:,:,::2], X3_noi], axis=0)

# Make labels
y_inj = np.tile([1,0], (len(X_inj), 1))
y_noi = np.tile([0,1], (len(X_noi), 1))

# Full dataset
X = torch.from_numpy(np.concatenate([X_inj, X_noi])).float()
y = torch.from_numpy(np.concatenate([y_inj, y_noi])).float()

# -----------------------------------------------------------------------------
# 4. Run inference
# -----------------------------------------------------------------------------
dl = DataLoader(TensorDataset(X,y), batch_size=128, shuffle=False)

y_true_list, y_score_list, usr_scores_list = [], [], []
with torch.no_grad():
    for xb, yb in dl:
        xb = xb.to(device)
        p = model(xb).float().clamp(1e-6, 1-1e-6)   # [B,2], probs after Sigmoid
        z = special.logit(p)                        # logits (inverse-sigmoid)
        s_usr = (z[:, 0] - z[:, 1]).cpu().numpy()   # USR = logit0 - logit1
        usr_scores_list.extend(s_usr)
        y_true_list.extend(yb[:, 0].cpu().numpy())
        y_score_list.extend(p[:,0].cpu().numpy())

y_true = np.array(y_true_list)
y_score = np.array(y_score_list)
scores_usr = np.array(usr_scores_list)

# -----------------------------------------------------------------------------
# 5. Compute metrics & plots
# -----------------------------------------------------------------------------
auroc = roc_auc_score(y_true, y_score)

fpr, tpr, usr_thresholds = roc_curve(y_true, scores_usr)

precisions, recalls, pr_thresholds = precision_recall_curve(y_true, scores_usr)

auprc = average_precision_score(y_true, scores_usr)

print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")


import matplotlib.pyplot as plt

# Create single figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# --- ROC curve (left y-axis) ---
color1 = 'tab:blue'
ax1.plot(fpr, tpr, color=color1, lw=2, label=f"AUROC = {auroc:.3f}")
ax1.set_xlabel("False Positive Rate / Recall",)
ax1.set_ylabel("True Positive Rate (ROC)", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, ls='--', alpha=0.5)

# --- PR curve (right y-axis) ---
ax2 = ax1.twinx()   # second y-axis sharing same x
color2 = 'tab:orange'
ax2.plot(recalls, precisions, color=color2, lw=2, label=f"AUPRC = {auprc:.3f}")
ax2.set_ylabel("Precision (PR)", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# --- Legends ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

#plt.title("ROC and Precision - Recall Curves (Same Axis)")
plt.tight_layout()
plt.savefig("ROC_PR_combined.png", dpi=300, bbox_inches='tight')