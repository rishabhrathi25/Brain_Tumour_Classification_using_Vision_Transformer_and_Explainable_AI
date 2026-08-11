# ================================================================
# COMPLETE ABLATION STUDY — SELF-CONTAINED CELL
# ================================================================

import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# ── 0. Constants (edit these 4 lines to match your paper values) ──
PROPOSED_ACC = 98.51   # np.mean(kfold_results) from your K-fold cell
PROPOSED_STD = 0.42    # np.std(kfold_results)
PROPOSED_SEN = 98.0    # from your sensitivity calculation
PROPOSED_SPE = 99.0    # from your specificity calculation

# ── 1. Setup ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

TRAIN_DIR  = "/kaggle/input/datasets/ogprakhar/brain-tumor-detection-2-0/Training"
TEST_DIR   = "/kaggle/input/datasets/ogprakhar/brain-tumor-detection-2-0/Testing"
NUM_CLASSES = 4
ABL_EPOCHS  = 10   # fair — matches your EPOCHS_KFOLD

# ── 2. Transform factory ──────────────────────────────────────────
def make_tf(use_norm=True, use_aug=True):
    ops = [transforms.Resize((224, 224))]
    if use_aug:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops.append(transforms.ToTensor())
    if use_norm:
        ops.append(transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]))
    return transforms.Compose(ops)

def build_ds(tf):
    tr = datasets.ImageFolder(TRAIN_DIR, transform=tf)
    te = datasets.ImageFolder(TEST_DIR,  transform=tf)
    return ConcatDataset([tr, te])

# ── 3. Train + evaluate one fold ─────────────────────────────────
def run_fold(ds, train_idx, val_idx):
    tl = DataLoader(Subset(ds, train_idx), batch_size=64,
                    shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(Subset(ds, val_idx),   batch_size=64,
                    shuffle=False, num_workers=2, pin_memory=True)

    m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    m = m.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ABL_EPOCHS)

    for ep in range(ABL_EPOCHS):
        m.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward()
            opt.step()
        sch.step()

    m.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            preds_all  += m(x).argmax(1).cpu().tolist()
            labels_all += y.cpu().tolist()

    del m; torch.cuda.empty_cache()
    acc = 100 * sum(p == l for p, l in zip(preds_all, labels_all)) / len(labels_all)
    return acc, labels_all, preds_all

# ── 4. Sen / Spe from CM ─────────────────────────────────────────
def sen_spe(labels, preds):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    s_list, p_list = [], []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        s_list.append(tp / (tp + fn + 1e-9))
        p_list.append(tn / (tn + fp + 1e-9))
    return np.mean(s_list) * 100, np.mean(p_list) * 100

# ── 5. Run one ablation config ───────────────────────────────────
def run_config(label, use_norm, use_aug, cross_val=True):
    print(f"\n  Running: {label}")
    ds = build_ds(make_tf(use_norm, use_aug))

    if cross_val:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs, lab_all, pred_all = [], [], []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(ds)))):
            acc, labs, preds = run_fold(ds, tr_idx, va_idx)
            accs.append(acc); lab_all += labs; pred_all += preds
            print(f"    Fold {fold+1}/5  acc={acc:.2f}%")
        mean_a, std_a = np.mean(accs), np.std(accs)
        sen, spe = sen_spe(lab_all, pred_all)
        print(f"  => {mean_a:.2f}% ± {std_a:.2f}  Sen={sen:.1f}% Spe={spe:.1f}%")
        return dict(label=label, acc=mean_a, std=std_a,
                    sen=sen, spe=spe, cv=True)
    else:
        n   = len(ds)
        idx = np.random.RandomState(42).permutation(n)
        cut = int(0.7 * n)
        acc, labs, preds = run_fold(ds, idx[:cut], idx[cut:])
        sen, spe = sen_spe(labs, preds)
        print(f"  => {acc:.2f}% (single split)  Sen={sen:.1f}% Spe={spe:.1f}%")
        return dict(label=label, acc=acc, std=None,
                    sen=sen, spe=spe, cv=False)

# ── 6. Run all configurations ─────────────────────────────────────
print("=" * 60)
print("ABLATION STUDY — starting runs")
print("=" * 60)

# Row 1: Proposed — no retraining, use your hard-coded values
proposed = dict(label="Proposed (all components)",
                acc=PROPOSED_ACC, std=PROPOSED_STD,
                sen=PROPOSED_SEN, spe=PROPOSED_SPE, cv=True)
print(f"\n  Proposed (all components): {PROPOSED_ACC:.2f}% ± {PROPOSED_STD:.2f}  "
      f"[from existing K-fold run — no retraining]")

r_no_aug  = run_config("(-) Data Augmentation",        use_norm=True,  use_aug=False, cross_val=True)
r_no_norm = run_config("(-) Preprocessing (norm)",     use_norm=False, use_aug=True,  cross_val=True)
r_no_both = run_config("(-) Preprocessing & Aug",      use_norm=False, use_aug=False, cross_val=True)
r_no_cv   = run_config("(-) 5-Fold CV (single split)", use_norm=True,  use_aug=True,  cross_val=False)

# Row 6: XAI is post-hoc — same accuracy as proposed
r_no_xai  = dict(label="(-) Explainability (XAI)",
                 acc=PROPOSED_ACC, std=PROPOSED_STD,
                 sen=PROPOSED_SEN, spe=PROPOSED_SPE, cv=True)
print(f"\n  (-) Explainability (XAI): {PROPOSED_ACC:.2f}% ± {PROPOSED_STD:.2f}  "
      f"[post-hoc, identical to Proposed]")

results = [proposed, r_no_aug, r_no_norm, r_no_both, r_no_cv, r_no_xai]

# ── 7. Print summary table ────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Configuration':<40} {'Acc (%)':>12}  {'Sen':>6}  {'Spe':>6}")
print("=" * 75)
for r in results:
    if r['std'] is None:
        acc_str = f"{r['acc']:.2f}†"
    else:
        acc_str = f"{r['acc']:.2f} ± {r['std']:.2f}"
    print(f"{r['label']:<40} {acc_str:>12}  {r['sen']:>5.1f}%  {r['spe']:>5.1f}%")
print("=" * 75)
print("† Single 70/30 split, no cross-fold variance")

# ── 8. Bar chart visualization ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

labels   = [r['label'] for r in results]
accs     = [r['acc']   for r in results]
errs     = [r['std'] if r['std'] is not None else 0 for r in results]
colors   = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results))]

bars = ax.bar(range(len(results)), accs, yerr=errs, capsize=5,
              color=colors, edgecolor='black', linewidth=0.8,
              error_kw=dict(elinewidth=1.5, ecolor='black'))

# Value labels on bars
for bar, acc, err, r in zip(bars, accs, errs, results):
    tag = "†" if r['std'] is None else ""
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (err if err else 0) + 0.3,
            f"{acc:.2f}%{tag}", ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')

ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, rotation=18, ha='right', fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Ablation Study — ViT-B/16 Brain Tumor Classification",
             fontsize=13, fontweight='bold')
ax.set_ylim(min(accs) - 5, 102)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

green_patch = mpatches.Patch(color='#2ecc71', label='Proposed (full pipeline)')
blue_patch  = mpatches.Patch(color='#3498db', label='Ablated variant')
ax.legend(handles=[green_patch, blue_patch], fontsize=9)

plt.tight_layout()
plt.savefig("/kaggle/working/ablation_bar_chart.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: ablation_bar_chart.png")

# ── 9. LaTeX table ───────────────────────────────────────────────
def to_latex(results):
    CHECK = r"\checkmark"
    DASH  = "--"
    rows  = []
    for r in results:
        n   = r['label']
        pre = DASH  if ("(-) Pre" in n) else CHECK
        aug = DASH  if ("(-) Aug" in n or "& Aug" in n) else CHECK
        cv  = DASH  if "5-Fold"  in n else CHECK
        xai = DASH  if "XAI"    in n else CHECK

        if "Proposed" in n:
            acc_str = r"\textbf{" + f"{r['acc']:.2f}" + r" $\pm$ " + f"{r['std']:.2f}" + "}"
        elif r['std'] is None:
            acc_str = f"{r['acc']:.2f}" + r"$^{\dagger}$"
        else:
            acc_str = f"{r['acc']:.2f}" + r" $\pm$ " + f"{r['std']:.2f}"

        rows.append(
            f"  {n:<42s} & {pre} & {aug} & {cv} & {xai} "
            f"& {acc_str} & {r['sen']:.1f} / {r['spe']:.1f} \\\\"
        )

    return (
        "\\begin{table}[!t]\n"
        "\\caption{Ablation Study of the Proposed ViT Pipeline Components}\n"
        "\\label{tab:ablation}\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.25}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\hline\n"
        "\\textbf{Configuration} & \\textbf{Pre.} & \\textbf{Aug.} & "
        "\\textbf{CV} & \\textbf{XAI} & \\textbf{Acc (\\%)} & \\textbf{Sen/Spe} \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\multicolumn{7}{l}{\\scriptsize $^{\\dagger}$Single 70/30 split; "
        "no cross-fold variance.}\\\\\n"
        "\\multicolumn{7}{l}{\\scriptsize XAI is post-hoc; "
        "removing it does not alter Acc/Sen/Spe.}\\\\\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

latex = to_latex(results)
print("\n========== LaTeX Table ==========")
print(latex)

with open("/kaggle/working/ablation_table.tex", "w") as f:
    f.write(latex)
print("\nSaved: /kaggle/working/ablation_table.tex")
