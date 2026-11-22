"""Bazinių modelių ir CompactHeadlightCNN mokymas, testavimas ir greičio matavimas.

Lyginami trys sprendimai ta pačia 240 kadrų testine imtimi:
    1. Bandymas 1 (atmestas): viso kadro vidutinio ryškumo slenkstis
    2. Bazinis: 8 OpenCV/NumPy ROI požymiai + StandardScaler + LogisticRegression
    3. Patobulintas: lengvas konvoliucinis tinklas CompactHeadlightCNN (PyTorch)

Paleidimas:
    python code/train_and_eval.py                 # visa grandinė
    python code/train_and_eval.py --skip-cnn      # be PyTorch (tik 1 ir 2 modeliai)

Rezultatai: outputs/results.json, outputs/*.png, models/*
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import generate_dataset as gd  # noqa: E402

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # leidžia paleisti --skip-cnn be PyTorch
    torch = None

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
SEED = 42

# Hiperparametrai (žr. ataskaitos 2 lentelę)
EPOCHS, BATCH, LR, WEIGHT_DECAY, STEP, GAMMA, DROPOUT = 22, 32, 0.0012, 1e-4, 8, 0.5, 0.25
ROI_ROWS = slice(18, 56)   # kelio sritis (domėjimosi sritis)
SKY_ROWS = slice(0, 18)    # dangus


# ----------------------------------------------------------------------------------------
# Bandymas 1 ir bazinis modelis
# ----------------------------------------------------------------------------------------
def luminance(X):
    """RGB uint8 (N,H,W,3) -> ryškumas (N,H,W) intervale [0,1]."""
    X = X.astype(np.float32) / 255.0
    return 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]


class BrightnessThreshold:
    """Paprasčiausia taisyklė: jei viso kadro vidutinis ryškumas didesnis už slenkstį -> artimosios."""

    def fit(self, X, y):
        m = luminance(X).mean(axis=(1, 2))
        cands = np.unique(m)
        accs = [((m > c).astype(int) == y).mean() for c in cands]
        self.thr = float(cands[int(np.argmax(accs))])
        return self

    def predict(self, X):
        return (luminance(X).mean(axis=(1, 2)) > self.thr).astype(int)


def roi_features(X):
    """8 rankiniu būdu sudaryti požymiai iš kelio srities (ROI) ir dangaus."""
    X = X.astype(np.float32) / 255.0
    lum = 0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]
    roi, sky = lum[:, ROI_ROWS, :].reshape(len(X), -1), lum[:, SKY_ROWS, :].reshape(len(X), -1)
    rgb = X[:, ROI_ROWS, :, :].reshape(len(X), -1, 3)
    white = ((rgb > 0.8).all(axis=2)).mean(axis=1)
    red = ((rgb[..., 0] > 0.5) & (rgb[..., 1] < 0.35) & (rgb[..., 2] < 0.35)).mean(axis=1)
    return np.stack([roi.mean(1), roi.max(1), np.percentile(roi, 95, axis=1), np.percentile(roi, 99, axis=1),
                     white, red, sky.mean(1), roi.std(1)], axis=1)


FEATURE_NAMES = ["ROI vidutinis ryškumas", "ROI maksimalus ryškumas", "ROI 95 procentilis", "ROI 99 procentilis",
                 "ryškių baltų pikselių dalis", "raudonų pikselių dalis", "dangaus ryškumas", "ROI ryškumo std"]


class Baseline:
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(roi_features(X))
        self.clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(self.scaler.transform(roi_features(X)), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(roi_features(X)))[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ----------------------------------------------------------------------------------------
# CompactHeadlightCNN
# ----------------------------------------------------------------------------------------
if torch is not None:

    class CompactHeadlightCNN(nn.Module):
        """3 blokai (Conv2d, BatchNorm2d, ReLU, MaxPool2d) -> AdaptiveAvgPool2d -> Dropout -> 2 pilnai sujungti sluoksniai.
        Iš viso 43 826 parametrai."""

        def __init__(self, dropout: float = DROPOUT):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))

            self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 48))
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(48 * 4 * 4, 32), nn.ReLU(inplace=True),
                                            nn.Linear(32, 2))

        def forward(self, x):
            return self.classifier(self.pool(self.features(x)))

    def to_tensor(X):
        """uint8 (N,H,W,3) -> float32 (N,3,H,W) intervale [0,1]."""
        return torch.from_numpy(X).float().div(255.0).permute(0, 3, 1, 2).contiguous()

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def cnn_proba(model, X, batch=256):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out.append(torch.softmax(model(to_tensor(X[i:i + batch])), dim=1)[:, 1].numpy())
        return np.concatenate(out)

    def train_cnn(Xtr, ytr, Xva, yva, epochs=EPOCHS):
        torch.manual_seed(SEED)
        model = CompactHeadlightCNN()
        print(f"CompactHeadlightCNN parametrų: {count_params(model):,}".replace(",", " "))
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=GAMMA)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(to_tensor(Xtr), torch.from_numpy(ytr)), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        xva, yva_t = to_tensor(Xva), torch.from_numpy(yva)
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_acc, best_state, best_epoch = -1.0, None, 0
        for ep in range(1, epochs + 1):
            model.train()
            tl, tc, n = 0.0, 0, 0
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tl += loss.item() * len(yb); tc += (logits.argmax(1) == yb).sum().item(); n += len(yb)
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = model(xva)
                vloss = loss_fn(vl, yva_t).item()
                vacc = (vl.argmax(1) == yva_t).float().mean().item()
            hist["train_loss"].append(tl / n); hist["train_acc"].append(tc / n)
            hist["val_loss"].append(vloss); hist["val_acc"].append(vacc)
            if vacc > best_acc:
                best_acc, best_state, best_epoch = vacc, copy.deepcopy(model.state_dict()), ep
            print(f"Epocha {ep:2d}/{epochs}: mokymo nuostolis {tl / n:.4f}, tikslumas {tc / n:.3f} | "
                  f"validavimo nuostolis {vloss:.4f}, tikslumas {vacc:.3f}")
        model.load_state_dict(best_state)  # galutiniam testavimui - geriausia pagal validavimo tikslumą epocha
        hist["best_epoch"] = best_epoch
        print(f"Geriausia epocha: {best_epoch} (validavimo tikslumas {best_acc:.3f})")
        return model, hist

    def load_cnn(path=MODELS / "cnn.pt"):
        model = CompactHeadlightCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def saliency(model, X):
        """Paprasčiausias XAI žemėlapis: įvesties gradiento modulis klasei 1 (artimosios šviesos)."""
        model.eval()
        x = to_tensor(X).requires_grad_(True)
        model(x)[:, 1].sum().backward()
        s = x.grad.abs().amax(dim=1).numpy()
        s = (s - s.min(axis=(1, 2), keepdims=True)) / (np.ptp(s, axis=(1, 2), keepdims=True) + 1e-9)
        return s


# ----------------------------------------------------------------------------------------
# Vertinimas
# ----------------------------------------------------------------------------------------
def evaluate(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, pred), prec=p, recall=r, f1=f, tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))


def ms_per_frame(fn, frame, runs=100, warmup=10):
    for _ in range(warmup):
        fn(frame)
    t = time.perf_counter()
    for _ in range(runs):
        fn(frame)
    return (time.perf_counter() - t) / runs * 1000


def fmt(v, pct=True):
    s = f"{v * 100:.1f} %" if pct else f"{v:.3f}"
    return s.replace(".", ",")


# ----------------------------------------------------------------------------------------
# Grafikai
# ----------------------------------------------------------------------------------------
def plot_training(hist, path):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, hist["train_loss"], label="Mokymas"); ax[0].plot(ep, hist["val_loss"], label="Validavimas")
    ax[0].set_xlabel("Epocha"); ax[0].set_ylabel("Nuostolis"); ax[0].set_title("a) Nuostolio funkcija"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["train_acc"], label="Mokymas"); ax[1].plot(ep, hist["val_acc"], label="Validavimas")
    ax[1].axhline(0.80, color="gray", ls="--", lw=1, label="80 % riba")
    ax[1].set_xlabel("Epocha"); ax[1].set_ylabel("Tikslumas"); ax[1].set_title("b) Tikslumas"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_confusions(results, path):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        r = results[n]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Tolimosios", "Artimosios"]); ax.set_yticks([0, 1], ["Tolimosios", "Artimosios"])
        ax.set_xlabel("Prognozė"); ax.set_ylabel("Tikroji klasė"); ax.set_title(n, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def pick_examples(y, prob):
    """Parenka po vieną TP, TN, FP, FN pavyzdį (didžiausio užtikrintumo)."""
    pred = (prob >= 0.5).astype(int)
    groups = {"TP": ((y == 1) & (pred == 1), +1), "TN": ((y == 0) & (pred == 0), -1),
              "FP": ((y == 0) & (pred == 1), +1), "FN": ((y == 1) & (pred == 0), -1)}
    sel = {}
    for key, (mask, sign) in groups.items():
        idx = np.where(mask)[0]
        if len(idx):
            sel[key] = int(idx[np.argmax(sign * prob[idx])])
    return sel


def plot_examples(X, y, prob, path, model=None):
    sel = pick_examples(y, prob)
    titles = {"TP": "SĖKMĖ (TP)", "TN": "SĖKMĖ (TN)", "FP": "KLAIDA (FP)", "FN": "KLAIDA (FN)"}
    colors = {"TP": "green", "TN": "green", "FP": "darkorange", "FN": "red"}
    keys = [k for k in ("TP", "TN", "FP", "FN") if k in sel]
    rows = 2 if model is not None else 1
    fig, axes = plt.subplots(rows, len(keys), figsize=(3 * len(keys), 3.2 * rows), squeeze=False)
    sal = saliency(model, X[[sel[k] for k in keys]]) if model is not None else None
    for c, k in enumerate(keys):
        i = sel[k]
        ax = axes[0, c]; ax.imshow(X[i]); ax.set_title(f"{titles[k]}, p={prob[i]:.2f}".replace(".", ","), color=colors[k], fontsize=10, fontweight="bold")
        for s in ax.spines.values():
            s.set_edgecolor(colors[k]); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        if model is not None:
            ax = axes[1, c]; ax.imshow(X[i]); ax.imshow(sal[c], cmap="jet", alpha=0.5)
            ax.set_title("Dėmesio žemėlapis", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-cnn", action="store_true", help="nemokyti CompactHeadlightCNN (nereikia PyTorch)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--regenerate", action="store_true", help="iš naujo sugeneruoti duomenų rinkinį")
    a = ap.parse_args()

    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    X, y, tr, va, te = gd.load(regenerate=a.regenerate)
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    print(f"Mokymas {len(tr)}, patvirtinimas {len(va)}, testavimas {len(te)} kadrų")
    frame = Xte[:1]
    results, probs = {}, {}

    thr = BrightnessThreshold().fit(Xtr, ytr)
    r = evaluate(yte, thr.predict(Xte)); r["ms"] = ms_per_frame(thr.predict, frame); r["threshold"] = thr.thr
    results["Bandymas 1: ryškumo slenkstis"] = r

    base = Baseline().fit(Xtr, ytr)
    r = evaluate(yte, base.predict(Xte)); r["ms"] = ms_per_frame(base.predict, frame)
    results["Bazinis: OpenCV ROI + LogisticRegression"] = r
    probs["base"] = base.predict_proba(Xte)
    import joblib
    joblib.dump(base, MODELS / "baseline.joblib")

    cnn, hist = None, None
    if not a.skip_cnn:
        if torch is None:
            raise SystemExit("PyTorch neįdiegtas. Įdiekite jį (pip install torch) arba paleiskite su --skip-cnn.")
        torch.set_num_threads(1)
        cnn, hist = train_cnn(Xtr, ytr, Xva, yva, a.epochs)
        prob = cnn_proba(cnn, Xte)
        r = evaluate(yte, (prob >= 0.5).astype(int))
        x1 = to_tensor(frame)

        def one(_):
            with torch.no_grad():
                cnn(x1)
        r["ms"] = ms_per_frame(one, None); r["params"] = count_params(cnn); r["best_epoch"] = hist["best_epoch"]
        results["Patobulintas: CompactHeadlightCNN"] = r
        torch.save(cnn.state_dict(), MODELS / "cnn.pt")
        json.dump(hist, open(OUT / "training_history.json", "w"), indent=1)
        plot_training(hist, OUT / "training_curves.png")
        plot_examples(Xte, yte, prob, OUT / "examples.png", model=cnn)

    json.dump(results, open(OUT / "results.json", "w"), indent=1, ensure_ascii=False)
    plot_confusions(results, OUT / "confusion_matrices.png")

    print("\n" + "=" * 100)
    print(f"{'Modelis':44s}{'Acc':>9s}{'Prec':>9s}{'Recall':>9s}{'F1':>8s}{'FN':>5s}{'FP':>5s}{'Laikas/kadrą':>14s}")
    for n, r in results.items():
        print(f"{n:44s}{fmt(r['acc']):>9s}{fmt(r['prec']):>9s}{fmt(r['recall']):>9s}{fmt(r['f1'], False):>8s}"
              f"{r['fn']:>5d}{r['fp']:>5d}{r['ms']:>11.2f} ms".replace(".", ","))
    print("=" * 100)
    print(f"Rezultatai išsaugoti: {OUT}")


if __name__ == "__main__":
    main()
