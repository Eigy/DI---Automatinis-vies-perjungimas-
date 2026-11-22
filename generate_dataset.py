"""Sintetinių naktinio kelio kadrų generatorius.

Sukuriama 1600 spalvotų (RGB) 64x64 pikselių kadrų, subalansuotų tarp dviejų klasių:
    0 - Tolimosios šviesos (kelias laisvas, priešais automobilio nėra)
    1 - Artimosios šviesos (priešais atvažiuoja automobilis)

Kadrai imituoja priekinės kameros vaizdą naktį: tamsus asfaltas, kelio linija, gatvės
žibintai, atšvaitai, lietus, rūkas, šlapio asfalto atspindžiai, priešais atvažiuojančio
automobilio žibintai, kartais vienas žibintas (motociklas / perdegusi lemputė) ir
priekyje važiuojančio automobilio raudoni galiniai žibintai.

Paleidimas:
    python code/generate_dataset.py            # sugeneruoja data/dataset.npz ir data/preview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

IMG = 64
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.npz"
CLASS_NAMES = {0: "Tolimosios šviesos (kelias laisvas)", 1: "Artimosios šviesos (priekyje automobilis)"}

_YY, _XX = np.mgrid[0:IMG, 0:IMG].astype(np.float32)


def _blob(canvas, x, y, sx, sy, color, intensity):
    """Prideda ryškų šviesos taškelį su švytėjimu (dviejų Gauso funkcijų suma)."""
    color = np.asarray(color, np.float32)
    for scale, weight in ((2.3, 0.35), (1.0, 1.0)):
        g = np.exp(-(((_XX - x) ** 2) / (2 * (sx * scale) ** 2) + ((_YY - y) ** 2) / (2 * (sy * scale) ** 2)))
        canvas += g[..., None] * (color * intensity * weight)[None, None, :]


def _random_street_lamp_color(rng):
    return [(1.0, 0.75, 0.40), (1.0, 0.92, 0.80), (0.85, 0.90, 1.0)][int(rng.integers(0, 3))]


def make_frame(label: int, rng: np.random.Generator) -> np.ndarray:
    """Sugeneruoja vieną 64x64x3 uint8 (RGB) kadrą nurodytai klasei."""
    import cv2

    img = np.zeros((IMG, IMG, 3), np.float32)
    horizon = int(rng.integers(24, 31))
    cx = IMG / 2 + float(rng.uniform(-6, 6))

    # Dangus ir asfaltas
    img[:horizon] += rng.uniform(0.010, 0.040) * np.array([0.8, 0.9, 1.2], np.float32)
    rows = np.arange(IMG, dtype=np.float32)[:, None]
    t = np.clip((rows - horizon) / (IMG - horizon), 0, 1)
    cone = np.exp(-((_XX[0][None, :] - cx) ** 2) / (2 * 22.0 ** 2))  # savo šviesų ruožas
    road = rng.uniform(0.025, 0.055) * (0.6 + 0.8 * t) * (0.7 + 0.5 * cone)
    img[horizon:] += road[horizon:, :, None] * np.array([1.0, 1.0, 1.05], np.float32)

    # Kelio vidurio linija (brūkšninė)
    layer = np.zeros_like(img)
    xb = cx + float(rng.uniform(-4, 4))
    gray = float(rng.uniform(0.18, 0.30))
    for t0, t1 in ((0.15, 0.30), (0.42, 0.62), (0.72, 1.0)):
        p0 = (cx + (xb - cx) * t0, horizon + (IMG - horizon) * t0)
        p1 = (cx + (xb - cx) * t1, horizon + (IMG - horizon) * t1)
        cv2.line(layer, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), (gray,) * 3, 1)
    img += layer

    # Atšvaitai kelio šonuose
    for _ in range(int(rng.integers(0, 5))):
        tt = float(rng.uniform(0.1, 0.9))
        side = 1 if rng.random() < 0.5 else -1
        x = cx + side * (2 + 30 * tt)
        y = horizon + (IMG - horizon) * tt
        _blob(img, x, y, 0.6, 0.6, [(1.0, 0.55, 0.2), (1.0, 1.0, 0.9)][int(rng.integers(0, 2))], rng.uniform(0.4, 0.7))

    wet = rng.random() < 0.30
    rain = rng.random() < 0.25
    fog = rng.random() < 0.20

    # Gatvės žibintai viršutinėje dalyje (+ atspindžiai ant šlapio asfalto)
    n_lamps = int(rng.integers(0, 6)) if label == 0 else int(rng.integers(0, 4))
    for _ in range(n_lamps):
        x = float(rng.uniform(2, IMG - 2))
        y = float(rng.uniform(2, max(3, horizon - 3)))
        col = _random_street_lamp_color(rng)
        s = float(rng.uniform(0.9, 1.8))
        _blob(img, x, y, s, s, col, rng.uniform(0.6, 1.0))
        if wet:
            _blob(img, x + rng.normal(0, 1.5), horizon + rng.uniform(4, 24), 0.8, rng.uniform(3, 5), col, rng.uniform(0.25, 0.45))

    # Sunkūs neigiami pavyzdžiai: dvi šalia viena kitos esančios atspindžių dėmės (klasė 0)
    if label == 0 and (wet or rng.random() < 0.15) and rng.random() < 0.5:
        gap = float(rng.uniform(6, 14))
        xm = float(rng.uniform(gap / 2 + 4, IMG - gap / 2 - 4))
        ym = horizon + float(rng.uniform(5, 22))
        for dx in (-gap / 2, gap / 2):
            _blob(img, xm + dx, ym, 0.9, rng.uniform(1.5, 3.0), (1.0, 0.97, 0.9), rng.uniform(0.30, 0.55))

    if label == 1:
        # Priešais atvažiuojantis automobilis: du ryškūs balti žibintai (arba vienas)
        dist = float(rng.uniform(0, 1))  # 0 - arti, 1 - toli
        near = 1 - dist
        y = horizon + 2 + near * float(rng.uniform(8, 16))
        spacing = 4 + near * 16 * float(rng.uniform(0.8, 1.1))
        sigma = 0.9 + near * 1.6
        x0 = cx - float(rng.uniform(0, 12))
        x0 = float(np.clip(x0, spacing / 2 + 2, IMG - spacing / 2 - 2))
        col = [(1.0, 0.97, 0.90), (0.90, 0.95, 1.0)][int(rng.integers(0, 2))]
        inten = float(rng.uniform(0.8, 1.0))
        if rng.random() < 0.08:  # motociklas arba perdegusi lemputė
            _blob(img, x0, y, sigma, sigma, col, inten)
        else:
            for dx in (-spacing / 2, spacing / 2):
                _blob(img, x0 + dx, y, sigma, sigma, col, inten)
        # Priekyje ta pačia kryptimi važiuojančio automobilio raudoni galiniai žibintai
        if rng.random() < 0.35:
            x1 = cx + float(rng.uniform(-3, 8))
            y1 = horizon + float(rng.uniform(2, 12))
            sp = float(rng.uniform(3, 9))
            for dx in (-sp / 2, sp / 2):
                _blob(img, x1 + dx, y1, rng.uniform(0.9, 1.4), rng.uniform(0.9, 1.4), (1.0, 0.10, 0.08), rng.uniform(0.5, 0.9))

    # Lietus
    if rain:
        layer = np.zeros_like(img)
        for _ in range(int(rng.integers(30, 80))):
            x = int(rng.integers(0, IMG)); y = int(rng.integers(0, IMG - 4))
            cv2.line(layer, (x, y), (x, y + int(rng.integers(2, 5))), (float(rng.uniform(0.08, 0.2)),) * 3, 1)
        img += layer

    # Rūkas
    if fog:
        a = float(rng.uniform(0.15, 0.40))
        img = img * (1 - a) + a * float(rng.uniform(0.12, 0.22))
        img = cv2.GaussianBlur(img, (0, 0), 1.2)

    img += rng.normal(0, rng.uniform(0.008, 0.025), img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def generate(n_per_class: int = 800, seed: int = SEED):
    """Sugeneruoja subalansuotą rinkinį ir stratifikuotą 70/15/15 padalijimą."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = np.zeros((2 * n_per_class, IMG, IMG, 3), np.uint8)
    y = np.zeros(2 * n_per_class, np.int64)
    for i in range(2 * n_per_class):
        label = i % 2
        X[i] = make_frame(label, rng)
        y[i] = label
    idx = np.arange(len(y))
    train_idx, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return X, y, train_idx, val_idx, test_idx


def save(path: Path = DATA_PATH, n_per_class: int = 800, seed: int = SEED):
    X, y, tr, va, te = generate(n_per_class, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, train_idx=tr, val_idx=va, test_idx=te)
    return X, y, tr, va, te


def load(path: Path = DATA_PATH, n_per_class: int = 800, seed: int = SEED, regenerate: bool = False):
    """Įkelia rinkinį; jei failo nėra (arba prašoma) - sugeneruoja iš naujo."""
    if regenerate or not Path(path).exists():
        return save(Path(path), n_per_class, seed)
    d = np.load(path)
    return d["X"], d["y"], d["train_idx"], d["val_idx"], d["test_idx"]


def save_preview(X, y, path: Path):
    """Išsaugo kadrų pavyzdžių lentelę (viršuje klasė 0, apačioje klasė 1)."""
    import cv2

    rows = []
    for cls in (0, 1):
        imgs = [cv2.resize(X[np.where(y == cls)[0][k]], (128, 128), interpolation=cv2.INTER_NEAREST) for k in range(6)]
        rows.append(np.hstack(imgs))
    grid = np.vstack(rows)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    a = ap.parse_args()
    X, y, tr, va, te = save(a.out, a.n_per_class, a.seed)
    save_preview(X, y, a.out.parent / "preview.png")
    print(f"Sugeneruota {len(y)} kadrų: klasė 0 = {(y == 0).sum()}, klasė 1 = {(y == 1).sum()}")
    print(f"Padalijimas: mokymas {len(tr)}, patvirtinimas {len(va)}, testavimas {len(te)}")
    print(f"Išsaugota: {a.out}")
