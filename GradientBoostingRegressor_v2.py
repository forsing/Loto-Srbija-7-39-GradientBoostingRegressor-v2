#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GradientBoostingRegressor

# ===== v2 nadogradnja =====
#   • multi-label umesto pozicionog regresora: 39 GBR-ova (jedan po broju 1..39)
#   • features iz prošlosti: lag(5), rolling freq (20/50/100), gap, statistike prošlog kola
#   • vremenski split (bez shuffle), poslednje 100 izvlačenja za back-test
#   • predikcija iz POSLEDNJEG reda CSV-a (a ne nasumičnog test-reda)
#   • top-7 jedinstvenih, sortirano, 1..39 + validatori
#   • back-test: hits/7, hit%, AUC, LRAP + slučajan baseline
#   • snimanje u TXT, vreme start/stop/elapsed
#   • SHOW_PLOTS=False — sns.countplot je opcioni (default isključen)

SHOW_PLOTS = False  # postavi True ako želiš 7 sns.countplot prozora pre treninga
"""



import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONHASHSEED"] = "39"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)


# =========================
# Konfiguracija
# =========================
CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/KvantniRegresor/GradientBoostingRegressor_v2_predikcija.txt")
N_MIN, N_MAX = 1, 39
K = 7
LAG = 5
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
SHOW_PLOTS = False  # postavi True ako želiš 7 sns.countplot prozora pre treninga

T0 = time.time()
print()
print("START", datetime.today())
print()
"""
START 2026-05-24 19:09:12.867992
"""


# 1. Učitaj loto podatke
df = pd.read_csv(CSV_PATH)


# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7].astype(int)


# Kreiranje ulaznih (X) i izlaznih (y) podataka  (zadržano radi kompatibilnosti)
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# Train/test split  (zadržano radi kompatibilnosti, ne koristi se za v2 trening)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)


# sns.distplot(df.sum(axis=1), fit=stats.gamma)


if SHOW_PLOTS:
    mpl.rc("figure", figsize=(12, 12))

    sns.countplot(x="Num1", data=df)
    plt.show()
    sns.countplot(x="Num2", data=df)
    plt.show()
    sns.countplot(x="Num3", data=df)
    plt.show()
    sns.countplot(x="Num4", data=df)
    plt.show()
    sns.countplot(x="Num5", data=df)
    plt.show()
    sns.countplot(x="Num6", data=df)
    plt.show()
    sns.countplot(x="Num7", data=df)
    plt.show()


###################################


# =========================
# v2: multi-label feature engineering + GBR po broju 1..39
# =========================
draws = np.sort(df.values, axis=1)
N = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")


def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def build_features(draws_arr: np.ndarray, y_multi: np.ndarray) -> np.ndarray:
    n, _ = draws_arr.shape
    lag_blocks = []
    for lag in range(1, LAG + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[lag:] = draws_arr[:-lag]
        lag_blocks.append(shifted)
    lag_block = np.concatenate(lag_blocks, axis=1).astype(float)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        rolling_blocks.append(rolled / float(w))
    rolling_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stats_block = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, rolling_block, gap, stats_block], axis=1)


def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d: np.ndarray, y_true: np.ndarray) -> float:
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
X_full = build_features(draws, Y_full)
START = max(LAG, max(WINDOWS))

X_all = X_full[START:].astype(float)
Y_all = Y_full[START:].astype(float)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train_v2, Y_train_v2 = X_all[:n_train], Y_all[:n_train]
X_back_v2, Y_back_v2 = X_all[n_train:], Y_all[n_train:]
X_next_raw = X_full[-1:].astype(float)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_v2)
X_back_scaled = scaler.transform(X_back_v2)
X_next_scaled = scaler.transform(X_next_raw)


# 39 GBR-ova kroz MultiOutputRegressor
gbr_base = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    random_state=SEED,
)
mor = MultiOutputRegressor(gbr_base, n_jobs=1)


print()
print("Treniranje GradientBoostingRegressor multi-label (39 izlaza) ...")
mor.fit(X_train_scaled, Y_train_v2)
print("✅ GBR_v2 treniran.")
print()
"""
Treniranje GradientBoostingRegressor multi-label (39 izlaza) ...
✅ GBR_v2 treniran.
"""



# Back-test + predikcija sledećeg kola
scores_back = mor.predict(X_back_scaled)
scores_next = mor.predict(X_next_scaled)[0]
predicted_numbers_v2 = topk_from_scores(scores_next)

assert len(set(predicted_numbers_v2.tolist())) == K
assert predicted_numbers_v2.min() >= N_MIN and predicted_numbers_v2.max() <= N_MAX
assert list(predicted_numbers_v2) == sorted(predicted_numbers_v2.tolist())

h = avg_hits(scores_back, Y_back_v2)
a = safe_auc(Y_back_v2, scores_back)
l = safe_lrap(Y_back_v2, scores_back)

print()
print(f"Prediction of GBR_v2 (top-7): {predicted_numbers_v2.tolist()}  ({describe(predicted_numbers_v2)})")
print()
"""
Prediction of GBR_v2 (top-7): [8, 18, 23, 27, 28, 29, 30]  (suma=163, neparnih=3/7, niskih(<=19)=2/7, raspon=22)
"""


# =========================
# Stari pozicioni tok (zadržan kao referenca)
# =========================
X = np.array([df["Num1"], df["Num2"], df["Num3"], df["Num4"], df["Num5"], df["Num6"], df["Num7"]] )
bindex = 0
final = []

print()
for ball in [ df["Num1"], df["Num2"], df["Num3"], df["Num4"], df["Num5"], df["Num6"], df["Num7"] ]:
    Y = np.array(ball.values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X.transpose(), Y, test_size=0.8, random_state=SEED)
    reg = GradientBoostingRegressor(random_state=SEED)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)               
    # accuracy = accuracy_score(y_test, y_pred)  
    final.append(y_pred[bindex])
    if len(final)!=len(set(final)):
        Y = np.array(ball.values.tolist())
        X_train, X_test, y_train, y_test = train_test_split(X.transpose(), Y, test_size=0.9, random_state=SEED)
        reg = GradientBoostingRegressor(random_state=SEED)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)               
        # accuracy = accuracy_score(y_test, y_pred)  
        final.append(y_pred[bindex])        
        
    print(f"Prediction of Ball {bindex + 1} is [{y_pred[bindex]}] ")
    bindex = bindex + 1  

    
print()


"""
Prediction of Ball 1 is [3.0000538661179] 
Prediction of Ball 2 is [9.000023235489401] 
Prediction of Ball 3 is [4.001086158537905] 
Prediction of Ball 4 is [25.999830653097156] 
Prediction of Ball 5 is [29.99987162756386] 
Prediction of Ball 6 is [33.999901963504676] 
Prediction of Ball 7 is [35.99998040171298] 
"""


print()
final.sort()
print("Predicted Numbers:", np.round(final).astype(int).tolist())
print()
S = sum(final)
print(f"Sum of numbers: {S}")
print(f"Sum is good!") if S >= 120 and S <= 190 else print(f"Sum of prediction is out of ideal range. Re-run prediction.")
print()

"""
Predicted Numbers: [3, 4, 9, 26, 30, 34, 36]

Sum of numbers: 142.00074790602386
Sum is good!
"""


# =========================
# Back-test rezime + snimanje
# =========================
print()
print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<8} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'GBR_v2':<8} {h:>8.3f} {100*h/K:>6.1f}% {a:>7.3f} {l:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()
"""
Back-test (poslednjih 100 izvlačenja):
model      hits/7    hit%     AUC    LRAP
GBR_v2      1.140   16.3%   0.497   0.238
(slučajan baseline ≈ 1.256 hits/7)
"""



with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N}) ---\n")
    f.write(f"GBR_v2 (top-7)                    -> {predicted_numbers_v2.tolist()}  ({describe(predicted_numbers_v2)})\n")
    f.write(f"GBR_pozicioni (stari tok, sortiran)-> {np.round(final).astype(int).tolist()}\n")
    f.write(f"back-test GBR_v2: hits/7={h:.3f}, baseline={7*7/39:.3f}, AUC={a:.3f}, LRAP={l:.3f}\n")
print()
print(f"Snimljeno u: {OUT_TXT}")
print()
"""
Snimljeno u: /GradientBoostingRegressor_v2_predikcija.txt
"""

elapsed = time.time() - T0
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()
"""
STOP 2026-05-24 19:18:05.449784
Ukupno vreme: 0:08:52  (532.6 s)
"""





"""
START 2026-05-24 19:09:12.867992

Treniranje GradientBoostingRegressor multi-label (39 izlaza) ...
✅ GBR_v2 treniran.


Prediction of GBR_v2 (top-7): [8, 18, 23, 27, 28, 29, 30]  (suma=163, neparnih=3/7, niskih(<=19)=2/7, raspon=22)


Prediction of Ball 1 is [3.0000538661179] 
Prediction of Ball 2 is [9.000023235489401] 
Prediction of Ball 3 is [4.001086158537905] 
Prediction of Ball 4 is [25.999830653097156] 
Prediction of Ball 5 is [29.99987162756386] 
Prediction of Ball 6 is [33.999901963504676] 
Prediction of Ball 7 is [35.99998040171298] 


Predicted Numbers: [3, 4, 9, 26, 30, 34, 36]

Sum of numbers: 142.00074790602386
Sum is good!


Back-test (poslednjih 100 izvlačenja):
model      hits/7    hit%     AUC    LRAP
GBR_v2      1.140   16.3%   0.497   0.238
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /Users/4c/Desktop/GHQ/KvantniRegresor/GradientBoostingRegressor_v2_predikcija.txt

STOP 2026-05-24 19:18:05.449784
Ukupno vreme: 0:08:52  (532.6 s)
"""
