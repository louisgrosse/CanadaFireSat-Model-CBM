import numpy as np
import argparse
from pathlib import Path

def to_channel_last(probs):
    # Accept (N,H,W,C), (N,C,H,W), or (N_pixels,C)
    if probs.ndim == 2:
        # (N_pixels, C) already flattened
        return probs, None, None
    if probs.ndim != 4:
        raise ValueError(f"Unexpected probs.ndim={probs.ndim}. Expect 2 or 4.")
    N, *rest = probs.shape
    # If channel is 1st: (N,C,H,W)
    if probs.shape[1] in (1,2):
        probs = np.moveaxis(probs, 1, -1)  # -> (N,H,W,C)
    return probs, probs.shape[1], probs.shape[2]  # H,W if not flattened

def sigmoid(x):
    # numerically stable
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x, dtype=np.float64)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def maybe_to_probs(P):
    """Detect whether P are logits or probs; return probs in [0,1]."""
    pmin, pmax = float(P.min()), float(P.max())
    # Heuristic: if inside [0,1] and sum across classes ≈1 => probs
    if P.ndim == 2 and P.shape[1] == 2:
        rowsum = P.sum(axis=1)
        if 0 <= pmin and pmax <= 1.0 and np.allclose(rowsum.mean(), 1.0, atol=1e-3):
            return P
        # else likely logits -> softmax
        ex = np.exp(P - P.max(axis=1, keepdims=True))
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    elif P.ndim == 2 and P.shape[1] == 1:
        # 1-logit (sigmoid) case: if in [0,1] accept, else sigmoid
        if 0 <= pmin and pmax <= 1.0:
            return P
        return sigmoid(P)
    else:
        raise ValueError(f"Unexpected P shape {P.shape}")

def f1_from_counts(tp, fp, fn):
    denom = 2*tp + fp + fn
    return 0.0 if denom == 0 else (2*tp) / denom

def counts_from_preds(y_true, y_pred):
    tp = int(((y_true==1) & (y_pred==1)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    return tp, fp, fn, tn

def f1_at_threshold(scores, y_true, thr):
    y_pred = (scores >= thr).astype(np.uint8)
    tp, fp, fn, _ = counts_from_preds(y_true, y_pred)
    return f1_from_counts(tp, fp, fn)

def best_threshold(scores, y_true, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_at_threshold(scores, y_true, t) for t in grid]
    j = int(np.argmax(f1s))
    return float(grid[j]), float(f1s[j])

def flatten_apply_mask(y, HWC_or_flat, unk):
    """Return y_true (0/1), P (N_pixels,C)."""
    P = HWC_or_flat
    # Move to flattened if needed
    if P.ndim == 4:
        N, H, W, C = P.shape
        P = P.reshape(N*H*W, C)
        y = y.reshape(N*H*W)
        if unk is not None:
            unk = unk.reshape(N*H*W).astype(bool)
    elif P.ndim == 2:
        # P already flattened: y should be flattened too
        y = y.reshape(-1)
        if unk is not None:
            unk = unk.reshape(-1).astype(bool)
    else:
        raise ValueError(f"Unexpected probs ndim {P.ndim}")

    if unk is None:
        mask = np.ones_like(y, dtype=bool)
    else:
        mask = unk.astype(bool)
    return y[mask].astype(np.uint8), P[mask, :]

def main():

    # --- load your saved arrays ---
    probs = np.load("results/models/MS-CLIP_Fixed_Ontario_Alberta_Fast/metrics/val_temp_adapt_spa_224/val_probs.npy")      # shape: [N, H, W, C] or [N, C, H, W]
    target = np.load("results/models/MS-CLIP_Fixed_Ontario_Alberta_Fast/metrics/val_temp_adapt_spa_224/val_target.npy")

    # Ensure channel-last or flattened
    probs4, H, W = to_channel_last(probs)  # returns flattened=None if probs was 2D
    # Flatten + mask align
    y_true, P = flatten_apply_mask(target, probs4 if probs4 is not None else probs, None)

    # Convert to probabilities if logits
    P = maybe_to_probs(P)

    C = P.shape[1]
    print(f"[Info] Using {y_true.size} valid pixels; classes={C}")

    # Per-class supports (from labels)
    n_pos = int((y_true==1).sum())
    n_neg = int((y_true==0).sum())
    print(f"[Support] positives={n_pos}, negatives={n_neg}")

    if C == 2:
        # Try to detect which column is fire by scanning thresholds
        thr0, f1_0 = best_threshold(P[:,0], y_true)
        thr1, f1_1 = best_threshold(P[:,1], y_true)
        fire_col = 1 if f1_1 >= f1_0 else 0
        fire_scores = P[:, fire_col]

        # Argmax (equiv to 0.5 on proper softmax probs)
        y_pred_argmax = (np.argmax(P, axis=1) == fire_col).astype(np.uint8)
        tp, fp, fn, tn = counts_from_preds(y_true, y_pred_argmax)
        f1_argmax = f1_from_counts(tp, fp, fn)

        # 0.5 threshold on fire prob
        f1_thr05 = f1_at_threshold(fire_scores, y_true, 0.5)

        # Tuned threshold
        thr_star, f1_star = best_threshold(fire_scores, y_true)

        print(f"[Detect] fire_col={fire_col} (best_single_col_F1={max(f1_0,f1_1):.3f})")
        print(f"[Results] F1(argmax)   = {f1_argmax:.3f}")
        print(f"[Results] F1(p>=0.5)   = {f1_thr05:.3f}")
        print(f"[Results] F1(tuned)    = {f1_star:.3f} at thr={thr_star:.2f}")
        print(f"[Counts ] TP={tp} FP={fp} FN={fn} TN={tn}")

        # Helpful sanity: if argmax and 0.5 differ a lot, you probably passed logits, not probs.
        if abs(f1_argmax - f1_thr05) > 1e-3:
            print("[Warn ] Argmax vs 0.5 differ notably; ensure you applied softmax to logits before thresholding.")

    elif C == 1:
        # Sigmoid case
        scores = P[:,0]
        f1_thr05 = f1_at_threshold(scores, y_true, 0.5)
        thr_star, f1_star = best_threshold(scores, y_true)
        y_pred05 = (scores >= 0.5).astype(np.uint8)
        tp, fp, fn, tn = counts_from_preds(y_true, y_pred05)

        print(f"[Results] F1(p>=0.5)   = {f1_thr05:.3f}")
        print(f"[Results] F1(tuned)    = {f1_star:.3f} at thr={thr_star:.2f}")
        print(f"[Counts ] TP={tp} FP={fp} FN={fn} TN={tn}")
    else:
        raise ValueError(f"Unexpected number of classes C={C}")

if __name__ == "__main__":
    main()
