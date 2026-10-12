# BLOCK 20 (follow-up): post-processing analysis of the SAVED models - no retraining, no extra privacy cost
#   (a) Binary IDS view (attack vs normal): accuracy, precision, DR, FAR, F1 - shown NEXT TO the 10-class results
#   (b) Per-class ROC-AUC from the DDQN Q-values (ranking quality, independent of the argmax decision)
#   (c) Per-class decision calibration: offsets b added to the Q-values, learned on VALIDATION only
#       (max val macro-F1 with val FAR <= cap). The trained DDQN is unchanged; this is post-processing of a DP
#       model, so the (eps, delta) guarantee is unchanged.
# DISCLOSURE: test numbers produced here are a follow-up, second use of the test set after the main evaluation.
CALIB_TAG = "final"          # which saved runs to analyse (stage tag)
CALIB_EVAL_TEST = True       # False -> validation only
CALIB_GRID = np.linspace(-1.0, 1.0, 21)   # offsets, in units of the run's Q-value std
CALIB_PASSES = 3

from sklearn.metrics import roc_auc_score


@torch.no_grad()
def q_values(agent, X, bs=4096):
    net = agent.online_net; net.eval()
    out = [net(torch.as_tensor(X[i:i + bs], device=device)).cpu() for i in range(0, len(X), bs)]
    net.train()
    return torch.cat(out).numpy()


def metrics_from_preds(y, p):
    K, n = len(class_names), NORMAL_IDX
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    P, R, F1, sup = prf_from_cm(cm)
    att = [i for i in range(K) if i != n]
    tn, fp = cm[n, n], cm[n].sum() - cm[n, n]                       # normal -> normal / normal -> any attack
    fn = cm[att, n].sum(); tp = cm[att].sum() - fn                  # attack -> normal / attack -> any attack
    far = fp / (fp + tn) if fp + tn else 0.0
    dr = tp / (tp + fn) if tp + fn else 0.0
    bprec = tp / (tp + fp) if tp + fp else 0.0
    return {"Acc": np.trace(cm) / cm.sum(), "MacroF1": F1.mean(), "FAR": far, "DR": dr,
            "BinAcc": (tp + tn) / cm.sum(), "BinPrec": bprec,
            "BinF1": 2 * bprec * dr / (bprec + dr) if bprec + dr else 0.0,
            "F1_per_class": dict(zip(class_names, F1))}


def calib_objective(m, cap):
    return m["MacroF1"] - 10.0 * max(0.0, m["FAR"] - cap)


def fit_offsets(Q, y, cap):
    """Coordinate ascent over per-class offsets on VALIDATION. b = 0 is always a candidate, so the objective
    can only stay the same or improve on validation."""
    scale = float(Q.std()) or 1.0
    b = np.zeros(Q.shape[1])
    best = calib_objective(metrics_from_preds(y, Q.argmax(1)), cap)
    for _ in range(CALIB_PASSES):
        for c in range(Q.shape[1]):
            for v in CALIB_GRID * scale:
                trial = b.copy(); trial[c] = v
                obj = calib_objective(metrics_from_preds(y, (Q + trial).argmax(1)), cap)
                if obj > best + 1e-12:
                    best, b = obj, trial
    return b


def per_class_auc(Q, y):
    return {class_names[c]: float(roc_auc_score((y == c).astype(int), Q[:, c]))
            for c in range(Q.shape[1]) if 0 < (y == c).sum() < len(y)}


rows, auc_rows = [], []
print(f"Block 20 | runs tagged '{CALIB_TAG}' | test evaluated: {CALIB_EVAL_TEST}"
      + ("  <- DISCLOSE: follow-up second use of the test set" if CALIB_EVAL_TEST else ""))
for fn in sorted(os.listdir(CFG["RESULTS_DIR"]) if os.path.isdir(CFG["RESULTS_DIR"]) else []):
    if not (fn.endswith(".json") and "__seed" in fn) or fn.endswith("_TEST.json"):
        continue
    if not fn.split("__seed")[0].endswith(f"__{CALIB_TAG}"):
        continue
    with open(os.path.join(CFG["RESULTS_DIR"], fn)) as f:
        r = json.load(f)
    agent = ARCH[r["arch"]](input_dim, action_dim, device, class_weights_dp)
    agent.set_fl_parameters(torch.load(result_path(r["run"], r["seed"], "_best.pt")))
    Qv = q_values(agent, X_val)
    b = fit_offsets(Qv, y_val, CFG["FAR_CAP"])
    splits = [("val", Qv, y_val)] + ([("test", q_values(agent, X_test), y_test)] if CALIB_EVAL_TEST else [])
    for split, Q, y in splits:
        for mode, pred in (("raw", Q.argmax(1)), ("calibrated", (Q + b).argmax(1))):
            m = metrics_from_preds(y, pred)
            rows.append({"run": r["run"], "seed": r["seed"], "split": split, "mode": mode,
                         **{k: 100 * v for k, v in m.items() if k != "F1_per_class"},
                         **{f"F1_{k}": 100 * v for k, v in m["F1_per_class"].items()}})
        auc_rows.append({"run": r["run"], "seed": r["seed"], "split": split, **per_class_auc(Q, y)})

cdf, adf = pd.DataFrame(rows), pd.DataFrame(auc_rows)
if cdf.empty:
    print(f"No saved runs tagged '{CALIB_TAG}' in {CFG['RESULTS_DIR']}")
else:
    def pm(s):
        s = s.dropna()
        return "-" if s.empty else (f"{s.mean():.2f} ± {s.std(ddof=1):.2f}" if len(s) > 1 else f"{s.mean():.2f} (n=1)")
    show = "test" if CALIB_EVAL_TEST else "val"
    raw = cdf[(cdf["split"] == show) & (cdf["mode"] == "raw")]
    print(f"\n(a) BINARY IDS VIEW ({show}, attack vs normal) - report NEXT TO the 10-class results\n",
          raw.groupby("run")[["BinAcc", "BinPrec", "DR", "FAR", "BinF1", "Acc", "MacroF1"]].agg(pm).to_string())
    print(f"\n(b) PER-CLASS ROC-AUC from Q-values ({show}, mean over seeds)\n",
          adf[adf["split"] == show].groupby("run")[list(class_names)].mean().round(3).T.to_string())
    cols = ["Acc", "MacroF1", "FAR", "DR", "BinAcc"]
    for split in (["val", "test"] if CALIB_EVAL_TEST else ["val"]):
        t = cdf[cdf["split"] == split].groupby(["run", "mode"])[cols].agg(pm)
        print(f"\n(c) CALIBRATION ({split}) - offsets learned on validation only\n", t.to_string())
    # paired ADP-Meta minus fixed noise, after calibration
    cal = cdf[(cdf["split"] == show) & (cdf["mode"] == "calibrated")]
    pr = []
    for _, c in cal[cal["run"].str.startswith("dp_adpmeta_eps")].iterrows():
        f = cal[(cal["run"] == c["run"].replace("dp_adpmeta", "dp_fixed")) & (cal["seed"] == c["seed"])]
        if len(f):
            pr.append({"run": c["run"], "seed": c["seed"], "dMacroF1": c["MacroF1"] - f.iloc[0]["MacroF1"],
                       "dAcc": c["Acc"] - f.iloc[0]["Acc"], "dFAR": c["FAR"] - f.iloc[0]["FAR"]})
    if pr:
        pdt = pd.DataFrame(pr)
        print(f"\nADP-Meta minus fixed noise after calibration ({show}, paired by seed)\n",
              pdt.groupby("run").agg(dMacroF1=("dMacroF1", pm), dAcc=("dAcc", pm), dFAR=("dFAR", pm),
                                     seeds_ADP_better=("dMacroF1", lambda x: f"{int((x > 0).sum())}/{len(x)}")).to_string())
    cdf.to_csv(os.path.join(CFG["RESULTS_DIR"], f"postproc_{CALIB_TAG}.csv"), index=False)
    adf.to_csv(os.path.join(CFG["RESULTS_DIR"], f"auc_{CALIB_TAG}.csv"), index=False)
    print(f"\nSaved postproc_{CALIB_TAG}.csv and auc_{CALIB_TAG}.csv in {CFG['RESULTS_DIR']}")
