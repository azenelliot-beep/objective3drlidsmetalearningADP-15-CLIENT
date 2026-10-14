# ============================ CELL 1: BLOCK 18c (with the fixed summary reader) ============================
# Fixed helpers (override the old versions already loaded in your session)
def is_run_file(fn):
    """True only for a run's own result file '<run>__seed<k>.json' (not _TEST / _CALIB / meta logs / CSVs)."""
    import re
    return re.fullmatch(r".+__seed\d+\.json", fn) is not None


def summarize_val(results_dir=None):
    """Validation-only summary of every finished run (mean +/- std over seeds). Never reads test results."""
    rows = []
    d = results_dir or CFG["RESULTS_DIR"]
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if is_run_file(fn):
            with open(os.path.join(d, fn)) as f:
                r = json.load(f)
            if "val_at_best" not in r:                    # not a finished run result (e.g. a foreign file)
                print(f"summarize_val: skipping {fn} (no val_at_best)"); continue
            v = r["val_at_best"]
            rows.append({"run": r["run"], "seed": r["seed"], "best_round": r["best_round"], "cap_met": r["cap_met"],
                         "Acc": 100 * v["Accuracy"], "MacroF1": 100 * v["Macro_F1"], "FAR": 100 * v["FAR"],
                         "DR": 100 * v["DR"], "eps": r.get("final_epsilon"),
                         **{f"F1_{k}": 100 * x for k, x in v["F1_per_class"].items()}})
    return pd.DataFrame(rows)


# BLOCK 18c (v4): Validation summary - SEND THIS OUTPUT BACK (no test numbers are involved)
REF = "dp_fixed_eps8__S2base"      # reference run for the "vs reference" table (fixed noise, same settings)
vdf = summarize_val()
if vdf.empty:
    print("No finished runs in", CFG["RESULTS_DIR"])
else:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    main = ["run", "seed", "best_round", "cap_met", "Acc", "MacroF1", "FAR", "DR", "eps"]
    print("VALIDATION - selected round per run\n", vdf[main].round(2).to_string(index=False))
    print("\nVALIDATION - per-class F1 at the selected round\n",
          vdf[["run", "seed"] + [c for c in vdf.columns if c.startswith("F1_")]].round(1).to_string(index=False))

    # Before/after: every tagged variant vs its untagged baseline run (same method, eps and seed)
    rows = []
    for _, c in vdf[vdf["run"].str.contains("__")].iterrows():
        base_name, variant = c["run"].split("__", 1)
        bb = vdf[(vdf["run"] == base_name) & (vdf["seed"] == c["seed"])]
        if bb.empty:
            continue
        bb = bb.iloc[0]
        dF1, dFAR = c["MacroF1"] - bb["MacroF1"], c["FAR"] - bb["FAR"]
        rows.append({"baseline": base_name, "variant": variant, "seed": c["seed"],
                     "base_MacroF1": bb["MacroF1"], "MacroF1": c["MacroF1"], "dMacroF1": dF1,
                     "base_Acc": bb["Acc"], "Acc": c["Acc"], "dAcc": c["Acc"] - bb["Acc"],
                     "base_FAR": bb["FAR"], "FAR": c["FAR"], "dFAR": dFAR, "cap_met": c["cap_met"],
                     "verdict": "IMPROVES" if (dF1 >= 2.0 and dFAR <= 1.0) else "no"})
    if rows:
        cmp_df = pd.DataFrame(rows).sort_values(["baseline", "dMacroF1"], ascending=[True, False])
        print("\nBEFORE / AFTER (validation) - IMPROVES = +2 macro-F1 or more and FAR not worse by more than 1 point\n",
              cmp_df.round(2).to_string(index=False))
        cmp_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "before_after.csv"), index=False)

    ref = vdf[vdf["run"] == REF]
    if len(ref):
        rr = ref.iloc[0]
        t = vdf[vdf["seed"] == rr["seed"]][["run", "best_round", "cap_met", "Acc", "MacroF1", "FAR", "DR", "eps"]].copy()
        t["dMacroF1"], t["dAcc"], t["dFAR"] = t["MacroF1"] - rr["MacroF1"], t["Acc"] - rr["Acc"], t["FAR"] - rr["FAR"]
        t["verdict"] = np.where((t["dMacroF1"] >= 2.0) & (t["dFAR"] <= 1.0), "IMPROVES", "no")
        t.loc[t["run"] == REF, "verdict"] = "REFERENCE"
        print(f"\nVS REFERENCE {REF} (validation)\n", t.sort_values("MacroF1", ascending=False).round(2).to_string(index=False))
        t.to_csv(os.path.join(CFG["RESULTS_DIR"], "vs_reference.csv"), index=False)

    # Paired ADP-Meta vs fixed noise (same tag, eps and seed)
    pairs = []
    for _, c in vdf[vdf["run"].str.startswith("dp_adpmeta_eps")].iterrows():
        for ref_method in ("dp_fixed", "dp_fixed_proj"):           # v4: also vs the fair projection baseline
            f = vdf[(vdf["run"] == c["run"].replace("dp_adpmeta", ref_method)) & (vdf["seed"] == c["seed"])]
            if len(f):
                f = f.iloc[0]
                pairs.append({"run": c["run"] + ("" if ref_method == "dp_fixed" else "  vs fixed_proj"),
                              "seed": c["seed"], "ADP_F1": c["MacroF1"], "fixed_F1": f["MacroF1"],
                              "dF1": c["MacroF1"] - f["MacroF1"], "ADP_FAR": c["FAR"], "fixed_FAR": f["FAR"]})
    if pairs:
        pdf_ = pd.DataFrame(pairs)
        print("\nADP-Meta vs fixed noise, paired by seed (validation)\n", pdf_.round(2).to_string(index=False))
        g = pdf_.groupby("run")["dF1"].agg(["mean", "std", "count", lambda x: int((x > 0).sum())])
        g.columns = ["mean_dF1", "std_dF1", "seeds", "seeds_ADP_better"]
        print(g.round(2).to_string())

    agg = vdf.groupby("run")[["MacroF1", "Acc", "FAR", "DR"]].agg(["mean", "std", "count"]).round(2)
    print("\nVALIDATION - mean / std / n over seeds\n", agg.to_string())
    vdf.to_csv(os.path.join(CFG["RESULTS_DIR"], "val_summary.csv"), index=False)
    import zipfile
    with zipfile.ZipFile("results_jsons.zip", "w", zipfile.ZIP_DEFLATED) as z:      # JSON + CSV only (no .pt)
        for fn in os.listdir(CFG["RESULTS_DIR"]):
            if fn.endswith((".json", ".csv")):
                z.write(os.path.join(CFG["RESULTS_DIR"], fn), fn)
    print("Download results_jsons.zip from the Output panel (JSON + CSV only).")

# ============================ CELL 2: BLOCK 18d ============================
# BLOCK 18d (v4): (A) weak-class confusion check + (B) detection-threshold calibration - EVALUATION ONLY
# Nothing here trains anything. DDQN, 15-client FedProx, record-level DP-SGD, RDP accounting, ADP pacing and the
# meta-learned controller are untouched; the models are the saved validation-selected ones (results/*_best.pt).
# Post-processing a DP model costs no privacy, so every reported epsilon stays exactly as it is.
#
# (A) For the weak classes: where do their validation rows go - correct, "normal" (a threshold problem, which (B) can
#     help) or another attack (a representation problem, which (B) cannot fix)?
# (B) Calibrated decision rule on the SAME Q-values: s(x) = Q_normal(x) - max_attack Q(x).
#       predict "normal" if s(x) >= tau, else the best attack class.   tau = 0 reproduces the original argmax rule.
#     tau is the largest value with validation FAR <= FAR_CAP (10%, the same cap as model selection), so the
#     detector uses its whole false-alarm budget. Chosen on VALIDATION only.
#     Honest check without the test set: 2-fold cross-fit on the fixed validation halves (tau from half A scored on
#     half B and vice versa). The original argmax numbers are always reported next to the calibrated ones.
#     CALIB_TEST = True scores the test set ONCE with the validation tau (a second, disclosed, evaluation-only use of
#     the test set; nothing is retrained or re-selected from it).
CALIB_TAG = "final"                 # runs of this stage tag are evaluated (ADP-Meta v2 final settings)
CALIB_TEST = True                   # see the header: disclosed, evaluation-only second use of the test set
WEAK = ["backdoor", "mitm", "ransomware"]
N_IDX = class_names.index("normal")


@torch.no_grad()
def q_values(agent, X, bs=4096):
    net = agent.online_net; net.eval()
    out = [net(torch.as_tensor(X[i:i + bs], device=device)).float().cpu() for i in range(0, len(X), bs)]
    net.train()
    return torch.cat(out).numpy()


def normal_margin(q):
    att = np.delete(q, N_IDX, axis=1)
    return q[:, N_IDX] - att.max(1)


def calibrated_pred(q, tau):
    att = q.copy(); att[:, N_IDX] = -np.inf
    return np.where(normal_margin(q) >= tau, N_IDX, att.argmax(1))


def choose_tau(q, y, cap):
    """Largest tau whose FAR (= share of normal rows with s < tau) is <= cap."""
    s = np.sort(normal_margin(q[y == N_IDX]))
    return float(s[min(int(np.floor(cap * len(s))), len(s) - 1)])


def metrics_from_pred(pred, y):
    m = MulticlassMetricsTracker(len(class_names), class_names)
    m.update(pred, y)
    ids = m.ids_metrics()
    P, R, F1, sup = prf_from_cm(m.cm)
    return {"Accuracy": ids["Accuracy"], "FAR": ids["IDS_FAR"], "DR": ids["IDS_DR"], "Macro_F1": float(F1.mean()),
            "Weighted_F1": float((F1 * sup / sup.sum()).sum()),
            "F1_per_class": {n: float(f) for n, f in zip(class_names, F1)}, "cm": m.cm}


def row_breakdown(cm, cls):
    i = class_names.index(cls); n = cm[i].sum()
    if n == 0:
        return {"class": cls, "support": 0}
    others = [(class_names[j], cm[i, j]) for j in np.argsort(-cm[i]) if j not in (i, N_IDX) and cm[i, j] > 0][:2]
    return {"class": cls, "support": int(n), "correct_%": 100 * cm[i, i] / n, "as_normal_%": 100 * cm[i, N_IDX] / n,
            "as_other_attack_%": 100 * (n - cm[i, i] - cm[i, N_IDX]) / n,
            "top_confusions": ", ".join(f"{c} {100 * k / n:.0f}%" for c, k in others)}


def run_calibration():
    ia, ib = val_halves()
    rows, weak_rows, pcls = [], [], []
    if not os.path.isdir(CFG["RESULTS_DIR"]):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    files = sorted(f for f in os.listdir(CFG["RESULTS_DIR"])
                   if is_run_file(f) and f.split("__seed")[0].endswith(f"__{CALIB_TAG}"))
    for fn in files:
        with open(os.path.join(CFG["RESULTS_DIR"], fn)) as f:
            r = json.load(f)
        if "arch" not in r:
            print(f"skip {fn}: not a run result"); continue
        pt = result_path(r["run"], r["seed"], "_best.pt")
        if not os.path.exists(pt):
            print(f"skip {r['run']} seed {r['seed']}: {pt} missing"); continue
        agent = ARCH[r["arch"]](input_dim, action_dim, device, class_weights_dp)
        agent.set_fl_parameters(torch.load(pt))
        qv = q_values(agent, X_val)
        orig_val = metrics_from_pred(calibrated_pred(qv, 0.0), y_val)
        # 2-fold cross-fit on the fixed validation halves (no test data)
        cf = np.empty(len(y_val), dtype=np.int64)
        cf[ib] = calibrated_pred(qv[ib], choose_tau(qv[ia], y_val[ia], CFG["FAR_CAP"]))
        cf[ia] = calibrated_pred(qv[ia], choose_tau(qv[ib], y_val[ib], CFG["FAR_CAP"]))
        cf_val = metrics_from_pred(cf, y_val)
        tau = choose_tau(qv, y_val, CFG["FAR_CAP"])                      # final threshold: full validation
        row = {"run": r["run"], "seed": r["seed"], "tau": tau}
        for k, m in (("val_argmax", orig_val), ("val_calib_crossfit", cf_val)):
            row.update({f"{k}_{x}": 100 * m[x] for x in ("DR", "FAR", "Macro_F1", "Accuracy")})
        for cls in WEAK + [c for c in class_names if c not in WEAK and c != "normal"
                           and orig_val["F1_per_class"][c] < 0.30]:
            weak_rows.append({"run": r["run"], "seed": r["seed"], "rule": "argmax", **row_breakdown(orig_val["cm"], cls)})
            weak_rows.append({"run": r["run"], "seed": r["seed"], "rule": "calibrated (cross-fit)",
                              **row_breakdown(cf_val["cm"], cls)})
        out = {"run": r["run"], "seed": r["seed"], "tau": tau, "far_cap": CFG["FAR_CAP"],
               "final_epsilon": r.get("final_epsilon"), "val_argmax": orig_val, "val_calibrated_crossfit": cf_val}
        if CALIB_TEST:
            qt = q_values(agent, X_test)
            t0, t1 = metrics_from_pred(calibrated_pred(qt, 0.0), y_test), metrics_from_pred(calibrated_pred(qt, tau), y_test)
            row.update({f"test_argmax_{x}": 100 * t0[x] for x in ("DR", "FAR", "Macro_F1", "Accuracy")})
            row.update({f"test_calib_{x}": 100 * t1[x] for x in ("DR", "FAR", "Macro_F1", "Accuracy")})
            out.update({"test_argmax": t0, "test_calibrated": t1})
            pcls.append({"run": r["run"], "seed": r["seed"],
                         **{f"{c}_argmax": 100 * t0["F1_per_class"][c] for c in class_names},
                         **{f"{c}_calib": 100 * t1["F1_per_class"][c] for c in class_names}})
        rows.append(row)
        with open(result_path(r["run"], r["seed"], "_CALIB.json"), "w") as f:
            json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return pd.DataFrame(rows), pd.DataFrame(weak_rows), pd.DataFrame(pcls)


cal_df, weak_df, pcls_df = run_calibration()
if cal_df.empty:
    print(f"No '{CALIB_TAG}' runs with saved models in {CFG['RESULTS_DIR']}.")
else:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print("(A) WEAK CLASSES - where their VALIDATION rows go (argmax vs calibrated)\n",
          weak_df.round(1).to_string(index=False))
    keep = ["run", "seed", "tau"] + [c for c in cal_df.columns if c.startswith("val_")]
    print("\n(B) CALIBRATION - VALIDATION (cross-fit = tau from the other half; no test data)\n",
          cal_df[keep].round(2).to_string(index=False))
    if CALIB_TEST:
        tcols = ["run", "seed"] + [c for c in cal_df.columns if c.startswith("test_")]
        print("\n(B) CALIBRATION - TEST (tau from validation; disclosed evaluation-only use)\n",
              cal_df[tcols].round(2).to_string(index=False))
        agg = cal_df.groupby("run")[[c for c in cal_df.columns if c.startswith("test_")]].agg(["mean", "std"])
        print("\nTEST mean / std over seeds\n", agg.round(2).to_string())
        print("\nTEST per-class F1 (argmax vs calibrated)\n", pcls_df.round(1).T.to_string())
        # Equal-FAR comparison: ADP-Meta vs fixed noise, paired by eps and seed
        pr = []
        for _, a in cal_df[cal_df["run"].str.startswith("dp_adpmeta_eps")].iterrows():
            f = cal_df[(cal_df["run"] == a["run"].replace("dp_adpmeta", "dp_fixed")) & (cal_df["seed"] == a["seed"])]
            if len(f):
                f = f.iloc[0]
                pr.append({"run": a["run"], "seed": a["seed"],
                           "dDR_argmax": a["test_argmax_DR"] - f["test_argmax_DR"],
                           "dDR_calib": a["test_calib_DR"] - f["test_calib_DR"],
                           "dF1_calib": a["test_calib_Macro_F1"] - f["test_calib_Macro_F1"],
                           "ADP_FAR_calib": a["test_calib_FAR"], "fixed_FAR_calib": f["test_calib_FAR"]})
        if pr:
            print("\nADP-Meta minus fixed noise at (approximately) EQUAL FAR - TEST\n",
                  pd.DataFrame(pr).round(2).to_string(index=False))
    cal_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "calibration_summary.csv"), index=False)
    weak_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "weak_class_breakdown.csv"), index=False)
    import zipfile
    with zipfile.ZipFile("calibration_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(CFG["RESULTS_DIR"]):
            if fn.endswith((".csv", "_CALIB.json")):
                z.write(os.path.join(CFG["RESULTS_DIR"], fn), fn)
    print("Download calibration_results.zip from the Output panel.")

# ============================ CELL 3: BLOCK 19 (optional - see note) ============================
# BLOCK 19 (v4 - C0): FINAL TEST - run ONCE, only after every ablation is frozen and the "final" stage is done.
# Evaluates each run's validation-selected model on the test set and reports mean +/- std over seeds.
# Nothing in this cell feeds back into any training or selection decision.
FINAL_TEST = False          # <- set True only for the single final evaluation
TEST_TAG = "final"          # only runs of this stage tag are evaluated ("final" = v2 final stage, "v3")

if not FINAL_TEST:
    print("FINAL_TEST is False - test set not touched.")
else:
    rows = []
    for fn in sorted(os.listdir(CFG["RESULTS_DIR"])):
        if not is_run_file(fn):
            continue
        if TEST_TAG and not fn.split("__seed")[0].endswith(f"__{TEST_TAG}"):
            continue
        with open(os.path.join(CFG["RESULTS_DIR"], fn)) as f:
            r = json.load(f)
        agent = ARCH[r["arch"]](input_dim, action_dim, device, class_weights_dp)
        agent.set_fl_parameters(torch.load(result_path(r["run"], r["seed"], "_best.pt")))
        t = eval_metrics(agent, test_dataset)
        with open(result_path(r["run"], r["seed"], "_TEST.json"), "w") as f:
            json.dump({"run": r["run"], "seed": r["seed"], "best_round": r["best_round"], "cap_met": r["cap_met"],
                       "final_epsilon": r.get("final_epsilon"), "test": t}, f, indent=2)
        rows.append({"run": r["run"], "seed": r["seed"], "cap_met": r["cap_met"], "eps": r.get("final_epsilon"),
                     "Acc": 100 * t["Accuracy"], "MacroF1": 100 * t["Macro_F1"], "WeightedF1": 100 * t["Weighted_F1"],
                     "FAR": 100 * t["FAR"], "DR": 100 * t["DR"],
                     **{f"F1_{k}": 100 * x for k, x in t["F1_per_class"].items()}})
    tdf = pd.DataFrame(rows)
    tdf["eps"] = pd.to_numeric(tdf["eps"])                        # None (non-private) -> NaN
    tdf.to_csv(os.path.join(CFG["RESULTS_DIR"], "test_all_runs.csv"), index=False)
    def pm(s):
        s = s.dropna()
        if s.empty: return "-"
        return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}" if len(s) > 1 else f"{s.mean():.2f} (n=1)"
    cols = ["Acc", "MacroF1", "WeightedF1", "FAR", "DR", "eps"]
    table = tdf.groupby("run").agg({c: pm for c in cols} | {"seed": "count", "cap_met": "all"})
    print("TEST - mean ± std over seeds (sample std, ddof=1)\n", table.to_string())
    per_class = tdf.groupby("run")[[c for c in tdf.columns if c.startswith("F1_")]].agg(pm)
    print("\nTEST - per-class F1 mean ± std\n", per_class.T.to_string())
    table.to_csv(os.path.join(CFG["RESULTS_DIR"], "test_table_mean_std.csv"))

    # Paired ADP-Meta vs fixed noise on TEST (same eps and seed)
    pr = []
    for _, c in tdf[tdf["run"].str.startswith("dp_adpmeta_eps")].iterrows():
        for ref_method in ("dp_fixed", "dp_fixed_proj"):           # v4: also vs the fair projection baseline
            f = tdf[(tdf["run"] == c["run"].replace("dp_adpmeta", ref_method)) & (tdf["seed"] == c["seed"])]
            if len(f):
                pr.append({"run": c["run"] + ("" if ref_method == "dp_fixed" else "  vs fixed_proj"),
                           "seed": c["seed"], "dMacroF1": c["MacroF1"] - f.iloc[0]["MacroF1"],
                           "dAcc": c["Acc"] - f.iloc[0]["Acc"], "dFAR": c["FAR"] - f.iloc[0]["FAR"]})
    if pr:
        pdt = pd.DataFrame(pr)
        paired = pdt.groupby("run").agg(dMacroF1=("dMacroF1", pm), dAcc=("dAcc", pm), dFAR=("dFAR", pm),
                                        seeds_ADP_better=("dMacroF1", lambda x: f"{int((x > 0).sum())}/{len(x)}"))
        print("\nTEST - ADP-Meta minus fixed noise, paired by seed (mean ± std)\n", paired.to_string())
        paired.to_csv(os.path.join(CFG["RESULTS_DIR"], "test_paired_adp_vs_fixed.csv"))
