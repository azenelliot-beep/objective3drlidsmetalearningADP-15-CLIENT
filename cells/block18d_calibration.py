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
