# BLOCK 19 (v4 - C0): FINAL TEST - run ONCE, only after every ablation is frozen and the "final" stage is done.
# Evaluates each run's validation-selected model on the test set and reports mean +/- std over seeds.
# Nothing in this cell feeds back into any training or selection decision.
FINAL_TEST = False          # <- set True only for the single final evaluation
TEST_TAG = "bst10,bstbin"   # comma-separated stage tags to evaluate ("final" = v2 final stage, "v3", "v4")

if not FINAL_TEST:
    print("FINAL_TEST is False - test set not touched.")
else:
    rows = []
    for fn in sorted(os.listdir(CFG["RESULTS_DIR"])):
        if not is_run_file(fn):
            continue
        if TEST_TAG and not any(fn.split("__seed")[0].endswith(f"__{t}") for t in TEST_TAG.split(",")):
            continue
        with open(os.path.join(CFG["RESULTS_DIR"], fn)) as f:
            r = json.load(f)
        scenario_from_cfg(r.get("cfg", {}))          # the run's own data setting (Block 17f)
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
    apply_scenario()
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
