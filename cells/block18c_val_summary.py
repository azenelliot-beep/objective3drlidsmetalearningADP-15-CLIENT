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
