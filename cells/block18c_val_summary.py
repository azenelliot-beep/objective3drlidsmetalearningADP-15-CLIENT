# BLOCK 18c (v4): Validation summary - SEND THIS OUTPUT BACK (no test numbers are involved)
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

    agg = vdf.groupby("run")[["MacroF1", "Acc", "FAR", "DR"]].agg(["mean", "std", "count"]).round(2)
    print("\nVALIDATION - mean / std / n over seeds\n", agg.to_string())
    vdf.to_csv(os.path.join(CFG["RESULTS_DIR"], "val_summary.csv"), index=False)
    import zipfile
    with zipfile.ZipFile("results_jsons.zip", "w", zipfile.ZIP_DEFLATED) as z:      # JSON + CSV only (no .pt)
        for fn in os.listdir(CFG["RESULTS_DIR"]):
            if fn.endswith((".json", ".csv")):
                z.write(os.path.join(CFG["RESULTS_DIR"], fn), fn)
    print("Download results_jsons.zip from the Output panel (JSON + CSV only).")
