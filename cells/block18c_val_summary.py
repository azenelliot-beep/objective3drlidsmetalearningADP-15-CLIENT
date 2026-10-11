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

    # One-change-at-a-time comparison against the baseline dp_fixed eps 8 (same seed)
    base = vdf[vdf["run"] == "dp_fixed_eps8"]
    cand = vdf[vdf["run"].str.startswith("dp_fixed_eps8__")]
    if len(base) and len(cand):
        b = base.set_index("seed")
        rows = []
        for _, c in cand.iterrows():
            if c["seed"] not in b.index:
                continue
            bb = b.loc[c["seed"]]
            dF1, dFAR = c["MacroF1"] - bb["MacroF1"], c["FAR"] - bb["FAR"]
            rows.append({"variant": c["run"].split("__", 1)[1], "seed": c["seed"], "MacroF1": c["MacroF1"],
                         "dMacroF1": dF1, "Acc": c["Acc"], "dAcc": c["Acc"] - bb["Acc"], "FAR": c["FAR"],
                         "dFAR": dFAR, "cap_met": c["cap_met"],
                         "verdict": "IMPROVES" if (dF1 >= 2.0 and dFAR <= 1.0) else "no"})
        cmp_df = pd.DataFrame(rows).sort_values("dMacroF1", ascending=False)
        print(f"\nSWEEP vs baseline dp_fixed_eps8 (MacroF1={b['MacroF1'].iloc[0]:.2f}, FAR={b['FAR'].iloc[0]:.2f}) "
              f"- IMPROVES = +2 macro-F1 or more and FAR not worse by more than 1 point\n",
              cmp_df.round(2).to_string(index=False))
        cmp_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "sweep_vs_baseline.csv"), index=False)

    agg = vdf.groupby("run")[["MacroF1", "Acc", "FAR", "DR"]].agg(["mean", "std", "count"]).round(2)
    print("\nVALIDATION - mean / std / n over seeds\n", agg.to_string())
    vdf.to_csv(os.path.join(CFG["RESULTS_DIR"], "val_summary.csv"), index=False)
    import zipfile
    with zipfile.ZipFile("results_jsons.zip", "w", zipfile.ZIP_DEFLATED) as z:      # JSON + CSV only (no .pt)
        for fn in os.listdir(CFG["RESULTS_DIR"]):
            if fn.endswith((".json", ".csv")):
                z.write(os.path.join(CFG["RESULTS_DIR"], fn), fn)
    print("Download results_jsons.zip from the Output panel (JSON + CSV only).")
