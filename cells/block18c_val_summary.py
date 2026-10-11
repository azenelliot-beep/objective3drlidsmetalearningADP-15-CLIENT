# BLOCK 18c (v4 - C0): Validation summary - this is what to send back after every stage (no test numbers)
vdf = summarize_val()
if vdf.empty:
    print("No finished runs in", CFG["RESULTS_DIR"])
else:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    main = ["run", "seed", "best_round", "cap_met", "Acc", "MacroF1", "FAR", "DR", "eps"]
    print("VALIDATION - selected round per run\n", vdf[main].round(2).to_string(index=False))
    print("\nVALIDATION - per-class F1 at the selected round\n",
          vdf[["run", "seed"] + [c for c in vdf.columns if c.startswith("F1_")]].round(1).to_string(index=False))
    agg = vdf.groupby("run")[["MacroF1", "Acc", "FAR", "DR"]].agg(["mean", "std", "count"]).round(2)
    print("\nVALIDATION - mean / std / n over seeds\n", agg.to_string())
    vdf.to_csv(os.path.join(CFG["RESULTS_DIR"], "val_summary.csv"), index=False)
    import zipfile
    with zipfile.ZipFile("results_jsons.zip", "w", zipfile.ZIP_DEFLATED) as z:      # JSON + CSV only (no .pt)
        for fn in os.listdir(CFG["RESULTS_DIR"]):
            if fn.endswith((".json", ".csv")):
                z.write(os.path.join(CFG["RESULTS_DIR"], fn), fn)
    print("Download results_jsons.zip from the Output panel (JSON + CSV only).")
