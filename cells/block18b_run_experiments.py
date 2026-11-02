# BLOCK 18b (v4): Run experiments (VALIDATION only). Finished runs are skipped, so re-running resumes.
# Default STAGE = "enh" (Block 17g) - REQUIRED FRAMEWORK: 15 clients, 20 rounds, agreed preprocessing, DDQN, FedProx,
# record-level local DP-SGD + Opacus RDP accounting, ADP with a meta-learned controller. Changes (fixed before running):
# reward weights from validation counts (no private statistic outside DP), server momentum 0.5 and server EMA 0.8
# (post-processing, epsilon unchanged), ADP-Meta v6 = meta-learned learning-rate x clip schedule at the fixed noise
# schedule. Runs: "enhw" = fixed noise with the weights fix only (ablation), "enh" = fixed noise and ADP-Meta v6 with
# everything, seed 0, eps 8 (~3-4 h on a T4). Then Block 18c/18d. If ADP >= fixed + 2 val macro-F1: stage "enh_final".
# Other stages unchanged: "final", "boost" (5 clients + public slice: NOT the required framework), "v3_quick", ...
STAGE = "enh"

DP8 = [("dp_fixed", 8.0)]
C0 = [("", {}, [0], [("fedavg_np", None), ("fedprox_np", None), ("dp_clip_nonoise", None),
                     ("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])]
SWEEP1 = [  # (TAG, one change, seeds, runs)
    ("W_val",      dict(WEIGHTS_FROM="val"),                [0], DP8),   # C1 weights from validation counts
    ("B1024_S100", dict(BATCH=1024, LOCAL_STEPS=100),       [0], DP8),   # C3 bigger batches, fewer steps
    ("B1024_S200", dict(BATCH=1024, LOCAL_STEPS=200),       [0], DP8),
    ("B2048_S100", dict(BATCH=2048, LOCAL_STEPS=100),       [0], DP8),
    ("CLIPx0.5",   dict(CLIP_MULT=0.5),                     [0], DP8),   # C3 clip norm
    ("CLIPx2",     dict(CLIP_MULT=2.0),                     [0], DP8),
    ("LR3e-3",     dict(LR=3e-3),                           [0], DP8),   # C3 learning rate
    ("FA0.5",      dict(REWARD_FA=0.5),                     [0], DP8),   # C4 false-alarm penalty
    ("MISS0.5",    dict(REWARD_MISS=0.5),                   [0], DP8),   # C4 missed-attack penalty
    ("POW1.0",     dict(REWARD_POWER=1.0),                  [0], DP8),   # C4 fully balanced weights
    ("MU0",        dict(MU=0.0),                            [0], DP8),   # C11 FedProx mu
    ("MU0.1",      dict(MU=0.1),                            [0], DP8),
]
MAIN = [("",   {},                        [0], [("dp_adpmeta", 8.0)]),     # before
        ("C5", dict(ADP_ACTIONS="pacing"),   [0], [("dp_adpmeta", 8.0)])]     # after: ADP-Meta v2
BASE2 = dict(CLIP_MULT=0.5, REWARD_FA=0.5)                                  # Sweep-1 winners combined
NEXT = [
    ("S2base",    BASE2,                                              [0], [("dp_fixed", 8.0)]),   # reference
    ("S2_ADPv2",  BASE2 | dict(ADP_ACTIONS="pacing"),                 [0], [("dp_adpmeta", 8.0)]), # ADP-Meta online
    ("S2_ADPmeta", BASE2 | dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10), [0], [("dp_adpmeta", 8.0)]),  # + meta-learning
    ("S2_CLIP0.25", BASE2 | dict(CLIP_MULT=0.25),                     [0], [("dp_fixed", 8.0)]),   # refinements (one change each)
    ("S2_FA1.0",  BASE2 | dict(REWARD_FA=1.0),                        [0], [("dp_fixed", 8.0)]),
    ("S2_MISS0.5", BASE2 | dict(REWARD_MISS=0.5),                     [0], [("dp_fixed", 8.0)]),
    ("S2_B1024",  BASE2 | dict(BATCH=1024, LOCAL_STEPS=200),          [0], [("dp_fixed", 8.0)]),
]
FINAL_CFG = BASE2 | dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10)        # frozen (CHANGELOG, Sweep 2)
V3_CFG = BASE2 | dict(ADP_ACTIONS="pacing_clip", ADP_META_EPISODES=24)      # ADP-Meta v3 (Block 17c)
V4_CFG = BASE2 | dict(ADP_ACTIONS="pacing_clip_rank", ADP_META_EPISODES=24) # ADP-Meta v4 (Block 17d, Option 2)
V4_PRIVATE = lambda eps_list: [(m, e) for e in eps_list for m in ("dp_fixed", "dp_fixed_proj", "dp_adpmeta")]
V4Q_CFG = V4_CFG | dict(META_EPS=(8.0,), ADP_META_EPISODES=12, TWIN_POOL=2, META_TIME_LIMIT_MIN=150)
V3Q_CFG = V3_CFG | dict(META_EPS=(8.0,), ADP_META_EPISODES=12, TWIN_POOL=2, META_TIME_LIMIT_MIN=180)
DRIFT_CFG = V3_CFG | dict(DRIFT=True, ADP_ACTIONS="drift", META_EPS=(8.0,), ADP_META_EPISODES=12, TWIN_POOL=2,
                          META_TIME_LIMIT_MIN=180)
BOOST = dict(NUM_CLIENTS_DP=5, PUBLIC_FRAC=0.05)
# Enhanced stage (Block 17g) - REQUIRED FRAMEWORK (15 clients, 20 rounds, agreed preprocessing). Settings fixed
# before any run: weights from validation counts, server momentum 0.5, server EMA 0.8, ADP-Meta v6 (lr x clip).
ENH_W = FINAL_CFG | dict(WEIGHTS_FROM="val")                              # ablation: privacy fix only
ENH_CFG = ENH_W | dict(ENH=True, SERVER_MOMENTUM=0.5, SERVER_EMA=0.8, ADP_ACTIONS="lr_clip")
ENHQ_CFG = ENH_CFG | dict(META_EPS=(8.0,), ADP_META_EPISODES=12, TWIN_POOL=2, META_TIME_LIMIT_MIN=150)
ENHF_CFG = ENH_CFG | dict(ADP_META_EPISODES=24, TWIN_POOL=4)               # META_EPS = (8, 3, 1)
STAGES = {
    "enh": [("enhw", ENH_W, [0], [("dp_fixed", 8.0)]),
            ("enh", ENHQ_CFG, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "enh_final": [("enhf", ENHF_CFG, [0, 1, 2], [
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0), ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0)])],
    "boost": [("bst10", FINAL_CFG | BOOST, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)]),
              ("bstbin", FINAL_CFG | BOOST | dict(BINARY=True), [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "boost3": [("bst10", FINAL_CFG | BOOST, [0, 1, 2], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)]),
               ("bstbin", FINAL_CFG | BOOST | dict(BINARY=True), [0, 1, 2], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "drift_quick": [("drq", DRIFT_CFG, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "drift_val": [("drq", DRIFT_CFG, [0, 1, 2], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "v3_quick": [("v3q", V3Q_CFG, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "calib": [("final", FINAL_CFG, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])],
    "v4_quick": [("v4q", V4Q_CFG, [0], V4_PRIVATE([8.0]))],
    "v4_val": [("v4", V4_CFG, [0], V4_PRIVATE([8.0, 1.0]))],
    "final_v4": [("v4", V4_CFG, [0, 1, 2], [("fedavg_np", None), ("fedprox_np", None)] + V4_PRIVATE([8.0, 3.0, 1.0]))],
    "v3_val": [("v3", V3_CFG, [0], [("dp_fixed", 8.0), ("dp_adpmeta", 8.0), ("dp_fixed", 1.0), ("dp_adpmeta", 1.0)])],
    "final_v3": [("v3", V3_CFG, [0, 1, 2], [
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0), ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0)])],
    "eps16": [("final", FINAL_CFG, [0, 1, 2], [("dp_fixed", 16.0), ("dp_adpmeta", 16.0)])],
    "next": NEXT,
    "main": MAIN,
    "auto": C0 + SWEEP1,
    "C0_baseline": C0,
    "sweep1": SWEEP1,
    # ADP-Meta v2 (run after the fixed-noise settings are chosen; add the accepted settings to cfg)
    "C5_pacing_online": [("C5", dict(ADP_ACTIONS="pacing"), [0], [("dp_adpmeta", 8.0)])],
    "C6_meta": [("C6", dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10), [0], [("dp_adpmeta", 8.0)])],
    # FINAL: frozen settings, 3 paired seeds, eps 1/3/8 (Chapter 5 reference rows are a separate optional stage)
    "final": [("final", FINAL_CFG, [0, 1, 2], [
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0), ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0)])],
    "ch5_reference": [("ch5", {}, [0, 1, 2], [("fedavg_ch5", None), ("fedprox_ch5", None)])],
}

def import_previous_results(dst):
    """Copy results (*.json, *.pt) from an attached earlier notebook output into dst (never overwrites)."""
    import glob, shutil
    os.makedirs(dst, exist_ok=True)
    n = 0
    for src in glob.glob("/kaggle/input/**/results/*", recursive=True):
        fn = os.path.basename(src)
        if fn.endswith((".json", ".pt")) and not fn.endswith("_ckpt.pt") and not os.path.exists(os.path.join(dst, fn)):
            shutil.copy(src, os.path.join(dst, fn)); n += 1
    print(f"Imported {n} result files from earlier notebook outputs into {dst}")

if STAGE == "calib":
    import_previous_results(_DEFAULT_CFG["RESULTS_DIR"])

for tag, over, seeds, exps in STAGES[STAGE]:
    CFG.clear(); CFG.update(_DEFAULT_CFG); CFG.update(over); CFG["TAG"] = tag
    scenario_from_cfg(CFG)                     # Block 17f: data for this variant (default = Chapter 6 setting)
    print(f"\n##### Stage {STAGE} | variant '{tag or 'baseline'}' | changes: {over or 'none'}")
    if STAGE == "calib":                       # frozen final runs: an existing (imported) result is used as it is
        for seed in seeds:
            for exp in exps:
                if os.path.exists(result_path(run_name(*exp), seed)):
                    print(f">>> USE existing {run_name(*exp)} seed {seed}")
                else:
                    run_all([exp], [seed])
    else:
        run_all(exps, seeds)
CFG.clear(); CFG.update(_DEFAULT_CFG)
apply_scenario()                               # restore the Chapter 6 data for the cells below
