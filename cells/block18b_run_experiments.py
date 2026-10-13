# BLOCK 18b (v4): Run experiments (VALIDATION only). Finished runs are skipped, so re-running resumes.
# Default STAGE = "v4_val": ADP-Meta v4 (Option 2 - noise pacing + adaptive clip + meta-learned PUBLIC-SUBSPACE rank,
# Block 17d) vs BOTH fixed noise (dp_fixed) and the fair baseline fixed noise + fixed projection (dp_fixed_proj),
# frozen settings, seed 0, eps 8 and 1. VALIDATION only.
# Success rule written BEFORE running: ADP-Meta v4 must beat dp_fixed_proj by >= 2 val macro-F1 with FAR <= 10% at an
# eps; then "final_v4" (3 seeds, eps 8/3/1) decides (mean better AND better on >= 2 of 3 seeds). Gains of
# dp_fixed_proj over dp_fixed are credited to the projection, not to meta-learning.
# Any test numbers from v4 are a second, disclosed use of the test set (Chapter 6 v2 test results stay as reported).
# Other stages stay available unchanged: "v3_val" / "final_v3" (ADP-Meta v3, frozen at cd74dad), "eps16", "final", ...
# ADP-Meta v3 ("v3_val" / "final_v3") keeps its own pre-registered rule: beat fixed noise by >= 2 val macro-F1
# with FAR <= 10%, then mean better AND better on >= 2 of 3 seeds.
# Other choice: STAGE = "eps16" -> one more point on the privacy-utility curve (weaker privacy, eps = 16):
#   ADP-Meta v2 (frozen final settings, meta-trained for eps 16) vs fixed noise, seeds 0/1/2 (~1.5-2 h).
STAGE = "v4_val"

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
STAGES = {
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

for tag, over, seeds, exps in STAGES[STAGE]:
    CFG.clear(); CFG.update(_DEFAULT_CFG); CFG.update(over); CFG["TAG"] = tag
    print(f"\n##### Stage {STAGE} | variant '{tag or 'baseline'}' | changes: {over or 'none'}")
    run_all(exps, seeds)
CFG.clear(); CFG.update(_DEFAULT_CFG)
