# Candidate package - repair log and mechanism tests (synthetic data only)

Received files are kept unchanged (`SHA256SUMS_received.txt`). Working copies:
`chapter6_ddqn_adp_candidate_repaired.ipynb` and `ddqn_private_patch_repaired.py`.

## Repair (1 line, method unchanged)
`patched_private_step.one_loss`: `F.smooth_l1_loss(qv[ai], ti)` raised
`RuntimeError: vmap ... data-dependent control flow` on the first private step, so no run could start.
Replaced with `F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ti)`: the same per-record
Huber loss on Q(x_i, a_i), written with vmap-safe indexing. Nothing else changed: same Double-DQN public-anchor
target, gamma 0.1, clipping, Gaussian noise, RDP accounting, FedProx, controller, stage `repair_check`.

## Mechanism tests (RUN_WITH_CLAUDE.md step 6) - `tests/test_candidate_mechanisms.py`
Executed on the repaired notebook with conftest's synthetic ToN-IoT-like data (CPU). 11/11 passed.
On the received (unrepaired) notebook 5/11 failed with the vmap error above.
| Test | What it checks |
|---|---|
| patch active | DPClient step/fit and ADPMetaV2.finish_episode are the patched ones; FAST off; weights from val; gamma 0.1; anchors = val half A |
| one private step | params change, one accountant step, finite grads, epsilon > 0 |
| empty batch | no record sampled -> step still accounted, gradient = pure noise with std sigma*C (within 5%) |
| positive Gaussian noise | sigma = 0 -> zero grad and epsilon None; noise std scales with sigma (x4 -> x4) |
| target-network influence | matched randomness reproduces exactly; gamma 0.1 vs 0 differ; perturbing the target net changes the update only when gamma > 0 |
| FedProx active | grad(mu=0.5) - grad(mu=0) = mu (w - w_global) exactly |
| finite gradients | 25 private steps, all grads and params finite |
| first-episode policy gradient | entropy coef 0, first episode -> nonzero policy-gradient norm, params change |
| frozen policy varies with state | action probabilities differ across states (this does NOT show beneficial adaptation) |
| repair stage end to end | tiny dp_fixed + dp_adpmeta: all rounds run, epsilon <= budget, budget spent |
| repair_check config | 15 clients, 20 rounds, gamma 0.1, FAST off, ref (256, 400, 1e-3) vs candidate (512, 200, 3e-4), seed 0, eps 8 |

Not done: the real-data `repair_check` run (needs the ToN-IoT CSV). No performance result exists yet.
