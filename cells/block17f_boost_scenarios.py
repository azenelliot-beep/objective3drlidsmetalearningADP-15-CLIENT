# BLOCK 17f (boost scenarios): binary task, fewer/larger clients, 5% public pre-training - same four pillars
# DDQN detector, FedProx, record-level local DP-SGD + RDP accounting, ADP-Meta (meta-trained controller) are unchanged.
# What can change, per stage (CFG keys, defaults = the Chapter 6 setting):
#   BINARY=True          -> labels become attack / normal (all 9 attack types = "attack"); the DDQN has 2 actions.
#                           Reported ALONGSIDE the 10-class results, never instead of them.
#   NUM_CLIENTS_DP=N     -> the private training data is split over N clients instead of 15 (more records per client,
#                           so less noise per record for the same epsilon).
#   PUBLIC_FRAC=f        -> a stratified fraction f of the TRAINING records is treated as PUBLIC: it is removed from every
#                           client and used once, without DP, to pre-train the DDQN (PRETRAIN_STEPS TD steps, gamma 0);
#                           every method then starts federated DP training from that model. DISCLOSE: these records get
#                           NO privacy protection; epsilon is reported for the remaining private records.
# apply_scenario() rebinds the notebook's data globals (labels, class names, clients, datasets) and restores the
# originals exactly when called with the defaults, so every other stage is unaffected. Test set never used here.
from sklearn.model_selection import train_test_split

_ORIG_DATA = {}
_SCEN_KEYS = ("class_names", "X_dp", "y_dp", "y_val", "y_test", "val_dataset", "test_dataset", "NUM_CLIENTS",
              "action_dim", "NORMAL_IDX", "dp_client_indices", "class_weights_dp")
X_pub, y_pub = None, None
_PRETRAIN_CACHE = {}


def apply_scenario(binary=False, num_clients=None, public_frac=0.0):
    g = globals()
    if not _ORIG_DATA:
        _ORIG_DATA.update({k: g[k] for k in _SCEN_KEYS})
    g.update(_ORIG_DATA)                                              # always start from the original data
    g["X_pub"], g["y_pub"] = None, None
    if not (binary or num_clients or public_frac):
        return                                                       # default: the exact Chapter 6 setting
    if binary:
        n = _ORIG_DATA["class_names"].index("normal")
        to_bin = lambda y: (np.asarray(y) == n).astype(np.int64)     # attack -> 0, normal -> 1
        g["class_names"] = ["attack", "normal"]
        g["y_dp"], g["y_val"], g["y_test"] = to_bin(_ORIG_DATA["y_dp"]), to_bin(_ORIG_DATA["y_val"]), to_bin(_ORIG_DATA["y_test"])
        g["val_dataset"] = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(g["y_val"]).long())
        g["test_dataset"] = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(g["y_test"]).long())
        g["action_dim"], g["NORMAL_IDX"] = 2, 1
    if public_frac:
        idx = np.arange(len(g["y_dp"]))
        pub, priv = train_test_split(idx, train_size=public_frac, stratify=g["y_dp"], random_state=SPLIT_SEED)
        g["X_pub"], g["y_pub"] = g["X_dp"][pub], g["y_dp"][pub]
        g["X_dp"], g["y_dp"] = g["X_dp"][priv], g["y_dp"][priv]
    if num_clients:
        g["NUM_CLIENTS"] = int(num_clients)
    g["dp_client_indices"] = dirichlet_split(g["y_dp"], g["NUM_CLIENTS"], ALPHA, np.random.default_rng(SEED),
                                             MIN_PER_CLIENT)
    cw = compute_class_weight('balanced', classes=np.arange(len(g["class_names"])), y=g["y_dp"])
    g["class_weights_dp"] = (np.sqrt(cw) / np.sqrt(cw).mean()).astype(np.float32)
    print(f"Scenario: binary={binary} | clients={g['NUM_CLIENTS']} | public records={0 if g['X_pub'] is None else len(g['y_pub'])}"
          f" | private records={len(g['y_dp'])} | classes={g['class_names']}")


def scenario_from_cfg(cfg):
    apply_scenario(bool(cfg.get("BINARY", False)), cfg.get("NUM_CLIENTS_DP"), float(cfg.get("PUBLIC_FRAC", 0.0) or 0.0))


def public_init(seed):
    """Non-private pre-training on the public slice (TD loss, gamma 0, same reward and network as the DP clients)."""
    key = (seed, len(class_names), len(y_pub), CFG["PRETRAIN_STEPS"], CFG["LR"], CFG["BATCH"], CFG["REWARD_FA"],
           CFG["REWARD_MISS"], CFG["REWARD_POWER"], CFG["WEIGHTS_FROM"])
    if key not in _PRETRAIN_CACHE:
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        set_seed(seed)
        ag = SmallDDQN(input_dim, action_dim, device, reward_weights(), gamma=0.0, lr=CFG["LR"])
        X = torch.as_tensor(X_pub, dtype=torch.float32, device=device)
        Y = torch.as_tensor(y_pub, dtype=torch.long, device=device)
        net = ag.online_net
        for _ in range(int(CFG["PRETRAIN_STEPS"])):
            i = torch.randint(0, len(Y), (CFG["BATCH"],), device=device)
            x, yy = X[i], Y[i]
            a = ag.act(x)
            r = ag.reward(a, yy)
            q = net(x).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, r)
            ag.optimizer.zero_grad(set_to_none=True); loss.backward(); ag.optimizer.step()
            ag.steps += 1
        _PRETRAIN_CACHE[key] = ag.get_fl_parameters()
        print(f"Public pre-training (seed {seed}): {CFG['PRETRAIN_STEPS']} steps on {len(Y)} public records | "
              f"val macro-F1 = {100 * eval_metrics(ag, val_dataset)['Macro_F1']:.1f}")
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])                  # training random streams are not shifted
    return [t.clone() for t in _PRETRAIN_CACHE[key]]


def maybe_public_init(agent, seed):
    """Start `agent` from the public pre-trained model when PUBLIC_FRAC > 0 (no-op otherwise)."""
    if CFG.get("PUBLIC_FRAC") and X_pub is not None:
        agent.set_fl_parameters(public_init(seed))
