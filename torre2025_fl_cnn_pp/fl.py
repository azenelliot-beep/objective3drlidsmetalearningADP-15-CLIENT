"""Steps 2-9 of Fig. 3: CNN-based federated learning with (or without) the three PP layers.

Client  : Step 2 train L1 (lock + barrier)      -> Step 3 L2 = copy(L1); DP; DK offset; HE encrypt
Server  : Step 4 HE decrypt each client          -> Step 5 FedAvg       -> Step 6 + fixed offset; HE encrypt
Client  : Step 7 HE decrypt; - fixed offset; - DP noise -> Step 8 accuracies + convergence -> Step 9 next
"""
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import privacy as pp
from .metrics import overall_from_per_class, per_class_metrics
from .model import accuracy_with_weights, build_cnn, predict_classes, train_cnn

log = logging.getLogger("torre2025")


def _seed(cfg, *parts):
    return int(np.random.SeedSequence([cfg.seed, *parts]).generate_state(1)[0])


class ClientAgent:
    def __init__(self, idx, pieces, n_classes, n_features, cfg, use_pp, dh_party, fed, protected):
        self.idx, self.name = idx, f"C{idx + 1}"
        self.pieces, self.n_classes, self.n_features = pieces, n_classes, n_features
        self.cfg, self.use_pp, self.dh = cfg, use_pp, dh_party
        self.fed, self.protected = fed, protected      # exchanged model arrays; PP positions within them
        self.latency = cfg.latencies_s[idx]
        self.eval_model = None
        self.dp_noise = {}                 # Step 7: directory of DP noise, organised by iteration
        self.L1, self.L2 = {}, {}
        self.local_acc, self.fed_acc, self.cct, self.tfd = {}, {}, {}, {}
        self.max_diff = {}
        self.converged_at = None
        self.mech_counts = {"dp": 0, "dk": 0, "he_encrypt": 0, "he_decrypt": 0}

    # ------------------------------------------------------------------ Step 2
    def train(self, it, init_weights, lock, barrier):
        import keras
        piece = self.pieces[it]
        with lock:                                     # only one client inside the CNN at a time
            t0 = time.perf_counter()
            keras.utils.set_random_seed(_seed(self.cfg, self.idx, it))
            model = build_cnn(self.n_features, self.n_classes, self.cfg)
            if init_weights is not None:               # Step 9: start from the current federated L2
                model.set_weights([np.asarray(w, np.float32) for w in init_weights])
            start = [w.copy() for w in model.get_weights()]
            train_cnn(model, piece["X_train"], piece["y_train"], self.n_classes, self.cfg)
            L1 = model.get_weights()
            if self.eval_model is None:
                self.eval_model = build_cnn(self.n_features, self.n_classes, self.cfg)
            t_train = time.perf_counter() - t0
        barrier.wait()                                 # nobody continues until all clients have trained
        self.L1[it] = L1
        return start, t_train

    # ------------------------------------------------------------------ Step 3
    def encrypt(self, it, start, active_names, lock):
        cfg = self.cfg
        t0 = time.perf_counter()
        L2 = [np.asarray(self.L1[it][i], np.float64).copy() for i in self.fed]   # L2 = copy of L1
        payload = {"sender": self.name, "arrays": L2, "he_dir": None}
        if self.use_pp:
            prot = [L2[i] for i in self.protected]
            # DP (Algorithm 2)
            if cfg.dp_scale_mode == "fixed":
                scale = cfg.dp_fixed_scale
            else:
                delta = pp.max_abs_difference([L2[j] for j in self.protected],
                                              [start[self.fed[j]] for j in self.protected])
                scale = pp.dp_scale_from_sensitivity(delta, cfg.dp_epsilon)
            rng = np.random.default_rng(_seed(cfg, 1000 + self.idx, it))
            prot, noise = pp.add_dp_noise(prot, cfg.n_clients, cfg.dp_mu, scale, rng)
            self.dp_noise[it] = noise
            self.mech_counts["dp"] += 1
            # DK (Algorithm 3). Line 16: the deltas are updated before the next client runs the method.
            t_wait = time.perf_counter()
            with lock:
                t_wait = time.perf_counter() - t_wait
                o_tot = self.dh.offset_total(active_names)
                self.dh.update_deltas()
            prot = pp.add_offset(prot, o_tot)
            self.mech_counts["dk"] += 1
            # HE (Algorithm 4) with O_p = O_tot
            prot, he_dir = pp.he_encrypt(prot, o_tot, cfg.he_b_max, rng)
            self.mech_counts["he_encrypt"] += 1
            for j, i in enumerate(self.protected):
                L2[i] = prot[j]
            payload.update(he_dir=he_dir, dp_scale=scale, o_tot=o_tot)
        else:
            t_wait = 0.0
        t_enc = time.perf_counter() - t0 - t_wait
        return payload, t_enc

    # ------------------------------------------------------------------ Steps 7 and 8
    def receive(self, it, back, n_active):
        cfg = self.cfg
        L2 = [np.asarray(w, np.float64).copy() for w in back["arrays"]]   # exchanged arrays only
        if self.use_pp:
            prot = pp.he_decrypt([L2[i] for i in self.protected], back["he_dir"])      # Algorithm 5
            self.mech_counts["he_decrypt"] += 1
            prot = [a - back["fixed_offset"] for a in prot]                            # remove fixed offset
            share = 1.0 / n_active if cfg.dp_noise_removal == "own_share" else 1.0
            prot = [a - share * n for a, n in zip(prot, self.dp_noise[it])]            # remove DP noise
            for j, i in enumerate(self.protected):
                L2[i] = prot[j]
        model = [np.asarray(w, np.float64) for w in self.L1[it]]          # arrays not exchanged stay local
        for j, i in enumerate(self.fed):
            model[i] = L2[j]
        self.L2[it] = model
        piece = self.pieces[it]
        self.local_acc[it] = accuracy_with_weights(self.eval_model, self.L1[it], piece["X_test"], piece["y_test"])
        self.fed_acc[it] = accuracy_with_weights(self.eval_model, model, piece["X_test"], piece["y_test"])
        local = [self.L1[it][i] for i in self.fed]
        dw = max([float(np.max(np.abs(a - b))) for a, b in zip(L2, local) if pp.is_weight_array(a)], default=0.0)
        db = max([float(np.max(np.abs(a - b))) for a, b in zip(L2, local) if not pp.is_weight_array(a)], default=0.0)
        self.max_diff[it] = (dw, db)
        return bool(dw <= cfg.tolerance and db <= cfg.tolerance)


class ServerAgent:
    def __init__(self, cfg, use_pp, protected):
        self.cfg, self.use_pp, self.protected = cfg, use_pp, protected
        self.mech_counts = {"he_decrypt": 0, "fedavg": 0, "he_encrypt": 0}

    def aggregate(self, payloads):
        """Steps 4 and 5. Returns the federated average (with the DP noise still inside)."""
        models = []
        for p in payloads:
            arrays = list(p["arrays"])
            if self.use_pp:
                dec = pp.he_decrypt([arrays[i] for i in self.protected], p["he_dir"])       # Step 4
                self.mech_counts["he_decrypt"] += 1
                for j, i in enumerate(self.protected):
                    arrays[i] = dec[j]
            models.append(arrays)
        self.mech_counts["fedavg"] += 1
        return [np.mean([m[k] for m in models], axis=0) for k in range(len(models[0]))]     # Step 5

    def send_back(self, avg, rng):
        """Step 6, run for each client right before it re-obtains the federated L2."""
        if not self.use_pp:
            return {"arrays": avg}
        F = self.cfg.server_fixed_offset
        arrays = list(avg)
        prot = pp.add_offset([arrays[i] for i in self.protected], F)
        prot, he_dir = pp.he_encrypt(prot, F, self.cfg.he_b_max, rng)
        self.mech_counts["he_encrypt"] += 1
        for j, i in enumerate(self.protected):
            arrays[i] = prot[j]
        return {"arrays": arrays, "he_dir": he_dir, "fixed_offset": F}


def exchanged_arrays(cfg, n_features, n_classes):
    """Indices of the get_weights() arrays that form L2 (see Config.federated_arrays)."""
    model = build_cnn(n_features, n_classes, cfg)
    ws = model.weights
    if cfg.federated_arrays == "all":
        return list(range(len(ws)))
    if cfg.federated_arrays == "trainable":
        return [i for i, w in enumerate(ws) if w.trainable]
    if cfg.federated_arrays == "conv":
        conv = model.layers[1]
        return [i for i, w in enumerate(ws) if any(w is v for v in conv.weights)]
    raise ValueError(cfg.federated_arrays)


def run_federated(fd, cfg, use_pp, out_dir=None):
    """Runs Steps 2-9 for every iteration and returns a JSON-serialisable result dict."""
    n = cfg.n_clients
    names = [f"C{i + 1}" for i in range(n)]
    n_classes = len(fd.classes)
    fed = exchanged_arrays(cfg, fd.n_features, n_classes)
    protected = list(range(len(fed)))          # every exchanged array goes through DP, DK and HE
    dh = pp.diffie_hellman_setup(names, cfg.dh_curve, cfg.dh_delta_bits_slice) if use_pp else {}
    clients = [ClientAgent(i, {it: fd.pieces[(i, it)] for it in range(1, cfg.max_iterations + 1)},
                           n_classes, fd.n_features, cfg, use_pp, dh.get(names[i]), fed, protected) for i in range(n)]
    server = ServerAgent(cfg, use_pp, protected)
    active = list(range(n))
    init = None
    if cfg.common_initialization:
        import keras
        keras.utils.set_random_seed(_seed(cfg, 7))
        init = build_cnn(fd.n_features, n_classes, cfg).get_weights()
    federated = {i: init for i in range(n)}
    history, checks = [], []

    for it in range(1, cfg.max_iterations + 1):
        if not active:
            break
        lock, barrier = threading.Lock(), threading.Barrier(len(active))
        active_names = {names[i] for i in active}

        def client_round(i):
            c = clients[i]
            start, t_train = c.train(it, federated[i], lock, barrier)
            payload, t_enc = c.encrypt(it, start, active_names, lock)
            c.cct[it] = t_train + t_enc                    # client computation time (Steps 2-3)
            return payload

        with ThreadPoolExecutor(max_workers=len(active)) as ex:
            payloads = list(ex.map(client_round, active))

        t0 = time.perf_counter()
        avg = server.aggregate(payloads)
        rng = np.random.default_rng(_seed(cfg, 99, it))
        backs = {i: server.send_back(avg, rng) for i in active}
        server_time = time.perf_counter() - t0

        # Verification (not part of the protocol): the offsets must cancel in the average, i.e. the
        # server's average must equal mean(L1_i + DP_i) up to float64 rounding.
        expect = [np.mean([np.asarray(clients[i].L1[it][fed[k]], np.float64)
                           + (clients[i].dp_noise[it][protected.index(k)] if use_pp and k in protected else 0)
                           for i in active], axis=0) for k in range(len(avg))]
        checks.append(max(float(np.max(np.abs(a - b))) for a, b in zip(avg, expect)))

        # Simulated time to receive the federated data (PrivacyFL latency model, Section 4.4).
        slowest = max(clients[i].latency + clients[i].cct[it] + clients[i].latency for i in active)
        newly = []
        for i in active:
            c = clients[i]
            c.tfd[it] = slowest + server_time + c.latency
            converged = c.receive(it, backs[i], len(active))
            federated[i] = c.L2[it]
            history.append({"iteration": it, "client": c.name, "n_active": len(active),
                            "cct_s": c.cct[it], "tfd_s": c.tfd[it], "local_acc": c.local_acc[it],
                            "fed_acc": c.fed_acc[it], "max_abs_diff_weights": c.max_diff[it][0],
                            "max_abs_diff_biases": c.max_diff[it][1], "converged": converged,
                            "dp_scale": payloads[active.index(i)].get("dp_scale")})
            log.info("it %2d %s  L-A %.4f  F-A %.4f  CCT %.2fs  TFD %.2fs  max|dW| %.2e  max|dB| %.2e  %s",
                     it, c.name, c.local_acc[it], c.fed_acc[it], c.cct[it], c.tfd[it], *c.max_diff[it],
                     "CONVERGED -> drop client" if converged else "")
            if converged:
                c.converged_at = it
                newly.append(i)
            if out_dir and cfg.save_checkpoints:
                np.savez_compressed(os.path.join(out_dir, f"checkpoint_{c.name}_it{it:02d}.npz"),
                                    *[np.asarray(w, np.float32) for w in c.L2[it]])
        active = [i for i in active if i not in newly]

    # ---------------- final evaluation (Tables 3-10): each client's final federated model on its test parts
    y_true, y_pred = [], []
    per_client = []
    for c in clients:
        last = max(c.L2)
        c.eval_model.set_weights([np.asarray(w, np.float32) for w in c.L2[last]])
        X = np.concatenate([p["X_test"] for p in c.pieces.values()])
        y = np.concatenate([p["y_test"] for p in c.pieces.values()])
        y_true.append(y)
        y_pred.append(predict_classes(c.eval_model, X))
        its = sorted(c.L2)
        per_client.append({"client": c.name, "I#": c.converged_at or last, "converged": c.converged_at is not None,
                           "CCT_mean_s": float(np.mean([c.cct[t] for t in its])),
                           "TFD_mean_s": float(np.mean([c.tfd[t] for t in its])),
                           "L-A": 100 * c.local_acc[last], "F-A": 100 * c.fed_acc[last],
                           "mechanisms": c.mech_counts})
    rows = per_class_metrics(np.concatenate(y_true), np.concatenate(y_pred), fd.classes)
    return {"use_pp": use_pp, "per_class": rows, "overall": overall_from_per_class(rows),
            "clients": per_client, "history": history, "server_mechanisms": server.mech_counts,
            "exchanged_arrays": fed,
            "offset_cancellation_max_err": checks}
