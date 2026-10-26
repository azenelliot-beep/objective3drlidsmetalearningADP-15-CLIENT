"""Step 1 (Section 3.1) and Section 4.2: loading, pre-processing and splitting of the seven IoT datasets.

Order of operations (leakage-safe version of the paper's pipeline):
  1. load raw rows, map the raw attack labels to the classes of Tables 3-9 and drop the classes the paper
     excludes (row-wise, stateless);
  2. shuffle with the seed and cut the data into n_clients * max_iterations equal pieces (40); split each
     piece 75 % / 25 % at random (Section 3.1);
  3. FIT every data-dependent transform (column typing, empty-column removal, LabelEncoder, MinMaxScaler)
     on the union of the 75 % training parts only, then apply it to every part.
The paper pre-processes before splitting (Fig. 3); fitting on training rows only is the one deliberate
change, requested to avoid test-set leakage. The transforms themselves are the ones in Section 4.2.
"""
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------------------------------
# Label mappings. Class order = row order of the paper's per-dataset result table.
# -----------------------------------------------------------------------------------------------------
TON_IOT_MAP = {"normal": "Normal", "ddos": "DDoS", "scanning": "Reconnaissance", "backdoor": "Backdoor",
               "injection": "Injection", "password": "Password", "ransomware": "Ransomware", "xss": "XSS",
               "dos": "DoS", "mitm": "MITM"}

# IoT-23 (Section 4.2): HeartBeat, FileDownload and Mirai are not considered. Table 4 keeps exactly
# Normal, PortScan, C&C, Okiru, DDoS, Attack; every other detailed label (Torii, combined labels) is dropped.
IOT23_MAP = {"Benign": "Normal", "PartOfAHorizontalPortScan": "PortScan", "C&C": "C&C", "Okiru": "Okiru",
             "DDoS": "DDoS", "Attack": "Attack"}

BOT_IOT_MAP = {"DoS": "DoS", "Theft": "Theft", "DDoS": "DDoS", "Reconnaissance": "Reconnaissance",
               "Normal": "Normal"}


def cic_iot_2023_map(lbl):
    """CICIoT2023 34 labels -> Table 6 groups. Web-based and Brute Force attacks are dropped (Section 4.2).
    Spoofing = MITM-ArpSpoofing + DNS_Spoofing (Table 1: 486,504 = 307,593 + 178,911)."""
    s = str(lbl).strip()
    if s.startswith("DDoS-"):
        return "DDoS"
    if s.startswith("DoS-"):
        return "DoS"
    if s.startswith("Mirai-"):
        return "Mirai"
    if s == "BenignTraffic":
        return "Normal"
    if s in ("MITM-ArpSpoofing", "DNS_Spoofing"):
        return "Spoofing"
    if s.startswith("Recon-") or s == "VulnerabilityScan":
        return "Reconnaissance"
    return None   # SqlInjection, XSS, CommandInjection, BrowserHijacking, Backdoor_Malware, Uploading_Attack,
                  # DictionaryBruteForce


def cic_iomt_2024_map(filename):
    """CIC IoMT 2024 labels come from file names. Table 1/Table 7 have no Spoofing class -> ARP spoofing dropped."""
    s = os.path.basename(filename)
    if s.startswith("Benign"):
        return "Normal"
    if s.startswith("TCP_IP-DDoS"):
        return "DDoS"
    if s.startswith("TCP_IP-DoS"):
        return "DoS"
    if s.startswith("MQTT"):
        return "MQTT"
    if s.startswith("Recon"):
        return "Reconnaissance"
    return None


# RT-IoT2022 -> Table 8. Reconstructed from the Table 1 counts, which match the UCI per-label counts exactly:
# DDoS 95,193 = DOS_SYN_Hping 94,659 + DDOS_Slowloris 534; Spoofing 7,750 = ARP_poisioning;
# Normal 20,137 = MQTT_Publish + Thing_Speak + Wipro_bulb + the five NMAP_* scans; the 37
# Metasploit_Brute_Force_SSH rows are absent from Table 1 and are dropped.
RT_IOT_MAP = {"DOS_SYN_Hping": "DDoS", "DDOS_Slowloris": "DDoS", "ARP_poisioning": "Spoofing",
              "MQTT_Publish": "Normal", "Thing_Speak": "Normal", "Wipro_bulb": "Normal",
              "NMAP_UDP_SCAN": "Normal", "NMAP_XMAS_TREE_SCAN": "Normal", "NMAP_OS_DETECTION": "Normal",
              "NMAP_TCP_scan": "Normal", "NMAP_FIN_SCAN": "Normal"}

# Edge-IIoTset -> the dataset's five attack groups + Normal (Table 9; counts match Table 1 exactly).
EDGE_IIOT_MAP = {"Normal": "Normal",
                 "DDoS_UDP": "DDoS", "DDoS_ICMP": "DDoS", "DDoS_HTTP": "DDoS", "DDoS_TCP": "DDoS",
                 "Port_Scanning": "Information Gathering", "Vulnerability_scanner": "Information Gathering",
                 "Fingerprinting": "Information Gathering",
                 "SQL_injection": "Injection", "Uploading": "Injection", "XSS": "Injection",
                 "Backdoor": "Malware", "Password": "Malware", "Ransomware": "Malware",
                 "MITM": "MITM"}

BOT_IOT_COLUMNS = ["pkSeqID", "stime", "flgs", "proto", "saddr", "sport", "daddr", "dport", "pkts", "bytes",
                   "state", "ltime", "seq", "dur", "mean", "stddev", "smac", "dmac", "sum", "min", "max",
                   "soui", "doui", "sco", "dco", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate",
                   "drate", "attack", "category", "subcategory"]

TARGET = "__target__"


# -----------------------------------------------------------------------------------------------------
# Raw loaders: each returns a DataFrame of features + TARGET holding the *raw* label.
# -----------------------------------------------------------------------------------------------------
def _files(path, patterns):
    if os.path.isfile(path):
        return [path]
    out = []
    for p in patterns:
        out += glob.glob(os.path.join(path, "**", p), recursive=True)
    if not out:
        raise FileNotFoundError(f"no files matching {patterns} under {path}")
    return sorted(set(out))


def load_ton_iot(path):
    df = pd.read_csv(_files(path, ["*Train_Test_Network*.csv", "*.csv"])[0], low_memory=False)
    return df.rename(columns={"type": TARGET})


def _read_zeek(fn):
    fields, rows = None, []
    with open(fn, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                # the last Zeek column "tunnel_parents   label   detailed-label" uses 3 spaces as separator
                rows.append(re.split(r"\t| {3}", line.rstrip("\n")))
    if fields is None:
        raise ValueError(f"{fn}: missing #fields header")
    fields = [f for part in fields for f in re.split(r" {3}", part)]
    return pd.DataFrame(rows, columns=fields[: len(rows[0])] if rows else fields)


def load_iot_23(path):
    files = _files(path, ["conn.log.labeled", "*.labeled", "*.csv"])
    dfs = [pd.read_csv(f, low_memory=False) if f.endswith(".csv") else _read_zeek(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.replace("-", "_").replace(".", "_") for c in df.columns]
    detailed = df.pop("detailed_label").astype(str).str.strip()
    benign = df.pop("label").astype(str).str.strip().str.lower().eq("benign")
    df[TARGET] = np.where(benign | detailed.eq("-"), "Benign", detailed)
    return df


def load_bot_iot(path):
    dfs = []
    for f in _files(path, ["*.csv"]):
        head = pd.read_csv(f, nrows=0)
        has_header = "pkSeqID" in [c.strip().strip('"') for c in head.columns]
        d = pd.read_csv(f, header=0 if has_header else None, names=None if has_header else BOT_IOT_COLUMNS,
                        low_memory=False, sep=",")
        d.columns = [str(c).strip().strip('"') for c in d.columns]
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df.rename(columns={"category": TARGET})


def load_cic_iot_2023(path):
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in _files(path, ["*.csv"])], ignore_index=True)
    return df.rename(columns={"label": TARGET})


def load_cic_iomt_2024(path):
    dfs = []
    for f in _files(path, ["*.csv"]):
        d = pd.read_csv(f, low_memory=False)
        d[TARGET] = os.path.basename(f)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_rt_iot_2022(path):
    df = pd.read_csv(_files(path, ["*RT_IOT2022*", "*.csv"])[0], low_memory=False)
    return df.rename(columns={"Attack_type": TARGET})


def load_edge_iiot(path):
    df = pd.read_csv(_files(path, ["*DNN-EdgeIIoT*.csv", "*.csv"])[0], low_memory=False)
    return df.rename(columns={"Attack_type": TARGET})


def make_synthetic(n_rows=20000, n_features=24, seed=0):
    """Synthetic IoT-like table for smoke tests ONLY (never reported as a paper dataset).
    Contains the raw-data defects Section 4.2 handles: hex ports, stray strings, NaNs, an empty column,
    a categorical string column, a row-index column and imbalanced classes (one rare class)."""
    rng = np.random.default_rng(seed)
    classes = np.array(["normal", "ddos", "dos", "recon", "rare"])
    y = rng.choice(len(classes), size=n_rows, p=[0.5, 0.2, 0.15, 0.13, 0.02])
    centers = rng.normal(0, 0.3, size=(len(classes), n_features))     # overlapping classes
    X = centers[y] + rng.normal(0, 1.0, size=(n_rows, n_features))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df.loc[rng.random(n_rows) < 0.01, "f0"] = np.nan
    ports = (y * 1000 + rng.integers(0, 999, n_rows)).astype(object)
    hexmask = rng.random(n_rows) < 0.05
    ports[hexmask] = [hex(int(p)) for p in ports[hexmask]]
    ports[rng.random(n_rows) < 0.01] = "garbage"
    df["sport"] = ports
    df["proto"] = np.where(y == 0, "tcp", rng.choice(["udp", "icmp", "tcp"], n_rows))
    df["empty_col"] = np.nan
    df["Unnamed: 0"] = np.arange(n_rows)
    df[TARGET] = classes[y]
    return df


# -----------------------------------------------------------------------------------------------------
# Dataset specifications (Section 4.2)
# -----------------------------------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name: str
    loader: object
    label_map: object                 # dict or callable raw_label -> class name (None = drop row)
    classes: list                     # order of the paper's result table
    leak_columns: list = field(default_factory=list)   # other label columns, always removed
    drop_columns: list = field(default_factory=list)   # columns the paper says it dropped
    id_columns: list = field(default_factory=list)     # pure row / record identifiers
    numeric_columns: object = None    # list of named columns, "infer", or None
    drop_empty: bool = False          # "removed empty columns"
    minmax: bool = False              # MinMaxScaler used (CIC IoMT 2024, RT-IoT 2022, EdgeIIoT)


SPECS = {
    "ton_iot": DatasetSpec("TON-IoT", load_ton_iot, TON_IOT_MAP,
                           ["Normal", "DDoS", "Reconnaissance", "Backdoor", "Injection", "Password", "Ransomware",
                            "XSS", "DoS", "MITM"], leak_columns=["label"]),
    "iot_23": DatasetSpec("IoT-23", load_iot_23, IOT23_MAP,
                          ["Normal", "PortScan", "C&C", "Okiru", "DDoS", "Attack"],
                          drop_columns=["local_orig", "local_resp", "missed_bytes", "tunnel_parents"],
                          id_columns=["uid"], numeric_columns=["duration", "orig_bytes", "resp_bytes"]),
    "bot_iot": DatasetSpec("BoT-IoT", load_bot_iot, BOT_IOT_MAP,
                           ["DoS", "Theft", "DDoS", "Reconnaissance", "Normal"],
                           leak_columns=["attack", "subcategory"],
                           drop_columns=["smac", "dmac", "soui", "sco", "dco"], id_columns=["pkSeqID"],
                           numeric_columns=["sport", "dport"]),
    "cic_iot_2023": DatasetSpec("CIC IoT 2023", load_cic_iot_2023, cic_iot_2023_map,
                                ["DDoS", "DoS", "Mirai", "Normal", "Spoofing", "Reconnaissance"]),
    "cic_iomt_2024": DatasetSpec("CIC IoMT 2024", load_cic_iomt_2024, cic_iomt_2024_map,
                                 ["DDoS", "DoS", "MQTT", "Reconnaissance", "Normal"],
                                 drop_empty=True, minmax=True),
    "rt_iot_2022": DatasetSpec("RT-IoT 2022", load_rt_iot_2022, RT_IOT_MAP, ["DDoS", "Spoofing", "Normal"],
                               id_columns=["Unnamed: 0"], numeric_columns="infer", drop_empty=True, minmax=True),
    "edge_iiot": DatasetSpec("EdgeIIoT", load_edge_iiot, EDGE_IIOT_MAP,
                             ["DDoS", "Information Gathering", "Injection", "Malware", "MITM", "Normal"],
                             leak_columns=["Attack_label"], numeric_columns="infer", drop_empty=True,
                             minmax=True),
    "synthetic": DatasetSpec("Synthetic (smoke test only)", None,
                             {"normal": "Normal", "ddos": "DDoS", "dos": "DoS", "recon": "Recon", "rare": "Rare"},
                             ["Normal", "DDoS", "DoS", "Recon", "Rare"], id_columns=["Unnamed: 0"],
                             numeric_columns="infer", drop_empty=True, minmax=True),
}


# -----------------------------------------------------------------------------------------------------
# Cleaning helpers (stateless) and the fitted pre-processor
# -----------------------------------------------------------------------------------------------------
_HEX = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")


def to_number(series):
    """Hexadecimal strings -> integers, other strings -> 0, missing -> 0 (Section 4.2: BoT-IoT, RT-IoT, ...)."""
    num = pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    hexmask = num.isna() & s.str.match(_HEX)
    if hexmask.any():
        num.loc[hexmask] = s[hexmask].map(lambda v: int(v, 16)).astype(float)
    return num.fillna(0.0).astype(np.float64)


def _looks_numeric(series):
    vals = series.dropna()
    if vals.empty:
        return True
    if pd.api.types.is_numeric_dtype(vals):
        return True
    s = vals.astype(str).str.strip()
    ok = pd.to_numeric(s, errors="coerce").notna() | s.str.match(_HEX)
    return ok.mean() >= 0.5


def _is_empty(series):
    s = series.dropna()
    return s.empty or s.astype(str).str.strip().isin(["", "-"]).all()


class Preprocessor:
    """Fitted on training rows only: column typing, empty-column removal, LabelEncoder, MinMaxScaler."""

    def __init__(self, spec):
        self.spec = spec

    def fit(self, df):
        from sklearn.preprocessing import MinMaxScaler
        spec = self.spec
        cols = [c for c in df.columns if c != TARGET]
        self.dropped_empty = [c for c in cols if spec.drop_empty and _is_empty(df[c])]
        cols = [c for c in cols if c not in self.dropped_empty]
        if spec.numeric_columns == "infer":
            self.numeric = [c for c in cols if _looks_numeric(df[c])]
        else:
            named = set(spec.numeric_columns or [])
            self.numeric = [c for c in cols if c in named or pd.api.types.is_numeric_dtype(df[c])]
        self.categorical = [c for c in cols if c not in self.numeric]
        # LabelEncoder fitted on training values; values unseen in training get the next free code.
        self.codes = {c: {v: i for i, v in enumerate(sorted(df[c].fillna("0").astype(str).unique()))}
                      for c in self.categorical}
        self.columns = cols
        X = self._encode(df)
        self.scaler = MinMaxScaler().fit(X) if spec.minmax else None
        return self

    def _encode(self, df):
        out = np.empty((len(df), len(self.columns)), np.float64)
        for j, c in enumerate(self.columns):
            if c in self.numeric:
                out[:, j] = to_number(df[c]).to_numpy()
            else:
                codes = self.codes[c]
                out[:, j] = df[c].fillna("0").astype(str).map(codes).fillna(len(codes)).to_numpy(np.float64)
        return out

    def transform(self, df):
        X = self._encode(df)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X.astype(np.float32)

    def summary(self):
        return {"n_features": len(self.columns), "numeric": self.numeric, "categorical": self.categorical,
                "dropped_empty": self.dropped_empty, "minmax": self.scaler is not None}


# -----------------------------------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------------------------------
@dataclass
class FederatedData:
    pieces: dict           # (client_idx, iteration) -> dict(X_train, y_train, X_test, y_test)
    classes: list
    n_features: int
    info: dict


def _map_labels(df, spec):
    raw = df[TARGET]
    mapper = spec.label_map.get if isinstance(spec.label_map, dict) else spec.label_map
    mapped = raw.astype(str).str.strip().map(lambda v: mapper(v))
    keep = mapped.notna()
    df = df.loc[keep].copy()
    df[TARGET] = mapped[keep].to_numpy()
    return df, raw.loc[~keep].astype(str).value_counts().to_dict()


def prepare(dataset, data_path, cfg):
    spec = SPECS[dataset]
    df = make_synthetic(n_rows=cfg.synthetic_rows, seed=cfg.seed) if dataset == "synthetic" else spec.loader(data_path)
    n_raw = len(df)
    df, dropped_labels = _map_labels(df, spec)
    remove = [c for c in spec.leak_columns + spec.drop_columns if c in df.columns]
    if cfg.drop_row_id_columns:
        remove += [c for c in spec.id_columns if c in df.columns]
    df = df.drop(columns=remove)
    rng = np.random.default_rng(cfg.seed)
    if cfg.sample_frac < 1.0:
        df = df.iloc[np.sort(rng.choice(len(df), int(len(df) * cfg.sample_frac), replace=False))]
    if cfg.max_rows and len(df) > cfg.max_rows:
        df = df.iloc[np.sort(rng.choice(len(df), cfg.max_rows, replace=False))]
    df = df.reset_index(drop=True)

    # Section 3.1: random sampling into n_clients * iterations equal pieces, each split 75/25.
    n_pieces = cfg.n_clients * cfg.max_iterations
    order = rng.permutation(len(df))
    piece_idx = np.array_split(order, n_pieces)
    from sklearn.model_selection import train_test_split
    y_all = df[TARGET].to_numpy()
    split = {}
    for p, idx in enumerate(piece_idx):
        strat = y_all[idx] if cfg.stratify_piece_split else None
        tr, te = train_test_split(idx, test_size=cfg.test_fraction, random_state=cfg.seed + p, stratify=strat)
        split[p] = (np.sort(tr), np.sort(te))

    train_rows = np.sort(np.concatenate([tr for tr, _ in split.values()]))
    pre = Preprocessor(spec).fit(df.iloc[train_rows])
    X = pre.transform(df)
    class_index = {c: i for i, c in enumerate(spec.classes)}
    y = np.array([class_index[v] for v in y_all], np.int64)

    pieces = {}
    for p, (tr, te) in split.items():
        client, it = divmod(p, cfg.max_iterations)
        pieces[(client, it + 1)] = {"X_train": X[tr], "y_train": y[tr], "X_test": X[te], "y_test": y[te]}
    counts = pd.Series(y_all).value_counts().to_dict()
    info = {"dataset": spec.name, "rows_raw": n_raw, "rows_used": len(df), "class_counts": counts,
            "dropped_raw_labels": {str(k): int(v) for k, v in dropped_labels.items()},
            "removed_columns": remove, "sample_frac": cfg.sample_frac, "max_rows": cfg.max_rows,
            "n_pieces": n_pieces, "rows_per_piece": int(len(piece_idx[0])),
            "preprocessing": pre.summary()}
    return FederatedData(pieces, list(spec.classes), X.shape[1], info)


def data_fingerprint(fd):
    h = hashlib.sha1()
    for k in sorted(fd.pieces):
        for v in fd.pieces[k].values():
            h.update(np.ascontiguousarray(v).tobytes())
    h.update(json.dumps(fd.classes).encode())
    return h.hexdigest()[:12]
