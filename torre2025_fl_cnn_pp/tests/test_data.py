"""Loader tests on tiny fake files that follow each public dataset's file format and column names."""
import numpy as np
import pandas as pd
import pytest

from torre2025_fl_cnn_pp.config import Config
from torre2025_fl_cnn_pp.data import BOT_IOT_COLUMNS, SPECS, Preprocessor, prepare, to_number

CFG = Config().with_overrides(max_iterations=2, n_clients=2)
N = 400


def _rng():
    return np.random.default_rng(0)


def _write_ton(d):
    r = _rng()
    lab = r.choice(["normal", "ddos", "scanning", "mitm", "xss"], N)
    pd.DataFrame({"ts": np.arange(N), "src_ip": r.choice(["1.1.1.1", "2.2.2.2"], N), "src_port": r.integers(0, 9, N),
                  "proto": r.choice(["tcp", "udp"], N), "duration": r.random(N), "dns_query": "-",
                  "label": (lab != "normal").astype(int), "type": lab}).to_csv(d / "Train_Test_Network.csv", index=False)
    return d / "Train_Test_Network.csv"


def _write_iot23(d):
    r = _rng()
    det = r.choice(["-", "PartOfAHorizontalPortScan", "Okiru", "C&C-HeartBeat", "DDoS", "Attack", "C&C"], N)
    lab = np.where(det == "-", "benign", "Malicious")
    cols = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
            "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
            "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents   label   detailed-label"]
    lines = ["#separator \\x09", "#fields\t" + "\t".join(cols), "#types\t..."]
    for i in range(N):
        dur = "-" if i % 7 == 0 else f"{r.random():.3f}"
        vals = [str(i), f"C{i}", "1.1.1.1", str(r.integers(1, 9)), "2.2.2.2", "23", "tcp", "-", dur, "-", "12",
                "S0", "-", "-", "0", "S", "1", "40", "0", "0", f"-   {lab[i]}   {det[i]}"]
        lines.append("\t".join(vals))
    p = d / "conn.log.labeled"
    p.write_text("\n".join(lines) + "\n")
    return d


def _write_bot(d):
    r = _rng()
    cat = r.choice(["DoS", "DDoS", "Reconnaissance", "Normal", "Theft"], N)
    df = pd.DataFrame({c: r.random(N) for c in BOT_IOT_COLUMNS})
    df["pkSeqID"] = np.arange(N)
    df["sport"] = [hex(v) if i % 5 == 0 else ("abc" if i % 11 == 0 else str(v)) for i, v in enumerate(r.integers(1, 999, N))]
    df["dport"] = r.integers(0, 99, N).astype(str)
    df["saddr"] = r.choice(["a", "b"], N)
    for c in ["smac", "dmac", "soui", "sco", "dco", "doui"]:
        df[c] = np.nan
    df["category"], df["attack"], df["subcategory"] = cat, (cat != "Normal").astype(int), "x"
    df.to_csv(d / "UNSW_2018_IoT_Botnet_Dataset_1.csv", index=False, header=False)   # full release: no header
    return d


def _write_cic23(d):
    r = _rng()
    labels = ["DDoS-ICMP_Flood", "DoS-UDP_Flood", "Mirai-greeth_flood", "BenignTraffic", "DNS_Spoofing",
              "Recon-PortScan", "SqlInjection", "DictionaryBruteForce", "MITM-ArpSpoofing", "VulnerabilityScan"]
    pd.DataFrame({"flow_duration": r.random(N), "Rate": r.random(N), "label": r.choice(labels, N)}).to_csv(
        d / "part-00000.csv", index=False)
    return d


def _write_iomt(d):
    r = _rng()
    for f in ["TCP_IP-DDoS-ICMP1_train.pcap.csv", "TCP_IP-DoS-SYN_test.pcap.csv", "MQTT-DDoS-Connect_Flood_train.pcap.csv",
              "Recon-Port_Scan_train.pcap.csv", "Benign_train.pcap.csv", "ARP_Spoofing_train.pcap.csv"]:
        pd.DataFrame({"Header_Length": r.random(80), "Protocol Type": r.random(80), "empty": np.nan}).to_csv(d / f, index=False)
    return d


def _write_rt(d):
    r = _rng()
    labels = ["DOS_SYN_Hping", "DDOS_Slowloris", "ARP_poisioning", "MQTT_Publish", "NMAP_UDP_SCAN",
              "Thing_Speak", "Metasploit_Brute_Force_SSH"]
    df = pd.DataFrame({"id.orig_p": [hex(v) if v % 9 == 0 else v for v in r.integers(1, 999, N)],
                       "proto": r.choice(["tcp", "udp"], N), "flow_duration": r.random(N), "blank": np.nan,
                       "Attack_type": r.choice(labels, N)})
    df.to_csv(d / "RT_IOT2022.csv")          # writes the unnamed index column, like the UCI file
    return d / "RT_IOT2022.csv"


def _write_edge(d):
    r = _rng()
    labels = ["Normal", "DDoS_UDP", "Port_Scanning", "SQL_injection", "Backdoor", "MITM", "Fingerprinting"]
    at = r.choice(labels, N)
    pd.DataFrame({"tcp.len": [("0x1f" if i % 13 == 0 else str(i % 50)) for i in range(N)],
                  "http.request.method": r.choice(["GET", "0", "POST"], N), "mqtt.topic": np.nan,
                  "Attack_label": (at != "Normal").astype(int), "Attack_type": at}).to_csv(
        d / "DNN-EdgeIIoT-dataset.csv", index=False)
    return d


WRITERS = {"ton_iot": _write_ton, "iot_23": _write_iot23, "bot_iot": _write_bot, "cic_iot_2023": _write_cic23,
           "cic_iomt_2024": _write_iomt, "rt_iot_2022": _write_rt, "edge_iiot": _write_edge}
LEAKS = {"ton_iot": ["label"], "bot_iot": ["attack", "subcategory", "pkSeqID"], "edge_iiot": ["Attack_label"],
         "iot_23": ["uid", "local_orig", "tunnel_parents"], "rt_iot_2022": ["Unnamed: 0"]}


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_loader_and_split(name, tmp_path):
    path = WRITERS[name](tmp_path)
    fd = prepare(name, str(path), CFG)
    assert fd.classes == SPECS[name].classes
    assert len(fd.pieces) == CFG.n_clients * CFG.max_iterations
    feats = fd.info["preprocessing"]["numeric"] + fd.info["preprocessing"]["categorical"]
    for leak in LEAKS.get(name, []):
        assert leak not in feats
    for piece in fd.pieces.values():
        n_tr, n_te = len(piece["y_train"]), len(piece["y_test"])
        assert abs(n_te / (n_tr + n_te) - 0.25) < 0.05
        assert np.isfinite(piece["X_train"]).all() and np.isfinite(piece["X_test"]).all()
    if SPECS[name].minmax:
        Xtr = np.concatenate([p["X_train"] for p in fd.pieces.values()])
        assert Xtr.min() >= -1e-6 and Xtr.max() <= 1 + 1e-6


def test_dropped_classes(tmp_path):
    def sub(name):
        (tmp_path / name).mkdir()
        return tmp_path / name
    fd = prepare("rt_iot_2022", str(_write_rt(sub("rt"))), CFG)
    assert set(fd.info["dropped_raw_labels"]) == {"Metasploit_Brute_Force_SSH"}
    fd = prepare("cic_iot_2023", str(_write_cic23(sub("cic"))), CFG)
    assert set(fd.info["dropped_raw_labels"]) == {"SqlInjection", "DictionaryBruteForce"}
    fd = prepare("iot_23", str(_write_iot23(sub("iot23"))), CFG)
    assert set(fd.info["dropped_raw_labels"]) == {"C&C-HeartBeat"}
    fd = prepare("cic_iomt_2024", str(_write_iomt(sub("iomt"))), CFG)
    assert set(fd.info["dropped_raw_labels"]) == {"ARP_Spoofing_train.pcap.csv"}


def test_hex_and_strings_to_numbers():
    s = pd.Series(["0x1f", "12", "abc", None, "0X0A"])
    assert to_number(s).tolist() == [31.0, 12.0, 0.0, 0.0, 10.0]


def test_preprocessor_is_fitted_on_training_rows_only():
    spec = SPECS["synthetic"]
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0], "c": ["x", "y", "x"], "__target__": ["Normal"] * 3})
    test = pd.DataFrame({"a": [4.0], "c": ["unseen"], "__target__": ["Normal"]})
    pre = Preprocessor(spec).fit(train)
    Xte = pre.transform(test)
    assert Xte[0, 0] == pytest.approx(2.0)        # scaled with the training range -> outside [0, 1]
    assert Xte[0, 1] == pytest.approx(2.0)        # unseen category -> next free code (2), scaled by train max 1
