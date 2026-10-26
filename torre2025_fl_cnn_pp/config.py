"""Central configuration for the reimplementation of

    Torre, Chennamaneni, Jo, Vyas, Sabrsula. "Toward Enhancing Privacy Preservation of a
    Federated Learning CNN Intrusion Detection System in IoT: Method and Empirical Study".
    ACM TOSEM 34(2), Article 53, 2025. https://doi.org/10.1145/3695998

Every field is tagged with its provenance:
    [PAPER §x]     value stated by the authors (section / algorithm / table)
    [PRIVACYFL]    taken from PrivacyFL (Mugunthan et al., ref [59]), the simulator the paper's DP and
                   DK layers are built on. This is NOT the authors' own code, which is not public.
    [ASSUMPTION]   not reported in the paper; chosen by this reimplementation (see README, "Assumptions")
"""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Config:
    # ---------------- FL topology (Section 3, 3.1, 3.9) ----------------
    n_clients: int = 4                      # [PAPER §3] four clients, one server
    max_iterations: int = 10                # [PAPER §3, §3.9] up to 10 FL iterations per client
    tolerance: float = 0.0005               # [PAPER §3.8] max |federated - local| for convergence
    # Communication latencies C1..C4 in seconds (Section 4.4). Applied in both directions.
    latencies_s: tuple = (0.6, 3.0, 5.0, 6.0)   # [PAPER §4.4]

    # ---------------- Data split (Section 3.1) ----------------
    # Dataset split at random into n_clients * max_iterations equal pieces (40), one per client-iteration,
    # each piece split 75 % train / 25 % test.
    test_fraction: float = 0.25             # [PAPER §3.1]
    stratify_piece_split: bool = False      # [ASSUMPTION] paper says "random sampling", no stratification
    sample_frac: float = 1.0                # 1.0 = full dataset as in the paper. <1 only for development runs
    max_rows: int = 0                       # 0 = no cap. >0 only for development runs (logged in results)
    synthetic_rows: int = 200000            # size of the synthetic smoke/verification table (not a paper dataset)
    minmax: object = None                   # None = per-dataset default (data.SPECS); True/False = override
    drop_row_id_columns: bool = True        # [ASSUMPTION] drop pure row-index / record-ID columns (leakage)

    # ---------------- CNN (Algorithm 1, Fig. 4) ----------------
    # "fig4": Dense -> Dropout(0.5) -> BatchNormalization -> softmax (Fig. 4 and the text, layers L1..L8).
    # "algorithm1_literal": Algorithm 1 as printed, where Dropout and BatchNormalization are instantiated
    #  but the output layer is called on dense_layer_1, so they are not part of the graph.
    architecture: str = "fig4"              # [ASSUMPTION] choice between two contradictory paper sources
    conv_filters: int = 32                  # [PAPER Alg.1 l.2]
    kernel_size: int = 3                    # [PAPER Alg.1 l.2]
    pool_size: int = 2                      # [PAPER Alg.1 l.3]
    dense_units: int = 128                  # [PAPER Alg.1 l.5]
    dropout: float = 0.5                    # [PAPER Alg.1 l.6]
    batch_size: int = 512                   # [PAPER Alg.1 l.11]
    epochs: int = 10                        # [PAPER Alg.1 l.11]
    # Iteration 1: True = every client starts from the same initial weights (FedAvg convention, refs [44, 51]);
    # False = each client initialises its own CNN independently. The paper does not say which.
    common_initialization: bool = True      # [ASSUMPTION]
    learning_rate: float = 0.001            # [ASSUMPTION] Keras Adam default ("optimizer=adam")
    fit_verbose: int = 0                    # paper uses verbose=1; 0 keeps logs readable (no effect on results)

    # ---------------- Differential privacy (Algorithm 2, §3.3, §3.10.1) ----------------
    dp_mu: float = 0.0                      # [PAPER §3.10.1] worked example: mean 0
    # lambda = Delta / x (Section 3.3). Neither Delta nor x (epsilon) is reported for the experiments.
    # dp_scale_mode = "fixed": lambda = dp_fixed_scale (0.1 = the paper's worked example, §3.10.1)
    # dp_scale_mode = "sensitivity": lambda = Delta / dp_epsilon with Delta = max_i |x_i - y_i| computed
    #                 between the client's new local weights and the weights it started the iteration from.
    dp_scale_mode: str = "fixed"            # [ASSUMPTION]
    dp_fixed_scale: float = 0.1             # [PAPER §3.10.1] example scale 0.1
    dp_epsilon: float = 1.0                 # [PRIVACYFL] config.epsilon = 1.0 (used only in "sensitivity" mode)
    # How the client removes DP noise from the federated L2 (Step 7):
    #   "own_share": subtract own_noise / n_active   [PRIVACYFL client_agent.receive_weights]
    #   "own_full" : subtract own_noise              (literal reading of §3.7 "subtraction of the noise")
    dp_noise_removal: str = "own_share"     # [ASSUMPTION, PrivacyFL-backed]
    # Which Keras model arrays form L2 = "weights and biases" (exchanged with the server, encrypted, averaged
    # and compared for convergence). Every exchanged array goes through all three PP layers.
    #   "trainable": the learnable parameters (Section 1.1: "the weights and biases are the learnable
    #                parameters"); BatchNormalization moving statistics are not exchanged and stay local.
    #   "all"      : everything in model.get_weights(), including BN moving mean/variance.
    #   "conv"     : only the Conv1D kernel (3, 1, 32) and bias (32) -- Fig. 1 shows exactly these shapes and
    #                Section 3.1 describes L2 as one 3-D weight array plus one 1-D bias array.
    federated_arrays: str = "trainable"     # [ASSUMPTION]

    # ---------------- Diffie-Hellman offsets (Algorithm 3, §3.10.2) ----------------
    dh_curve: str = "SECP384R1"             # [PRIVACYFL] ECDH + HKDF-SHA256
    dh_delta_bits_slice: int = 20           # [PRIVACYFL] delta = int(bin(seed)[:20], 2)

    # ---------------- Homomorphic layer (Algorithms 4, 5, §3.3, §3.6) ----------------
    he_b_max: int = 100                     # [ASSUMPTION] B_I drawn uniformly from 1..he_b_max
    server_fixed_offset: int = 7919         # [ASSUMPTION] "fixed offset value" of Step 6 is not reported

    # ---------------- Reproducibility / IO ----------------
    seed: int = 42                          # [ASSUMPTION]
    deterministic_ops: bool = False         # True -> tf.config.experimental.enable_op_determinism()
    results_dir: str = "results_torre2025"
    save_checkpoints: bool = True

    def to_dict(self):
        return asdict(self)

    def with_overrides(self, **kw):
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


# Smoke-test overrides: same code path, smaller compute. Used ONLY by tests and --smoke; never for results.
SMOKE_OVERRIDES = dict(epochs=2, batch_size=64, max_iterations=3, synthetic_rows=20000)
