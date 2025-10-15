from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError
from pymatching import Matching


def create_surface_code_circuit(distance=3):
    """
    Build a distance-d rotated surface code circuit with (3 * distance) rounds of syndrome extraction.
    Each round measures all X and Z stabilizers.
    Returns (qc, X_checks, Z_checks, n_data, n_rounds).
    """

    # ----- Data layout -----
    # For d=3:
    #  d0 d1 d2
    #  d3 d4 d5
    #  d6 d7 d8
    n_data = 9

    # Stabilizers (same as before)
    X_interior = [[0,1,3,4],[4,5,7,8]]
    Z_interior = [[3,4,6,7],[1,2,4,5]]
    X_border   = [[1,2],[6,7]]
    Z_border   = [[0,3],[5,8]]
    X_checks = X_interior + X_border
    Z_checks = Z_interior + Z_border

    # Number of syndrome-extraction rounds
    n_rounds = 3 * distance

    # ----- Quantum + Classical registers -----
    q_data = QuantumRegister(n_data, "d")
    q_ax   = QuantumRegister(len(X_checks), "ax")
    q_az   = QuantumRegister(len(Z_checks), "az")

    # Create separate classical registers for each round so we can compare syndromes across time
    c_x_rounds = [ClassicalRegister(len(X_checks), f"mx_r{r}") for r in range(n_rounds)]
    c_z_rounds = [ClassicalRegister(len(Z_checks), f"mz_r{r}") for r in range(n_rounds)]

    qc = QuantumCircuit(q_data, q_ax, q_az, *c_z_rounds, *c_x_rounds,
                        name=f"d{distance}_surface_code_{n_rounds}r")

    # ----- Main multi-round loop -----
    for r in range(n_rounds):
        # --- X stabilizer measurement ---
        for a_idx, ds in enumerate(X_checks):
            anc = q_ax[a_idx]
            qc.reset(anc)
            qc.h(anc)
            for qd in ds:
                qc.cx(anc, q_data[qd])
            qc.h(anc)
            qc.measure(anc, c_x_rounds[r][a_idx])

        # --- Z stabilizer measurement ---
        for a_idx, ds in enumerate(Z_checks):
            anc = q_az[a_idx]
            qc.reset(anc)
            for qd in ds:
                qc.cx(q_data[qd], anc)
            qc.measure(anc, c_z_rounds[r][a_idx])

        qc.barrier()

    return qc, X_checks, Z_checks, n_data, n_rounds



def run_with_errors(
    qc,
    p1=1e-3,            # 1-qubit depolarizing error prob per 1q gate
    p2=2e-3,            # 2-qubit depolarizing error prob per 2q gate
    p_meas=5e-3,        # measurement bit-flip prob
    p_reset=5e-3,       # reset flips to wrong state with this prob
    shots=10000,
    seed=1234,
    basis_gates=("id","x","y","z","h","s","sdg","t","tdg","rx","ry","rz","cx"),
):
    """
    Runs an existing QuantumCircuit `qc` under a simple local noise model and returns counts.
    Does NOT modify `qc`.
    """
    # --- Build noise model ---
    noise = NoiseModel()

    # 1-qubit depolarizing on all 1q basis gates
    err1 = depolarizing_error(p1, 1)
    for g in ["id","x","y","z","h","s","sdg","t","tdg","rx","ry","rz"]:
        noise.add_all_qubit_quantum_error(err1, g)

    # 2-qubit depolarizing on CX
    err2 = depolarizing_error(p2, 2)
    noise.add_all_qubit_quantum_error(err2, "cx")

    # Measurement error: classical flip with prob p_meas
    meas_error = ReadoutError([[1 - p_meas, p_meas],
                               [p_meas, 1 - p_meas]])
    noise.add_all_qubit_readout_error(meas_error)

    # Reset error: wrong reset with prob p_reset (|0>↔|1>)
    reset_error = pauli_error([("I", 1 - p_reset), ("X", p_reset)])
    noise.add_all_qubit_quantum_error(reset_error, "reset")

    # --- Simulate with Aer ---
    backend = AerSimulator(noise_model=noise, seed_simulator=seed)
    tqc = transpile(qc, backend=backend, basis_gates=list(basis_gates), optimization_level=0)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)
    return counts



def plot_ler_vs_p_from_stats(stats_by_p, title="Surface code d=3: LER vs p", per_round=True):
    """
    Plot LER curves from precomputed stats.

    Parameters
    ----------
    stats_by_p : dict[float -> stats] | list[tuple(float, stats)]
        Each `stats` is the dict returned by `decode_counts_and_tally_logicals_normal`,
        containing at least 'rate_X', 'rate_Z', and 'rate_either'.
        Example entry:
          stats_by_p[0.01] = {
              'rate_X': 1.2e-3,
              'rate_Z': 1.1e-3,
              'rate_either': 2.3e-3,
              ...
          }

    title : str
        Title for the plot.

    Returns
    -------
    (fig, ax, df) : (matplotlib.figure.Figure, matplotlib.axes.Axes, pandas.DataFrame)
        df has columns: ['p', 'rate_X', 'rate_Z', 'rate_either', 'shots', 'logical_X_fails',
                         'logical_Z_fails', 'either_fails'] (when present in stats).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Normalize input to a sorted list of (p, stats)
    if isinstance(stats_by_p, dict):
        items = sorted(stats_by_p.items(), key=lambda kv: kv[0])
    else:
        items = sorted(list(stats_by_p), key=lambda kv: kv[0])

    # Extract arrays (include optional fields if present)
    ps = np.array([p for p, _ in items], dtype=float)
    rate_X = np.array([s.get('rate_X', np.nan) for _, s in items], dtype=float)
    rate_Z = np.array([s.get('rate_Z', np.nan) for _, s in items], dtype=float)
    rate_E = np.array([s.get('rate_either', np.nan) for _, s in items], dtype=float)
    shots  = np.array([s.get('shots', np.nan) for _, s in items], dtype=float)
    failX  = np.array([s.get('logical_X_fails', np.nan) for _, s in items], dtype=float)
    failZ  = np.array([s.get('logical_Z_fails', np.nan) for _, s in items], dtype=float)
    failE  = np.array([s.get('either_fails', np.nan) for _, s in items], dtype=float)

    rate_X_per_round = np.array([s.get('rate_X_per_round', np.nan) for _, s in items], dtype=float)
    rate_Z_per_round = np.array([s.get('rate_Z_per_round', np.nan) for _, s in items], dtype=float)
    rate_E_per_round = np.array([s.get('rate_either_per_round', np.nan) for _, s in items], dtype=float)

    # Build a tidy dataframe for convenience
    df = pd.DataFrame({
        "p": ps,
        "rate_X": rate_X,
        "rate_Z": rate_Z,
        "rate_either": rate_E,
        "rate_X_per_round": rate_X_per_round,
        "rate_Z_per_round": rate_Z_per_round,
        "rate_E_per_round": rate_E_per_round,
        "shots": shots,
        "logical_X_fails": failX,
        "logical_Z_fails": failZ,
        "either_fails": failE,
    })

    # Plot (log-log)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if per_round:
        ax.loglog(ps, rate_X_per_round, marker="o", label="Logical X rate")
        ax.loglog(ps, rate_Z_per_round, marker="s", label="Logical Z rate")
        ax.loglog(ps, rate_E_per_round, marker="^", label="Either logical (X or Z)")
    else:
        ax.loglog(ps, rate_X, marker="o", label="Logical X rate")
        ax.loglog(ps, rate_Z, marker="s", label="Logical Z rate")
        ax.loglog(ps, rate_E, marker="^", label="Either logical (X or Z)")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate (per round)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()

    return fig, ax, df



# =======================
# New decoders






def _checks_touching_qubit(checks, n_data):
    """For each data qubit q, return the (sorted) list of check indices that include q."""
    touches = [[] for _ in range(n_data)]
    for ci, S in enumerate(checks):
        for q in S:
            touches[q].append(ci)
    return [sorted(v) for v in touches]

def _spacetime_H(checks, n_data, T, include_time_boundaries=True):
    """
    Build a space-time parity matrix H_det for detection events.
    Rows  : (t, check_index) for t=1..T (events between round t-1 and t).
    Cols  : error mechanisms:
            - data error on data qubit q in layer t, t=1..T
              -> flips the two checks touching q at layer t (one if spatial boundary)
            - measurement error on check i at round t, t=1..T-1
              -> flips (i at layer t) and (i at layer t+1)
            - (optional) time-boundary errors at t=0 and t=T on check i
              -> singleton columns that flip (i at layer 1) and (i at layer T), respectively
    """
    nC = len(checks)
    def row(t, i):  # 1..T, 0..nC-1
        return (t - 1) * nC + i
    n_rows = T * nC

    # Map which checks each data qubit touches
    touch = _checks_touching_qubit(checks, n_data)

    cols = []

    # --- Data-error columns (spatial edges) ---
    for t in range(1, T + 1):
        for neigh in touch:
            if len(neigh) == 2:
                i, j = neigh
                cols.append([row(t, i), row(t, j)])
            elif len(neigh) == 1:
                i = neigh[0]           # spatial boundary
                cols.append([row(t, i)])
            else:
                cols.append([])        # (shouldn't happen for d=3 data qubits)

    data_cols = len(cols)

    # --- Measurement-error columns (time edges between layers) ---
    for t in range(1, T):               # rounds 1..T-1 connect layers t and t+1
        for i in range(nC):
            cols.append([row(t, i), row(t + 1, i)])

    meas_internal_cols = len(cols) - data_cols

    # --- Time-boundary columns (first & last layer singletons) ---
    meas_tbound_cols = 0
    if include_time_boundaries:
        # t = 0 boundary -> flips layer 1 node; t = T boundary -> flips layer T node
        for i in range(nC):
            cols.append([row(1, i)])    # first layer singleton
        for i in range(nC):
            cols.append([row(T, i)])    # last layer singleton
        meas_tbound_cols = 2 * nC

    # Assemble H
    H = np.zeros((n_rows, len(cols)), dtype=np.uint8)
    for c, idxs in enumerate(cols):
        for r in idxs:
            H[r, c] ^= 1

    meta = dict(
        nC=nC,
        data_cols=data_cols,
        meas_internal_cols=meas_internal_cols,
        meas_tbound_cols=meas_tbound_cols,
        total_cols=len(cols),
        include_time_boundaries=include_time_boundaries,
    )
    return H, meta


def build_spacetime_decoders(X_checks, Z_checks, n_data, n_rounds):
    """
    Build PyMatching decoders for space-time detection events.
    n_rounds = total rounds of stabilizer measurement; T = n_rounds-1 detection layers.
    Returns: dec_Xst, dec_Zst, meta dict with shapes/mappings.
    """
    assert n_rounds >= 2, "Need at least 2 rounds to form detection events."
    T = n_rounds - 1

    # H for detection events over Z checks (decode X errors)
    H_Z_det, nCz = _spacetime_H(Z_checks, n_data, T)
    dec_Xst = Matching(H_Z_det)

    # H for detection events over X checks (decode Z errors)
    H_X_det, nCx = _spacetime_H(X_checks, n_data, T)
    dec_Zst = Matching(H_X_det)

    meta = dict(T=T, nCz=nCz, nCx=nCx,
                cols_X = H_Z_det.shape[1], cols_Z = H_X_det.shape[1])
    return dec_Xst, dec_Zst, meta

def _chunks_to_round_arrays(key, n_rounds):
    """
    Parse a multi-round count key into ordered round arrays.
    We added classical registers as: *all* mz_r0..mz_r{T} then *all* mx_r0..mx_r{T},
    and Qiskit prints the *last* regs on the left. If your add-order differs,
    adjust this parser accordingly.
    """
    parts = key.split()
    # Example order if you built qc as  (..., *c_z_rounds, *c_x_rounds):
    # printed key: <mx_r{T}> <mx_r{T-1}> ... <mx_r0>  <mz_r{T}> <mz_r{T-1}> ... <mz_r0>
    # so we separate, reverse, and convert each chunk
    mx_chunks = parts[:n_rounds]
    mz_chunks = parts[n_rounds:]

    mx_chunks = mx_chunks[::-1]   # r0..r{T}
    mz_chunks = mz_chunks[::-1]

    mx_rounds = [np.fromiter(map(int, s[::-1]), dtype=np.uint8) for s in mx_chunks]
    mz_rounds = [np.fromiter(map(int, s[::-1]), dtype=np.uint8) for s in mz_chunks]
    return mx_rounds, mz_rounds

def _detection_events(rounds):
    """Compute detection events d[t] = s[t] XOR s[t-1] for t=1..T."""
    return [rounds[t] ^ rounds[t-1] for t in range(1, len(rounds))]

def checks_to_H(checks, n_cols):
    H = np.zeros((len(checks), n_cols), dtype=np.uint8)
    for i, S in enumerate(checks):
        H[i, S] = 1
    return H


def decode_counts_spacetime(
    counts,
    dec_Xst, dec_Zst,
    X_checks, Z_checks, n_data, n_rounds,
    X_L_support, Z_L_support,
    H_X=None, H_Z=None
):
    """
    Space-time decoding from multi-round counts.
    - Builds detection events for X and Z checks.
    - Decodes with space-time matchers dec_Xst (for X errors) and dec_Zst (for Z errors).
    - Aggregates estimated *data* errors across time and reports logical failure rates,
      using a stabilizer-invariant logical test (mod out stabilizers).
    """
    T = n_rounds - 1
    nCz = len(Z_checks)
    nCx = len(X_checks)

    # Build H once if not provided
    if H_X is None:
        H_X = checks_to_H(X_checks, n_data)  # rows = X checks
    if H_Z is None:
        H_Z = checks_to_H(Z_checks, n_data)  # rows = Z checks

    shots = 0
    failX = failZ = either = 0

    # Column layout in H_det: [data errors (T*n_data)] + [meas errors ((T-1)*nC)]
    def split_cols(nC):
        data_cols = T * n_data
        meas_cols = (T - 1) * nC
        return data_cols, meas_cols

    data_cols_Z, _ = split_cols(nCz)  # for dec_Xst (built from Z checks)
    data_cols_X, _ = split_cols(nCx)  # for dec_Zst (built from X checks)

    for key, freq in counts.items():
        shots += freq

        # 1) Parse rounds (keys are "<mx_rT> ... <mx_r0> <mz_rT> ... <mz_r0>" given your register order)
        mx_rounds, mz_rounds = _chunks_to_round_arrays(key, n_rounds)

        # 2) Detection events
        dX_layers = _detection_events(mx_rounds)  # length T, each len nCx
        dZ_layers = _detection_events(mz_rounds)  # length T, each len nCz

        # 3) Flatten to feed PyMatching
        dX = np.concatenate(dX_layers).astype(np.uint8)  # shape T*nCx
        dZ = np.concatenate(dZ_layers).astype(np.uint8)  # shape T*nCz

        # 4) Decode detection events
        est_X = dec_Xst.decode(dZ)  # [data-X (T*n_data)] + [meas-Z ((T-1)*nCz)]
        est_Z = dec_Zst.decode(dX)  # [data-Z (T*n_data)] + [meas-X ((T-1)*nCx)]

        # 5) Aggregate *data* error estimates across time back to data qubits (mod 2 over t)
        x_hat_data = np.zeros(n_data, dtype=np.uint8)
        z_hat_data = np.zeros(n_data, dtype=np.uint8)
        # X decoder: first T*n_data columns are data-X at (q,t)
        for t in range(T):
            off = t * n_data
            x_hat_data ^= est_X[off:off + n_data].astype(np.uint8)
        # Z decoder: first T*n_data columns are data-Z at (q,t)
        for t in range(T):
            off = t * n_data
            z_hat_data ^= est_Z[off:off + n_data].astype(np.uint8)

        # 6) NEW: stabilizer-invariant logical test (detects ANY logical representative)
        logical_X, logical_Z = logical_parities_from_residual(
            x_residual=x_hat_data,   # X-type residual across data (after time aggregation)
            z_residual=z_hat_data,   # Z-type residual
            H_X=H_X, H_Z=H_Z,
            X_L=X_L_support, Z_L=Z_L_support
        )

        failX += logical_X * freq
        failZ += logical_Z * freq
        either += (1 if (logical_X or logical_Z) else 0) * freq

    return dict(
        shots=shots,
        logical_X_fails=int(failX),
        logical_Z_fails=int(failZ),
        either_fails=int(either),
        # per-experiment (what you have already)
        rate_X=(failX/shots) if shots else 0.0,
        rate_Z=(failZ/shots) if shots else 0.0,
        rate_either=(either/shots) if shots else 0.0,
        # NEW: per-round rates (what literature usually plots)
        rate_X_per_round=(failX/(shots*T)) if shots and T>0 else 0.0,
        rate_Z_per_round=(failZ/(shots*T)) if shots and T>0 else 0.0,
        rate_either_per_round=(either/(shots*T)) if shots and T>0 else 0.0,
        meta=dict(T=T)
    )



import numpy as np

def _rref_mod2(A):
    """RREF over GF(2). Returns (R, pivots) where R rows span rowspan(A) and pivots are pivot column indices."""
    A = (A.copy() & 1).astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        # find a row with a 1 in column c at/after r
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        pivots.append(c)
        # eliminate other 1s in column c
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        r += 1
        if r == m:
            break
    return A, pivots

def reduce_mod_stabilizers(e, H):
    """
    Reduce e (length n_data) modulo rowspan(H) over GF(2) using RREF(H).
    Returns a canonical representative with zeros in all pivot columns.
    """
    e = (np.asarray(e, dtype=np.uint8) & 1).ravel()
    H = (np.asarray(H, dtype=np.uint8) & 1)
    if H.size == 0:
        return e
    R, pivots = _rref_mod2(H)    # R has same row-span as H
    e_work = e.copy()
    for row_i, c in enumerate(pivots):
        if e_work[c]:            # cancel pivot column using the pivot row
            e_work ^= R[row_i, :]
    return (e_work & 1).astype(np.uint8)

def logical_parities_from_residual(x_residual, z_residual, H_X, H_Z, X_L, Z_L):
    """Stabilizer-invariant logical check."""
    x_red = reduce_mod_stabilizers(x_residual, H_Z)   # mod Z-stabilizers
    z_red = reduce_mod_stabilizers(z_residual, H_X)   # mod X-stabilizers

    n = len(x_red)
    X_L_vec = np.zeros(n, dtype=np.uint8)
    Z_L_vec = np.zeros(n, dtype=np.uint8)
    X_L_vec[X_L] = 1
    Z_L_vec[Z_L] = 1

    logical_Z = int(np.dot(x_red, Z_L_vec) % 2)
    logical_X = int(np.dot(z_red, X_L_vec) % 2)
    return logical_X, logical_Z
