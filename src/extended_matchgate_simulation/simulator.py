import os
from typing import *
import numpy as np
from qiskit.circuit import QuantumCircuit


# Treat “1”, “true” or “yes” (any case) as enabling Rust; everything else
# (including “0” or unset) leaves you in pure-Python mode.
_USE_RUST = os.getenv("USE_RUST_SIMULATOR", "0").lower() in ("1", "true", "yes")

if _USE_RUST:
    try:
        import emsim as _rust
    except ImportError as e:
        raise RuntimeError(
            "Rust backend requested (USE_RUST_SIMULATOR=1) but `emsim` is not installed."
        ) from e

# Pure‐Python fallbacks and helpers
from .simulator_py import (
    get_input_state as _py_get_input_state,
    get_controlled_phase_angles as _py_get_controlled_phase_angles,
    V_circuit as _py_V_circuit,
    calculate_expectation as _py_calculate_expectation,
    raw_estimate as _py_raw_estimate,
    get_circuit_data as _py_get_circuit_data,
    direct_calculation   as _py_direct_calculation,
    calculate_total_extent as _py_calculate_total_extent,
    calculate_number_of_samples as _py_calculate_number_of_samples,
    sample_bitstring as _py_sample_bitstring
)

def _extract_arrays(circuit: QuantumCircuit):
    """Build exactly the gate_types, params, qubits arrays plus angles & in_state."""
    angles = _py_get_controlled_phase_angles(circuit)
    in_state = _py_get_input_state(circuit)

    gate_types = []
    params = []
    qubits = []

    for instr in circuit.data:
        name = instr.operation.name
        if name == "cp":
            gate_types.append(1)
            θ = instr.operation.params[0]
            params.append([θ, 0.0])
            q1, q2 = instr.qubits[0]._index, instr.qubits[1]._index
            qubits.append([q1, q2])
        elif name == "xx_plus_yy":
            gate_types.append(2)
            θ, β = instr.operation.params
            params.append([θ, β])
            q1, q2 = instr.qubits[0]._index, instr.qubits[1]._index
            qubits.append([q1, q2])
        elif name == "p":
            gate_types.append(3)
            θ = instr.operation.params[0]
            params.append([θ, 0.0])
            q = instr.qubits[0]._index
            qubits.append([q, q])
        else:
            continue

    return (
        np.array(angles, dtype=np.float64),
        in_state,
        np.array(gate_types, dtype=np.uint8),
        np.array(params, dtype=np.float64),
        np.array(qubits, dtype=np.uint64),   # <-- cast to unsigned 64
    )


def get_circuit_data(circuit: QuantumCircuit) -> Tuple[float, int, list[float]]:
    return _py_get_circuit_data(circuit)


def get_input_state(circuit: QuantumCircuit) -> int:
    return _py_get_input_state(circuit)


def get_controlled_phase_angles(circuit: QuantumCircuit) -> list[float]:
    return _py_get_controlled_phase_angles(circuit)


def V_circuit(
    circuit: QuantumCircuit,
    x: int,
    c_phase_angles: list[float],
    in_state: int,
):
    if _USE_RUST:
        angles, in_state, gts, params, qubits = _extract_arrays(circuit)
        return _rust.v_circuit(angles, x, in_state, gts, params, qubits)
    return _py_V_circuit(circuit, x, c_phase_angles, in_state)


def calculate_expectation(
    in_state: int, out_state: int, V
) -> complex:
    if _USE_RUST:
        # V is already a numpy array from Rust or Python
        return _rust.calculate_expectation(V, in_state, out_state)
    return _py_calculate_expectation(in_state, out_state, V)


def raw_estimate(
    circuit: QuantumCircuit,
    out_state: int,
    epsilon: float,
    delta: float,
    p: float,
    data: Optional[Tuple[float, int, List[float]]] = None,
) -> float:
    """
    Monte‐Carlo estimate of the probability for `out_state` given `circuit`.

    If `data` is provided, it should be a tuple
        (extent, negative_mask, wrapped_angles)
    as returned by `get_circuit_data(circuit)`.  In that case neither
    Python nor Rust will re–walk the circuit to recompute them.

    Returns:
      float — the estimated probability.
    """
    if _USE_RUST:
        # use precomputed or compute once:
        if data is None:
            extent, negative_mask, wrapped_angles = _py_get_circuit_data(circuit)
        else:
            extent, negative_mask, wrapped_angles = data

        # Rust wants abs(θ) for sampling
        abs_angles = np.abs(np.array(wrapped_angles, dtype=np.float64))

        # unpack everything else for the Rust call
        _, in_state, gate_types, params, qubits = _extract_arrays(circuit)

        # call into Rust's raw_estimate
        return _rust.raw_estimate(
            abs_angles,
            negative_mask,
            extent,
            in_state,
            out_state,
            epsilon,
            delta,
            p,
            gate_types,
            params,
            qubits,
        )

    # pure‐Python fallback takes exactly the same `data` tuple
    return _py_raw_estimate(circuit, out_state, epsilon, delta, p, data)


def direct_calculation(circuit: QuantumCircuit, out_state: int) -> float:
    if _USE_RUST:
        angles, in_state, gts, params, qubits = _extract_arrays(circuit)
        return _rust.direct_calculation(angles, in_state, out_state, gts, params, qubits)
    return _py_direct_calculation(circuit, out_state)


def calculate_total_extent(c_phase_angles: list) -> float:
    return _py_calculate_total_extent(c_phase_angles)


def calculate_number_of_samples(
    epsilon: float,
    delta: float,
    total_extent: float,
    p: float,
) -> int:
    return _py_calculate_number_of_samples(epsilon, delta, total_extent, p)


def sample_bitstring(c_phase_angles: list[float]) -> int:
    return _py_sample_bitstring(c_phase_angles)
