from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.circuit import CircuitInstruction
from typing import *
from extended_matchgate_simulation.simulator import (
    get_input_state,
    raw_estimate,
    get_circuit_data
)
from extended_matchgate_simulation.estimation import estimate
from random import choices
from tqdm import tqdm


def generate_samples(
    circuit: QuantumCircuit,
    epsilon: float,
    delta: float,
    quantity: int,
    use_estimate: bool,
    hide_inner_loop: bool = True,
) -> int:
    """
    Generates a sample from the given circuit using the Gate-by-gate
    sampling algorithm.

    Args:
        circuit: a quantum circuit
        epsilon: the error tolerance
        delta: probability that the output of raw_estimate is incorrect
        p: the probability upper bound for the output state
        quantity: the number of samples
        use_estimate: toggles estimate vs raw_estimate (True for estimate)
        hide_inner_loop: toggles printing the progress bar for gates in a single sample
    """
    assert x_gates_first(circuit), (
        "All x gates must appear at the beginning of the circuit."
    )
    start_index = get_start_index(circuit)
    m = len(circuit.data)

    samples = []

    data = get_circuit_data(circuit)

    for q in tqdm(range(quantity), desc="Samples"):
        x = get_input_state(circuit)
        for t in tqdm(
            range(start_index, m), desc=f"Gates in sample {q}", disable=hide_inner_loop
        ):
            if circuit.data[t].operation.name in ("cp", "p"):
                continue

            supp_U_t = []
            for q in circuit.data[t].qubits:
                supp_U_t.append(q._index)

            prefix_circuit = make_prefix_circuit(circuit, t)
            S = compute_set_S(x, supp_U_t)
            if use_estimate:                
                probabilities = [estimate(epsilon, delta, circuit, s, data) for s in S]
            else:
                probabilities = [raw_estimate(prefix_circuit, s, epsilon, delta, 1, data) for s in S]

            total = sum(probabilities)
            normalized = [prob / total for prob in probabilities]
            x = choices(S, weights=normalized, k=1)[0]

        samples.append(x)

    return samples


def x_gates_first(
    circuit: QuantumCircuit,
) -> bool:
    """
    Confirms that all X gates appear first in the circuit.

    Args:
        circuit: a quantum circuit
    """
    last_x_index = -1
    for i, instruction in enumerate(circuit.data):
        if instruction.name == "x":
            if i - last_x_index > 1:
                return False
            last_x_index = i

    return True


def get_start_index(
    circuit: QuantumCircuit,
) -> int:
    """
    Returns the index of the first gate that is not an x gate.

    Args:
        circuit: a quantum circuit
    """
    for i, instruction in enumerate(circuit.data):
        if instruction.operation.name != "x":
            return i
    raise ValueError("Circuit is empty or all gates are x gates")


def make_prefix_circuit(
    circuit: QuantumCircuit,
    t: int,
) -> QuantumCircuit:
    """
    Creates a circuit that contains only gates up to index t
    (inclusive).

    Args:
        circuit: a quantum circuit
        t: the t-th gate with respect to the original circuit (including x gates)
    """
    assert t < len(circuit.data), "Last index must be within the circuit"

    prefix_circuit = QuantumCircuit(circuit.num_qubits)
    relevant_gates = circuit.data[: t + 1]
    for instruction in relevant_gates:
        prefix_circuit.append(
            instruction.operation, [q._index for q in instruction.qubits]
        )

    return prefix_circuit


def compute_set_S(x: int, supp_U_t: List[int]) -> List[int]:
    r"""
    Return all possible bitstrings for which bits of x \ supp_U_t are varied.

    Args:
        x: state
        supp_U_t: support of unitary operator U
    """
    k = len(supp_U_t)
    mask = sum(1 << q for q in supp_U_t)  # mask of positions we will vary
    base = x & ~mask  # start by setting masked positions to 0
    S = []
    for b in range(1 << k):  # b runs from 0 to 2^k - 1
        delta = 0
        for j in range(k):
            bit_j = (b >> j) & 1  # Extract the j-th bit of b (0 or 1):
            if bit_j:  # If bit is 1, set position supp_U_t[j] of delta to be 1
                delta |= 1 << supp_U_t[j]
        y = base | delta
        S.append(y)
    return S
