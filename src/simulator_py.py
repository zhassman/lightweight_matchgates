import cmath
import math
import warnings
from typing import *

import ffsim
import numpy as np
import scipy
import scipy.linalg.blas
from qiskit.circuit import QuantumCircuit


def get_circuit_data(
    circuit: QuantumCircuit
) -> Tuple[float, int, list[float]]:
    """
    Obtains relevant quantities (the extent, location of negative
    controlled phase angles, and the list of all controlled phase
    angles) from the circuit so they do not need to be recomputed.
    
    Args:
        circuit: a quantum circuit
    """
    c_phase_angles = get_controlled_phase_angles(circuit)
    
    negative_angles = 0

    for i in range(len(c_phase_angles)):
        while c_phase_angles[i] > math.pi:
            c_phase_angles[i] -= 2 * math.pi
        while c_phase_angles[i] < -math.pi:
            c_phase_angles[i] += 2 * math.pi
        if c_phase_angles[i] < 0:
            negative_angles |= 1 << i

    extent = calculate_total_extent(np.abs(c_phase_angles))

    return extent, negative_angles, c_phase_angles


def raw_estimate(
    circuit: QuantumCircuit,
    out_state: int,
    epsilon: float,
    delta: float,
    p: float,
    data: Optional[Tuple[float, int, list[float]]] = None
) -> float:
    """
    Returns the probability of a bitsring given a ffsim circuit with passive gates (input
    state is determined by x gates at the beginning of the circuit).

    Args:
        circuit: a quantum circuit
        out_state: the output state
        epsilon: the error tolerance (called epsilon in FFO)
        delta: probability that the output of this function is incorrect (called delta in FFO)
        p: the probability upper bound for the output state
        data: a tuple of three items...
            -the pre-computed extent of the circuit
            -an integer used as a bitstring to track the location of negative angles
            -the angles of the controlled phase gates
    """
    if not data:
        extent, negative_angles, c_phase_angles = get_circuit_data(circuit)
    else:
        extent, negative_angles, c_phase_angles = data


    in_state = get_input_state(circuit)
    if in_state.bit_count() != out_state.bit_count():
        return 0.0

    s = calculate_number_of_samples(epsilon, delta, extent, p)

    alpha = 0
    for _ in range(s):
        x = sample_bitstring(np.abs(c_phase_angles))
        V = V_circuit(circuit, x, c_phase_angles, in_state)
        amplitude = calculate_expectation(in_state, out_state, V)
        sign = 1
        if (negative_angles & x).bit_count() % 2 == 1:
            sign = -1

        alpha += ((1j) ** x.bit_count()) * amplitude * sign

    return (1 / s**2) * abs(alpha) ** 2 * extent


def sample_bitstring(c_phase_angles: list[float]) -> int:
    """
    For each controlled phase gate in the original circuit, randomly sample either
    0 or 1 to obtain a bitstring x of length k distributed according to P (eq. 80, FFO)

    Args:
        c_phase_angles: The list of angles associated with each controlled phase
        gate in the LUCJ circuit
    """
    angles = np.array(c_phase_angles)
    s, c = np.sin(angles / 4.0), np.cos(angles / 4.0)
    p1 = s / (s + c)
    u = np.random.random(len(angles))
    bits = u < p1

    mask = 0
    for i, bit in enumerate(bits):
        if bit:
            mask |= 1 << i
    return mask


def calculate_number_of_samples(
    epsilon: float,
    delta: float,
    total_extent: float,
    p: float,
) -> int:
    """
    Computes a lower bound on s, the number of samples needed (eq. 88 FFO)
    from probability distribution P (eq. 80 FFO)

    Args:
        epsilon: additive error
        delta: failure probability
        total_extent: as defined in eq. 66, FFO
        p: the Born rule probability upper bound
    """
    numerator = math.pow((math.sqrt(total_extent) + math.sqrt(p)), 2)
    log_term = math.log((2 * math.pow(math.e, 2)) / delta)
    denominator = math.pow(math.sqrt(p + epsilon) - math.sqrt(p), 2)

    return math.ceil((2 * numerator * log_term) / denominator)


def direct_calculation(
    circuit: QuantumCircuit,
    out_state: int,
) -> float:
    """
    A direct caculation of final statevector probabilities
    (equation 7 in notes)

    Args:
        circuit: a circuit
        out_state: the output state
    """
    c_phase_angles = get_controlled_phase_angles(circuit)
    in_state = get_input_state(circuit)
    extent = calculate_total_extent(c_phase_angles)
    num_c_phase = len(c_phase_angles)

    if in_state.bit_count() != out_state.bit_count():
        return 0.0

    total = 0

    for mask in range(2**num_c_phase):
        prob = 1
        coeff = 1
        for j in range(num_c_phase):
            bit = (mask >> j) & 1
            theta = c_phase_angles[j]
            if bit == 0:
                prob *= math.cos(theta / 4) / (
                    math.sin(theta / 4) + math.cos(theta / 4)
                )
                coeff *= 1
            else:
                prob *= math.sin(theta / 4) / (
                    math.sin(theta / 4) + math.cos(theta / 4)
                )
                coeff *= 1j

        V = V_circuit(circuit, mask, c_phase_angles, in_state)
        amplitude = calculate_expectation(in_state, out_state, V)
        total += coeff * amplitude * prob

    return extent * abs(total) ** 2


def V_circuit(
    circuit: QuantumCircuit,
    x: int,
    c_phase_angles: list[float],
    in_state=None,
) -> np.ndarray:
    """
    Create the full V matrix for the entire circuit

    Args:
        circuit: a quantum circuit
        x: sampled bitstring for controlled-phase decomposition
        c_phase_angles: list of all controlled-phase angles
    """
    nqubits = circuit.num_qubits
    x_ind = 0

    V = np.eye(nqubits, dtype=complex)

    for instruction in circuit.data:
        if instruction.name == "cp":
            theta = c_phase_angles[x_ind]
            q1 = instruction.qubits[0]._index
            q2 = instruction.qubits[1]._index
            if (x >> x_ind) & 1:
                apply_V_D_1(nqubits, theta, (q1, q2), V)
            else:
                apply_V_D_0(nqubits, theta, (q1, q2), V)
            x_ind += 1

        elif instruction.name == "xx_plus_yy":
            theta, beta = instruction.operation.params
            q1 = instruction.qubits[0]._index
            q2 = instruction.qubits[1]._index
            apply_V_xx_plus_yy(nqubits, theta, beta, (q1, q2), V)

        elif instruction.name == "p":
            theta = instruction.operation.params[0]
            q = instruction.qubits[0]._index
            apply_V_phase(nqubits, theta, q, V)

        elif instruction.name in ("x", "global_phase"):
            continue

        else:
            raise Exception(f"Unsupported gate: {instruction.name}")

    return V


def apply_V_D_0(nqubits: int, theta: float, q1q2: tuple[int, int], M: np.ndarray):
    """
    Applies the V matrix of the 1st gate in the controlled-phase decomposition in place to the matrix M

     Args:
         nqubits: the number of qubits
         theta: the angle of the gate
         q1q2: the two qubits that the gate acts on
         M: the matrix the V matrix is (left) multipled on to
    """
    scipy.linalg.blas.zscal(cmath.rect(1, 0.5 * theta), M[q1q2[0]])
    scipy.linalg.blas.zscal(cmath.rect(1, 0.5 * theta), M[q1q2[1]])


def apply_V_D_1(nqubits: int, theta: float, q1q2: tuple[int, int], M: np.ndarray):
    """
    Applies the V matrix of the 2nd gate in the controlled-phase decomposition in place to the matrix M

     Args:
         nqubits: the number of qubits
         theta: the angle of the gate
         q1q2: the two qubits that the gate acts on
         M: the matrix the V matrix is (left) multipled on to
    """
    scipy.linalg.blas.zscal(-cmath.rect(1, 0.5 * theta), M[q1q2[0]])
    scipy.linalg.blas.zscal(-cmath.rect(1, 0.5 * theta), M[q1q2[1]])


def apply_V_phase(nqubits: int, theta: float, q: int, M: np.ndarray):
    """
    Applies the V matrix of the single qubit phase gate

     Args:
         nqubits: the number of qubits
         theta: the angle of the gate
         q: the phase
         M: the matrix the V matrix is (left) multipled on to
    """
    scipy.linalg.blas.zscal(cmath.rect(1, theta), M[q])


def apply_V_xx_plus_yy(
    nqubits: int, theta: float, beta: float, q1q2: tuple[int, int], M: np.ndarray
):
    """
    Applies the V matrix of the xx_plus_yy gate

     Args:
         nqubits: the number of qubits
         theta: the first angle from the xx_plus_yy
         beta: the second angle from the xx_plus_yy
         q1q2: the two qubits that the gate acts on
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    scipy.linalg.blas.zscal(cmath.rect(1, beta + 0.5 * math.pi), M[q1q2[0]])
    scipy.linalg.blas.zdrot(
        M[q1q2[0]],
        M[q1q2[1]],
        c,
        s,
        n=len(M[q1q2[1]]),
        offx=0,
        incx=1,
        offy=0,
        incy=1,
        overwrite_x=True,
        overwrite_y=True,
    )
    scipy.linalg.blas.zscal(cmath.rect(1, -beta - 0.5 * math.pi), M[q1q2[0]])


def get_input_state(circuit: QuantumCircuit) -> int:
    """
    Gets the initial state of a circuit as described by only the first
    consecutive sequence of X gates. Raises a warning if any X gates
    appear later in the circuit.

    Args:
        circuit: a quantum circuit
    """
    in_state = 0
    seen_non_x_gate = False

    for instruction in circuit.data:
        if instruction.name == "x":
            q = instruction.qubits[0]._index
            if seen_non_x_gate:
                warnings.warn(
                    "All X gates should appear consecutively at the beginning of the circuit."
                )
            in_state |= 1 << q
        else:
            seen_non_x_gate = True

    return in_state


def get_controlled_phase_angles(
    circuit: QuantumCircuit,
) -> list[float]:
    """
    Gets the angles of the controlled phase gates in the order they appear.

    Args:
        circuit: any quantum circuit
    """

    c_phase_angles = []

    for instruction in circuit.data:
        if instruction.name == "cp":
            c_phase_angles.append(instruction.params[0])

    return c_phase_angles


def calculate_expectation(in_state: int, out_state: int, V: np.ndarray) -> complex:
    """
    Computes the expectation of an output state after applying a matchgate circuit
    to an input state. Equation 18 of T&D

    Args:
        in_state: the input state
        out_state: the output state
        V: An instance of the orbital rotation matrix describing the matchgate circuit
    """

    if V.shape[0] == V.shape[1]:
        n = V.shape[0]

        rows = [i for i in range(n) if ((out_state >> i) & 1)]

        cols = [i for i in range(n) if ((in_state >> i) & 1)]

        V_tilda = V[np.ix_(rows, cols)]

        return np.linalg.det(V_tilda)
    else:
        rows = [i for i in range(V.shape[0]) if ((out_state >> i) & 1)]
        V_tilda = V[rows]
        return np.linalg.det(V_tilda)


def calculate_total_extent(c_phase_angles: list[float]) -> float:
    """
    Computes a quantity known as total extent (equation 66 of FFO)

    Args:
        c_phase_angles: The list of angles associated with each controlled phase
        gate in the LUCJ circuit
    """
    prod = 1
    for theta in c_phase_angles:
        prod *= (math.cos(theta / 4) + math.sin(theta / 4)) ** 2

    return prod
