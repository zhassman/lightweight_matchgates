import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate, PhaseGate, CPhaseGate, XGate
import numpy as np
from collections import Counter
from extended_matchgate_simulation.gate_by_gate import compute_set_S, get_start_index
from extended_matchgate_simulation.gate_by_gate import generate_samples
import pytest


@pytest.mark.expensive
def test_generate_samples():
    circuit = create_test_circuit_C()
    samples = generate_samples(circuit, .1, .01, 100, False, True)
    percent_composition = {k: v / len(samples) for k, v in Counter(samples).items()}
    assert 5 in percent_composition.keys()
    assert 6 in percent_composition.keys()
    assert 9 in percent_composition.keys()
    assert 10 in percent_composition.keys()
    assert len(percent_composition.keys()) == 4
    assert abs(percent_composition[5] - 0.8141811982061599) <= .1
    assert abs(percent_composition[6] - 0.08041538785096941) <= .1
    assert abs(percent_composition[9] - 0.08041538785096945) <= .1
    assert abs(percent_composition[10] - 0.024988026091901105) <= .1


def create_test_circuit_C():
    """Makes a 4-qubit circuit with 4 controlled phase gates
    all with small angles."""
    qubits = QuantumRegister(4)
    circuit = QuantumCircuit(qubits)
    a, b, c, d = qubits
    circuit.append(XGate(), [a])
    circuit.append(XGate(), [c])
    circuit.append(XXPlusYYGate(theta=1.7868, beta=-2.5196), [a,b])
    circuit.append(XXPlusYYGate(theta=1.7868, beta=-2.5196), [c,d])
    circuit.append(PhaseGate(1.4009), [a])
    circuit.append(PhaseGate(-1.5783), [b])
    circuit.append(PhaseGate(1.4009), [c])
    circuit.append(PhaseGate(-1.5783), [d])
    circuit.append(CPhaseGate(.1), [a,b])
    circuit.append(CPhaseGate(-.8), [c,d])
    circuit.append(CPhaseGate(-.05), [a,c])
    circuit.append(CPhaseGate(.7), [b,d])
    circuit.append(XXPlusYYGate(theta=1.7868, beta=-2.5196), [a,b])
    circuit.append(XXPlusYYGate(theta=1.7868, beta=-2.5196), [c,d])
    circuit.append(PhaseGate(1.4009), [a])
    circuit.append(PhaseGate(-1.5783), [b])
    circuit.append(PhaseGate(1.4009), [c])
    circuit.append(PhaseGate(-1.5783), [d])
    return circuit


def test_1_compute_set_S():
    x = 0b111
    assert compute_set_S(x, [0, 1]) == [0b100, 0b101, 0b110, 0b111]


def test_2_compute_set_S():
    """empty support"""
    x = 0b10101
    assert compute_set_S(x, []) == [x]


def test_3_compute_set_S():
    """single bit support"""
    x = 0b1001
    supp = [1]
    result = compute_set_S(x, supp)
    expected = [0b1001, 0b1011]
    assert result == expected


def test_4_compute_set_S():
    """two bit support"""
    x = 0b10110
    supp = [0, 3]
    result = compute_set_S(x, supp)
    assert result == [22, 23, 30, 31]


def test_5_compute_set_S():
    """full support"""
    x = 0
    supp = [0, 1]
    result = compute_set_S(x, supp)
    assert result == [0, 1, 2, 3]


def test_1_get_start_index():
    """tutorial circuit"""

    norb = 5
    rng = np.random.default_rng(1234)
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    a, b, c, d, e = qubits

    circuit.append(XGate(), [a])
    circuit.append(XGate(), [c])
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=rng)
        ),
        qubits,
    )
    circuit.append(CPhaseGate(rng.uniform()), [a, e])
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=rng)
        ),
        qubits,
    )
    circuit.append(CPhaseGate(rng.uniform()), [b, d])
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=rng)
        ),
        qubits,
    )
    circuit.append(CPhaseGate(rng.uniform()), [b, c])
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=rng)
        ),
        qubits,
    )
    circuit.append(CPhaseGate(rng.uniform()), [d, e])
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=rng)
        ),
        qubits,
    )

    assert get_start_index(circuit) == 2
