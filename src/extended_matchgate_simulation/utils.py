from qiskit.circuit import QuantumCircuit
from typing import *

def prep_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Prepares circuit to make sure that all X gates are at the beginning of
    the circuit.

    Args:
        circuit: a quantum circuit
    """
    prepared_circuit = QuantumCircuit(circuit.num_qubits)

    for instruction in circuit.data:
        if instruction.operation.name == "x":
            q = instruction.qubits[0]._index
            prepared_circuit.append(instruction.operation, [q])

    for instruction in circuit.data:
        if instruction.operation.name in ("xx_plus_yy", "cp"):
            q1 = instruction.qubits[0]._index
            q2 = instruction.qubits[1]._index
            prepared_circuit.append(instruction.operation, [q1, q2])

        if instruction.operation.name == "p":
            q = instruction.qubits[0]._index
            prepared_circuit.append(instruction.operation, [q])

    return prepared_circuit
