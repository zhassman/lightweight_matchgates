import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CPhaseGate, XGate
import numpy as np
from qiskit.quantum_info import Statevector


from extended_matchgate_simulation.simulator import (
    raw_estimate,
    direct_calculation,
    get_input_state,
    get_controlled_phase_angles,
    V_circuit,
    calculate_expectation,
    get_circuit_data,
)

from extended_matchgate_simulation.utils import (
    prep_circuit,
)


def create_test_circuit_A():
    """
    Creates a 6-qubit test circuit.
    """
    norb = 3
    nelec = (1, 1)
    rng = np.random.default_rng(1234 * 2)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    a, b, c, d, e, f = qubits
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, norb, 1)]
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, interaction_pairs=(alpha_alpha_indices, alpha_beta_indices), seed=rng
    )
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    decomposed = circuit.decompose(reps=2)
    prepared = prep_circuit(decomposed)
    return prepared


def create_test_circuit_B():
    """
    Creates a 4-qubit test circuit.
    """
    norb = 2
    nelec = (1, 1)
    rng = np.random.default_rng(1234 * 2)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    a, b, c, d = qubits
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, norb, 1)]
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, interaction_pairs=(alpha_alpha_indices, alpha_beta_indices), seed=rng
    )
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    decomposed = circuit.decompose(reps=2)
    prepared = prep_circuit(decomposed)
    return prepared


def test_V_circuit_1():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b1110000
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[-0.8938588801288683-0.4142821062455402j, 0.1082578766649261+0.0489358186131308j, -0.0563215852158397+0.1100001622497158j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [-0.0005093312769444-0.0298703100138664j, -0.5589164615783597-0.5605834598268311j, -0.3263158580664249+0.5157364050375005j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.1686844616836020+0.0063005085004078j, 0.5886787060295666+0.1127008092665053j, 0.0328479003140629+0.7817817982545623j, 0.0000000000000000-0.0000000000000000j,
    0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.8957305868519502+0.4290366807223798j,
    -0.0988202499099040+0.0082241681520040j, -0.0613196039360226+0.0010343170040516j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0056431413272442+0.0875762618743543j,
    0.5873163232974462+0.7844582099138863j, -0.1478027849868599-0.1006863612872876j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, -0.0270449486196177+0.0718426505553726j,
    -0.0259424009622398+0.1708179334418285j, 0.1885754768788229+0.9636880819446939j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_V_circuit_2():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b0101010
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[-0.8524765962787092-0.3763875752669847j, -0.0182585492433002+0.3115952997729101j, 0.1813051179998094-0.0363248497277305j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.3308171312245078-0.1403628346872604j, 0.4932978003187941+0.5308739986721464j, 0.3000792918884769-0.5056093320352577j, -0.0000000000000000+0.0000000000000000j,
    -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j],
    [-0.0308598631973422+0.0390130125995366j, -0.5983784068870920-0.1390983624803379j, -0.0428506326095429-0.7863106286340444j, 0.0000000000000000-0.0000000000000000j,
    0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.8506048895556272+0.3616330007901453j,
    0.0088209224882781-0.3687552865380449j, -0.0636639288479471-0.0747096295260368j],
    [-0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, -0.3359509412748077+0.0826568828267725j,
    -0.5216976620378806-0.7547487487592015j, 0.1740393511648079+0.0905592882850449j],
    [0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, -0.1107796498666422-0.1171561716553169j,
    0.0356421018197651-0.1444203802279959j, -0.1785727445833430-0.9591592515652118j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_V_circuit_3():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b1101011
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[0.8524765962787092+0.3763875752669847j, 0.0182585492433002-0.3115952997729101j, -0.1813051179998094+0.0363248497277305j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [-0.3308171312245078+0.1403628346872604j, -0.4932978003187941-0.5308739986721464j, -0.3000792918884769+0.5056093320352577j, -0.0000000000000000+0.0000000000000000j,
    -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j],
    [0.0308598631973422-0.0390130125995366j, 0.5983784068870920+0.1390983624803379j, 0.0428506326095429+0.7863106286340444j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, -0.8957305868519502-0.4290366807223798j,
    0.0988202499099040-0.0082241681520040j, 0.0613196039360226-0.0010343170040516j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, -0.0056431413272442-0.0875762618743543j,
    -0.5873163232974462-0.7844582099138863j, 0.1478027849868599+0.1006863612872876j],
    [0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.0270449486196177-0.0718426505553726j,
    0.0259424009622398-0.1708179334418285j, -0.1885754768788229-0.9636880819446939j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_V_circuit_4():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b0010000
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[0.8957305868519502+0.4290366807223798j, -0.0988202499099040+0.0082241681520040j, -0.0613196039360226+0.0010343170040516j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.0056431413272442+0.0875762618743543j, 0.5873163232974462+0.7844582099138863j, -0.1478027849868599-0.1006863612872876j, -0.0000000000000000+0.0000000000000000j,
    -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j],
    [-0.0270449486196177+0.0718426505553726j, -0.0259424009622398+0.1708179334418285j, 0.1885754768788229+0.9636880819446939j, 0.0000000000000000-0.0000000000000000j,
    0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, -0.8938588801288683-0.4142821062455402j,
    0.1082578766649261+0.0489358186131308j, -0.0563215852158397+0.1100001622497158j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, -0.0005093312769444-0.0298703100138664j,
    -0.5589164615783597-0.5605834598268311j, -0.3263158580664249+0.5157364050375005j],
    [0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.1686844616836020+0.0063005085004078j,
    0.5886787060295666+0.1127008092665053j, 0.0328479003140629+0.7817817982545623j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_V_circuit_5():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b0000000
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[0.8957305868519502+0.4290366807223798j, -0.0988202499099040+0.0082241681520040j, -0.0613196039360226+0.0010343170040516j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.0056431413272442+0.0875762618743543j, 0.5873163232974462+0.7844582099138863j, -0.1478027849868599-0.1006863612872876j, -0.0000000000000000+0.0000000000000000j,
    -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j],
    [-0.0270449486196177+0.0718426505553726j, -0.0259424009622398+0.1708179334418285j, 0.1885754768788229+0.9636880819446939j, 0.0000000000000000-0.0000000000000000j,
    0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.8957305868519502+0.4290366807223798j,
    -0.0988202499099040+0.0082241681520040j, -0.0613196039360226+0.0010343170040516j],
    [-0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, 0.0056431413272442+0.0875762618743543j,
    0.5873163232974462+0.7844582099138863j, -0.1478027849868599-0.1006863612872876j],
    [0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, 0.0000000000000000-0.0000000000000000j, -0.0270449486196177+0.0718426505553726j,
    -0.0259424009622398+0.1708179334418285j, 0.1885754768788229+0.9636880819446939j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_V_circuit_6():
    """
    Confirms that V_circuit constructs the correct matrix given
    an arbitray controlled-phase gate decomposition using test
    circuit A.
    """
    decomp_config = 0b1111111
    circuit = create_test_circuit_A()
    V = V_circuit(
        circuit,
        decomp_config,
        get_controlled_phase_angles(circuit),
        get_input_state(circuit),
    )
    
    true_matrix = np.array(
    [[0.8524765962787092+0.3763875752669847j, 0.0182585492433002-0.3115952997729101j, -0.1813051179998094+0.0363248497277305j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [-0.3308171312245078+0.1403628346872604j, -0.4932978003187941-0.5308739986721464j, -0.3000792918884769+0.5056093320352577j, -0.0000000000000000+0.0000000000000000j,
    -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j],
    [0.0308598631973422-0.0390130125995366j, 0.5983784068870920+0.1390983624803379j, 0.0428506326095429+0.7863106286340444j, 0.0000000000000000+0.0000000000000000j,
    0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.8524765962787092+0.3763875752669847j,
    0.0182585492433002-0.3115952997729101j, -0.1813051179998094+0.0363248497277305j],
    [-0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, -0.0000000000000000+0.0000000000000000j, -0.3308171312245078+0.1403628346872604j,
    -0.4932978003187941-0.5308739986721464j, -0.3000792918884769+0.5056093320352577j],
    [0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0000000000000000+0.0000000000000000j, 0.0308598631973422-0.0390130125995366j,
    0.5983784068870920+0.1390983624803379j, 0.0428506326095429+0.7863106286340444j]], dtype=complex)

    assert np.allclose(V, true_matrix)


def test_calculate_expectation_1():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b001001
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = -0.589006592722482 - 0.6284407815057226j

    np.testing.assert_allclose(expectation, true_expectation)


def test_calculate_expectation_2():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b001010
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = 0.33215450247569694 + 0.0002410783806340684j

    np.testing.assert_allclose(expectation, true_expectation)


def test_calculate_expectation_3():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b010010
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = -0.0995363722447264 + 0.0744993392858985j

    np.testing.assert_allclose(expectation, true_expectation)


def test_calculate_expectation_4():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b100010
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = -0.053092178321585526 - 0.023207922931313837j

    np.testing.assert_allclose(expectation, true_expectation)


def test_calculate_expectation_5():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b100001
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = 0.05034093147835489 + 0.14156897824799972j

    np.testing.assert_allclose(expectation, true_expectation)


def test_calculate_expectation_6():
    """
    Tests that the expectation function works for an arbitrary
    output state using test circuit A and a fixed decomposition.
    """
    circuit = create_test_circuit_A()
    in_state = get_input_state(circuit)
    out_state = 0b001100
    decomp_config = 0b0101010
    V = V_circuit(
        circuit, decomp_config, get_controlled_phase_angles(circuit), in_state
    )
    expectation = calculate_expectation(in_state, out_state, V)
    true_expectation = -0.04035794334291115 + 0.022024714341432886j

    np.testing.assert_allclose(expectation, true_expectation)


def test_raw_estimate_1():
    """
    Checks the raw_estimate function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b0101
    epsilon = 0.1
    delta = 0.01
    p = 1
    prob = raw_estimate(circuit, out_state, epsilon=epsilon, delta=delta, p=p)
    true_probability = 0.16833469057684222
    # real epsilon/error is always smaller than the one we provide
    np.testing.assert_allclose(prob, true_probability, rtol=0, atol=0.5 * epsilon)


def test_raw_estimate_2():
    """
    Checks the raw_estimate function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b0110
    epsilon = 0.1
    delta = 0.01
    p = 1
    prob = raw_estimate(circuit, out_state, epsilon=epsilon, delta=delta, p=p)
    true_probability = 0.07170271707314622
    # real epsilon/error is always smaller than the one we provide
    np.testing.assert_allclose(prob, true_probability, rtol=0, atol=0.5 * epsilon)


def test_raw_estimate_3():
    """
    Checks the raw_estimate function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b1001
    epsilon = 0.1
    delta = 0.01
    p = 1
    prob = raw_estimate(circuit, out_state, epsilon=epsilon, delta=delta, p=p)
    true_probability = 0.07170271707314622
    # real epsilon/error is always smaller than the one we provide
    np.testing.assert_allclose(prob, true_probability, rtol=0, atol=0.5 * epsilon)


def test_raw_estimate_4():
    """
    Checks the raw_estimate function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b1010
    epsilon = 0.1
    delta = 0.01
    p = 1
    prob = raw_estimate(circuit, out_state, epsilon=epsilon, delta=delta, p=p)
    true_probability = 0.6882598752768656
    # real epsilon/error is always smaller than the one we provide
    np.testing.assert_allclose(prob, true_probability, rtol=0, atol=0.5 * epsilon)


def test_raw_estimate_precomputed():
    """
    Checks the raw_estimate function for an arbitrary
    output state using test circuit B. For this test, circuit
    data is pre-computed.
    
    """
    circuit = create_test_circuit_B()
    out_state = 0b1010
    epsilon = 0.1
    delta = 0.01
    p = 1
    data = get_circuit_data(circuit)
    prob = raw_estimate(circuit, out_state, epsilon=epsilon, delta=delta, p=p, data=data)
    true_probability = 0.6882598752768656
    # real epsilon/error is always smaller than the one we provide
    np.testing.assert_allclose(prob, true_probability, rtol=0, atol=0.5 * epsilon)


def test_direct_calculation_1():
    """
    Checks the direct_calculation function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b0101
    prob = direct_calculation(circuit, out_state)
    true_probability = 0.16833469057684222
    np.testing.assert_allclose(prob, true_probability)


def test_direct_calculation_2():
    """
    Checks the direct_calculation function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b0110
    prob = direct_calculation(circuit, out_state)
    true_probability = 0.07170271707314622
    np.testing.assert_allclose(prob, true_probability)


def test_direct_calculation_3():
    """
    Checks the direct_calculation function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b1001
    prob = direct_calculation(circuit, out_state)
    true_probability = 0.07170271707314622
    np.testing.assert_allclose(prob, true_probability)


def test_direct_calculation_4():
    """
    Checks the direct_calculation function for an arbitrary
    output state using test circuit B.
    """
    circuit = create_test_circuit_B()
    out_state = 0b1010
    prob = direct_calculation(circuit, out_state)
    true_probability = 0.6882598752768656
    np.testing.assert_allclose(prob, true_probability)
