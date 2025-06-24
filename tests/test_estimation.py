import numpy as np
import math
from extended_matchgate_simulation.simulator import calculate_number_of_samples, get_circuit_data
from extended_matchgate_simulation.estimation import estimate, calculate_epsilon
from test_gate_by_gate import create_test_circuit_C


def test_estimate_1():
    circuit = create_test_circuit_C()
    epsilon, delta, out_state = .1, .01, 0b0101
    est = estimate(epsilon, delta, circuit, out_state)
    true_prob = 0.8141811982061599

    np.testing.assert_allclose(est, true_prob, rtol=0, atol=epsilon)


def test_estimate_2():
    circuit = create_test_circuit_C()
    epsilon, delta, out_state = .1, .01, 0b0110
    est = estimate(epsilon, delta, circuit, out_state)
    true_prob = 0.08041538785096941

    np.testing.assert_allclose(est, true_prob, rtol=0, atol=epsilon)


def test_estimate_3():
    circuit = create_test_circuit_C()
    data = get_circuit_data(circuit)
    epsilon, delta, out_state = .1, .01, 0b1001
    est = estimate(epsilon, delta, circuit, out_state, data)
    true_prob = 0.08041538785096945

    np.testing.assert_allclose(est, true_prob, rtol=0, atol=epsilon)


def test_estimate_4():
    circuit = create_test_circuit_C()
    data = get_circuit_data(circuit)
    epsilon, delta, out_state = .1, .01, 0b1010
    est = estimate(epsilon, delta, circuit, out_state, data)
    true_prob = 0.024988026091901105

    np.testing.assert_allclose(est, true_prob, rtol=0, atol=epsilon)


def test_calculate_epsilon_1():

    delta = .0001
    p = .001
    extent = .787878
    s = 128128
    
    epsilon = calculate_epsilon(p, delta, s, extent)
    
    s_cont = (
      2*(math.sqrt(extent)+math.sqrt(p))**2 * math.log(2*math.e**2/delta)
    ) / (math.sqrt(p+epsilon)-math.sqrt(p))**2
    
    eps_recov = calculate_epsilon(p, delta, s_cont, extent)
    
    assert np.allclose(eps_recov, epsilon)


def test_calculate_epsilon_2():

    delta = .01
    p = 1
    extent = .333
    s = 9999
    epsilon = calculate_epsilon(p, delta, s, extent)
    
    s_cont = (
      2*(math.sqrt(extent)+math.sqrt(p))**2 * math.log(2*math.e**2/delta)
    ) / (math.sqrt(p+epsilon)-math.sqrt(p))**2
    
    eps_recov = calculate_epsilon(p, delta, s_cont, extent)

    assert np.allclose(eps_recov, epsilon)
