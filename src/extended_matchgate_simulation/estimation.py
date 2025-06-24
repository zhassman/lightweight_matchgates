import numpy as np
from typing import *
import math
from qiskit.circuit import QuantumCircuit
from extended_matchgate_simulation.simulator import (
    raw_estimate,
    calculate_number_of_samples,
    get_circuit_data,
)


def estimate(epsTot: float,
             deltaTot: float, 
             circuit: QuantumCircuit, 
             out_state: int, 
             data: Optional[Tuple[float, int, List[float]]] = None,
) -> float:
    if not data:
        data = get_circuit_data(circuit) 
        
    extent, _, _ = data
    
    pStar = 1
    pHat = 1
    exitCondition = False
    eStar = 1
    k = 1
    s = calculate_number_of_samples(eStar, 6*deltaTot/np.power(np.pi*k,2), extent, pHat)
    
    while not exitCondition:
        
        eStar = calculate_epsilon(pStar, 6*deltaTot/np.power(np.pi*k,2), s, extent)
    
        if eStar < epsTot:
            exitCondition = True

        pHat = raw_estimate(circuit, out_state, eStar, 6*deltaTot/np.power(np.pi*k,2), pStar, data)
        print("k:", k, "pHat:", pHat, "eStar:", eStar,"epsTot:", epsTot)
        pStar = max(0, min(1, pStar, pHat + eStar))

        k += 1
        s *= 2

    return pHat


def calculate_epsilon(p, delta, s, extent):
    sqrt_p = math.sqrt(p)
    sqrt_extent = math.sqrt(extent)
    last_term = math.sqrt(2 * math.log(2 * (math.e ** 2) / delta) / s)
    return -p + (sqrt_p + (sqrt_extent + sqrt_p) * last_term) ** 2
