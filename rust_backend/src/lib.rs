use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array2, Axis};
use ndarray_linalg::Determinant;
use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;


/// Core Rust: build the V matrix
fn v_circuit_rust(
    cpa: &[f64],
    x: usize,
    _in_state: usize,
    gtypes: &[u8],
    pmat: &Array2<f64>,
    qmat: &Array2<usize>,
) -> Array2<Complex64> {
    let max_gate_qubit = qmat.iter().cloned().max().unwrap_or(0);
    let nqubits = max_gate_qubit + 1;

    // initialize full matrix as identity
    let mut v = Array2::<Complex64>::zeros((nqubits, nqubits));
    for i in 0..nqubits {
        v[(i, i)] = Complex64::new(1.0, 0.0);
    }

    // apply gates to all columns
    let mut cp_idx = 0;
    for (k, &gate_type) in gtypes.iter().enumerate() {
        match gate_type {
            1 => {
                // controlled-phase
                let theta = cpa[cp_idx];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
                let base = Complex64::from_polar(1.0, theta * 0.5);
                let factor = if ((x >> cp_idx) & 1) == 1 { -base } else { base };
                for elem in v.row_mut(q1).iter_mut() { *elem *= factor; }
                for elem in v.row_mut(q2).iter_mut() { *elem *= factor; }
                cp_idx += 1;
            }
            2 => {
                // xx+yy
                let theta = pmat[(k, 0)];
                let beta = pmat[(k, 1)];
                let q1 = qmat[(k, 0)];
                let q2 = qmat[(k, 1)];
                // pre-phase
                let ph = Complex64::from_polar(1.0, beta + PI / 2.0);
                for elem in v.row_mut(q1).iter_mut() { *elem *= ph; }
                // rotation
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                let cols = v.ncols();
                for j in 0..cols {
                    let a = v[(q1, j)];
                    let b = v[(q2, j)];
                    v[(q1, j)] = Complex64::new(c, 0.0) * a + Complex64::new(s, 0.0) * b;
                    v[(q2, j)] = Complex64::new(-s, 0.0) * a + Complex64::new(c, 0.0) * b;
                }
                // post-phase
                let iph = Complex64::from_polar(1.0, -beta - PI / 2.0);
                for elem in v.row_mut(q1).iter_mut() { *elem *= iph; }
            }
            3 => {
                // phase
                let theta = pmat[(k, 0)];
                let q1 = qmat[(k, 0)];
                let factor = Complex64::from_polar(1.0, theta);
                for elem in v.row_mut(q1).iter_mut() { *elem *= factor; }
            }
            _ => {} // ignore other gates
        }
    }

    v
}


fn calculate_expectation_rust<S>(
    v: &ndarray::ArrayBase<S, ndarray::Ix2>,
    in_state: usize,
    out_state: usize,
) -> Complex64
where
    S: ndarray::Data<Elem = Complex64>,
{
    let (nrows, ncols) = (v.shape()[0], v.shape()[1]);

    // select row indices where out_state bit is set
    let rows: Vec<usize> = (0..nrows).filter(|&i| (out_state >> i) & 1 == 1).collect();

    // decide columns
    let cols: Vec<usize> = if nrows == ncols {
        // for square v, select cols where in_state bit is set
        (0..ncols).filter(|&j| (in_state >> j) & 1 == 1).collect()
    } else {
        // for rectangular v, keep all columns
        (0..ncols).collect()
    };

    // build submatrix and take determinant
    let sub = v.select(Axis(0), &rows).select(Axis(1), &cols);
    sub.det().unwrap()
}


/// total_extent = ∏ (cos(theta/4) + sin(theta/4))²
#[pyfunction]
fn calculate_total_extent(_py: Python, angles: &PyArray1<f64>) -> PyResult<f64> {
    let slice = unsafe { angles.as_slice()? };
    Ok(slice.iter()
        .map(|&theta| ((theta / 4.0).cos() + (theta / 4.0).sin()).powi(2))
        .product())
}


/// number of samples
#[pyfunction]
fn calculate_number_of_samples(
    _py: Python,
    epsilon: f64,
    delta: f64,
    total_extent: f64,
    p: f64,
) -> PyResult<usize> {
    let num = (total_extent.sqrt() + p.sqrt()).powi(2);
    let logt = ((2.0 * std::f64::consts::E * std::f64::consts::E) / delta).ln();
    let den = ((p + epsilon).sqrt() - p.sqrt()).powi(2);
    Ok(((2.0 * num * logt) / den).ceil() as usize)
}


/// sample bitstring mask
#[pyfunction]
fn sample_bitstring(_py: Python, angles: &PyArray1<f64>) -> PyResult<usize> {
    let slice = unsafe { angles.as_slice()? };
    let mut rng = rand::thread_rng();
    let mut mask = 0;
    for (i, &theta) in slice.iter().enumerate() {
        let s = (theta / 4.0).sin();
        let c = (theta / 4.0).cos();
        if rng.gen::<f64>() < s / (s + c) {
            mask |= 1 << i;
        }
    }
    Ok(mask)
}


/// Py wrapper: V circuit → numpy array
#[pyfunction]
fn v_circuit(
    py: Python,
    angles: &PyArray1<f64>,
    x: usize,
    in_state: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let cpa = unsafe { angles.as_slice()? };
    let gts = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };
    let v = v_circuit_rust(&cpa, x, in_state, &gts, &pmat, &qmat);
    Ok(v.into_pyarray(py).to_owned())
}


/// Py wrapper: expectation → Python complex
#[pyfunction]
fn calculate_expectation(
    py: Python,
    arr: &PyArray2<Complex64>,
    in_state: usize,
    out_state: usize,
) -> PyResult<PyObject> {
    let v = unsafe { arr.as_array() };
    let (nrows, ncols) = (v.shape()[0], v.shape()[1]);

    // mirror Python logic: if square, select rows+cols; if rectangular, select only rows.
    let det = if nrows == ncols {
        calculate_expectation_rust(&v, in_state, out_state)
    } else {
        let rows: Vec<usize> = (0..nrows).filter(|&i| (out_state >> i) & 1 == 1).collect();
        let sub = v.select(Axis(0), &rows);
        sub.det().unwrap()
    };

    let builtins = py.import("builtins")?;
    let complex_fn = builtins.getattr("complex")?;
    Ok(complex_fn.call1((det.re, det.im))?.into())
}


/// Monte-Carlo bitstring probability analogous to simulator_py.py
#[pyfunction]
fn raw_estimate(
    py: Python,
    angles: &PyArray1<f64>,
    negative_mask: usize,
    extent: f64,
    in_state: usize,
    out_state: usize,
    epsilon: f64,
    delta: f64,
    p: f64,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
) -> PyResult<f64> {
    // 1) quick exit if input/output Hamming weights differ
    if in_state.count_ones() != out_state.count_ones() {
        return Ok(0.0);
    }

    // 2) build absolute‐angle array for sampling
    let raw: &[f64] = unsafe { angles.as_slice()? };
    let abs_angles: Vec<f64> = raw.iter().map(|&angle| angle.abs()).collect();
    let abs_py = PyArray1::from_vec(py, abs_angles);

    // 3) compute number of samples
    let s = calculate_number_of_samples(py, epsilon, delta, extent, p)?;

    // 4) extract Rust‐side gate data
    let gts = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };

    // 5) Monte‐Carlo loop
    let mut alpha = Complex64::new(0.0, 0.0);
    for _ in 0..s {
        let x_mask = sample_bitstring(py, &abs_py)?;
        let v_mat = v_circuit_rust(raw, x_mask, in_state, gts, &pmat, &qmat);
        let amp = calculate_expectation_rust(&v_mat, in_state, out_state);

        // j‐phase factor based on popcount mod 4
        let count = x_mask.count_ones() % 4;
        let factor_j = match count {
            0 => Complex64::new(1.0,  0.0),
            1 => Complex64::new(0.0,  1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        // re‐introduce sign of originally negative angles
        let sign = if (negative_mask & x_mask).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };

        alpha += factor_j * amp * sign;
    }

    // final estimator: extent * |alpha|^2 / s^2
    Ok((alpha.norm_sqr() / ((s * s) as f64)) * extent)
}


/// Py wrapper: direct calculation
#[pyfunction]
fn direct_calculation(
    py: Python,
    angles: &PyArray1<f64>,
    in_state: usize,
    out_state: usize,
    gate_types: &PyArray1<u8>,
    params: &PyArray2<f64>,
    qubits: &PyArray2<usize>,
) -> PyResult<f64> {
    let slice = unsafe { angles.as_slice()? };
    let gts = unsafe { gate_types.as_slice()? };
    let pmat = unsafe { params.as_array().to_owned() };
    let qmat = unsafe { qubits.as_array().to_owned() };

    if in_state.count_ones() != out_state.count_ones() { return Ok(0.0); }
    let extent = calculate_total_extent(py, angles)?;
    let k = slice.len();
    let mut total = Complex64::new(0.0, 0.0);

    for mask in 0..(1 << k) {
        let mut prob = 1.0;
        let mut coeff = Complex64::new(1.0, 0.0);
        for j in 0..k {
            let theta = slice[j];
            let s = (theta / 4.0).sin();
            let c = (theta / 4.0).cos();
            let d = s + c;
            if (mask >> j) & 1 == 0 { prob *= c / d; } else { prob *= s / d; coeff *= Complex64::new(0.0, 1.0); }
        }
        let v_mat = v_circuit_rust(slice, mask, in_state, gts, &pmat, &qmat);
        let amp = calculate_expectation_rust(&v_mat, in_state, out_state);
        total += coeff * amp * prob;
    }

    Ok((extent * total.norm_sqr()) as f64)
}


#[pymodule]
fn emsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_total_extent, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_number_of_samples, m)?)?;
    m.add_function(wrap_pyfunction!(sample_bitstring, m)?)?;
    m.add_function(wrap_pyfunction!(v_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_expectation, m)?)?;
    m.add_function(wrap_pyfunction!(raw_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(direct_calculation, m)?)?;
    Ok(())
}
