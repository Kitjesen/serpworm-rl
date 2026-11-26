//! Fixed-size symmetric Jacobi eigendecomposition and SPD cone projection for
//! arbitrary small dimensions (`3x3`, `6x6`, `9x9`, `12x12`).

/// Symmetric cyclic Jacobi eigendecomposition returning `(eigenvalues, V)` such that
/// `A_sym == V * diag(evals) * V^T` (columns of `V` are orthonormal eigenvectors).
pub fn sym_eigen_n<const N: usize>(mat: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut a = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = 0.5 * (mat[i][j] + mat[j][i]);
        }
    }
    let mut v = [[0.0; N]; N];
    for i in 0..N {
        v[i][i] = 1.0;
    }

    let max_sweeps = 12 * N * N;
    for _ in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1.min(N.saturating_sub(1));
        for i in 0..N {
            for j in (i + 1)..N {
                let val = a[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 || p == q {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..N {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                let nrp = c * arp - s * arq;
                let nrq = s * arp + c * arq;
                a[r][p] = nrp;
                a[p][r] = nrp;
                a[r][q] = nrq;
                a[q][r] = nrq;
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    let mut evals = [0.0; N];
    for i in 0..N {
        evals[i] = a[i][i];
    }
    (evals, v)
}

/// Project an `N x N` symmetric matrix onto the positive semi-definite / SPD cone
/// by clamping all eigenvalues to `>= min_eig`.
pub fn project_spd_n<const N: usize>(mat: [[f64; N]; N], min_eig: f64) -> [[f64; N]; N] {
    let (evals, v) = sym_eigen_n(mat);
    let mut out = [[0.0; N]; N];
    for k in 0..N {
        let lam = evals[k].max(min_eig);
        if lam == 0.0 {
            continue;
        }
        for i in 0..N {
            let lvi = lam * v[i][k];
            for j in i..N {
                out[i][j] += lvi * v[j][k];
            }
        }
    }
    for i in 0..N {
        for j in 0..i {
            out[i][j] = out[j][i];
        }
    }
    out
}
