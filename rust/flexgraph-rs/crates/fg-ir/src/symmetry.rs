//! Finite symmetry group specifications ($\mathbb{Z}_2, \mathbb{Z}_2\times\mathbb{Z}_2, C_4, D_4$)
//! for exact Reynolds group-equivariant policy averaging and group-axiom verification.

use crate::schema::IrError;
use serde::{Deserialize, Serialize};

/// Canonical finite symmetry group classification for robotic morphologies.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryGroupKind {
    #[default]
    Trivial,
    /// Left-right bilateral reflection (`|G| = 2`, e.g., SerpWorm, FlexSpine-Centipede).
    Z2Bilateral,
    /// Left-right + head-tail reversal (`|G| = 4`, e.g., BristleWorm bidirectional crawler).
    Z2xZ2Bidirectional,
    /// 4-fold axial cyclic rotation (`|G| = 4`).
    C4Cyclic,
    /// 4-fold axial dihedral rotation + reflection (`|G| = 8`, e.g., TendonEndoscope-3D).
    D4Dihedral,
    Custom(String),
}

/// Signed permutation representation `(L_g, K_g, M_g)` of one group element `g in G`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymmetryElement {
    pub name: String,
    pub obs_perm: Vec<usize>,
    pub obs_sign: Vec<f64>,
    pub act_perm: Vec<usize>,
    pub act_sign: Vec<f64>,
    #[serde(default)]
    pub est_perm: Vec<usize>,
    #[serde(default)]
    pub est_sign: Vec<f64>,
}

impl SymmetryElement {
    /// Identity signed permutation of given dimensions.
    pub fn identity(obs_dim: usize, act_dim: usize, est_dim: usize) -> Self {
        Self {
            name: "e".to_string(),
            obs_perm: (0..obs_dim).collect(),
            obs_sign: vec![1.0; obs_dim],
            act_perm: (0..act_dim).collect(),
            act_sign: vec![1.0; act_dim],
            est_perm: (0..est_dim).collect(),
            est_sign: vec![1.0; est_dim],
        }
    }

    /// Apply `(L_g x)[i] = sign[i] * x[perm[i]]`.
    pub fn apply_signed_perm(perm: &[usize], sign: &[f64], x: &[f64]) -> Vec<f64> {
        perm.iter()
            .zip(sign.iter())
            .map(|(&p, &s)| s * x[p])
            .collect()
    }

    /// Apply inverse signed permutation `K_g^{-1} = K_g^T` (orthogonal signed permutation matrix).
    pub fn apply_signed_perm_inv(perm: &[usize], sign: &[f64], x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; perm.len()];
        for (i, (&p, &s)) in perm.iter().zip(sign.iter()).enumerate() {
            out[p] = s * x[i];
        }
        out
    }

    pub fn apply_obs(&self, obs: &[f64]) -> Vec<f64> {
        Self::apply_signed_perm(&self.obs_perm, &self.obs_sign, obs)
    }

    pub fn apply_act(&self, act: &[f64]) -> Vec<f64> {
        Self::apply_signed_perm(&self.act_perm, &self.act_sign, act)
    }

    pub fn apply_act_inv(&self, act: &[f64]) -> Vec<f64> {
        Self::apply_signed_perm_inv(&self.act_perm, &self.act_sign, act)
    }

    pub fn apply_est_inv(&self, est: &[f64]) -> Vec<f64> {
        Self::apply_signed_perm_inv(&self.est_perm, &self.est_sign, est)
    }

    /// Compose two group elements `self o rhs` on signed permutation representations.
    pub fn compose(&self, rhs: &Self, name: impl Into<String>) -> Self {
        let comp = |p1: &[usize], s1: &[f64], p2: &[usize], s2: &[f64]| -> (Vec<usize>, Vec<f64>) {
            let mut p = Vec::with_capacity(p1.len());
            let mut s = Vec::with_capacity(p1.len());
            for i in 0..p1.len() {
                let mid = p1[i];
                p.push(p2[mid]);
                s.push(s1[i] * s2[mid]);
            }
            (p, s)
        };
        let (obs_perm, obs_sign) =
            comp(&self.obs_perm, &self.obs_sign, &rhs.obs_perm, &rhs.obs_sign);
        let (act_perm, act_sign) =
            comp(&self.act_perm, &self.act_sign, &rhs.act_perm, &rhs.act_sign);
        let (est_perm, est_sign) =
            comp(&self.est_perm, &self.est_sign, &rhs.est_perm, &rhs.est_sign);
        Self {
            name: name.into(),
            obs_perm,
            obs_sign,
            act_perm,
            act_sign,
            est_perm,
            est_sign,
        }
    }

    /// Check representation equality with another element.
    pub fn same_action(&self, other: &Self) -> bool {
        let approx_eq = |a: &[f64], b: &[f64]| -> bool {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-12)
        };
        self.obs_perm == other.obs_perm
            && self.act_perm == other.act_perm
            && self.est_perm == other.est_perm
            && approx_eq(&self.obs_sign, &other.obs_sign)
            && approx_eq(&self.act_sign, &other.act_sign)
            && approx_eq(&self.est_sign, &other.est_sign)
    }
}

/// Complete finite symmetry group $\mathcal{G}$ attached to a `RigidFlexGraph`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SymmetryGroupSpec {
    #[serde(default)]
    pub kind: SymmetryGroupKind,
    #[serde(default)]
    pub elements: Vec<SymmetryElement>,
}

impl SymmetryGroupSpec {
    pub fn trivial() -> Self {
        Self {
            kind: SymmetryGroupKind::Trivial,
            elements: Vec::new(),
        }
    }

    /// Generate the complete finite subgroup from a set of generator elements via closure
    /// (up to `max_order`, e.g., 2 for `Z2`, 4 for `Z2xZ2`/`C4`, 8 for `D4`).
    pub fn from_generators(
        kind: SymmetryGroupKind,
        obs_dim: usize,
        act_dim: usize,
        est_dim: usize,
        generators: &[SymmetryElement],
        max_order: usize,
    ) -> Result<Self, IrError> {
        let mut elements = vec![SymmetryElement::identity(obs_dim, act_dim, est_dim)];
        let mut idx = 0;
        while idx < elements.len() {
            let cur = elements[idx].clone();
            for gen in generators {
                let prod = cur.compose(gen, format!("{}_{}", cur.name, gen.name));
                if !elements.iter().any(|e| e.same_action(&prod)) {
                    if elements.len() >= max_order {
                        return Err(IrError::InvalidSymmetry(format!(
                            "generated group exceeded max_order={max_order}"
                        )));
                    }
                    elements.push(prod);
                }
            }
            idx += 1;
        }
        let spec = Self { kind, elements };
        spec.validate()?;
        Ok(spec)
    }

    /// Validate that every element is a valid signed permutation (bijection with signs in `{-1, +1}`),
    /// consistent dimensions, and closure under composition if non-empty.
    pub fn validate(&self) -> Result<(), IrError> {
        if self.elements.is_empty() {
            return Ok(());
        }
        let obs_dim = self.elements[0].obs_perm.len();
        let act_dim = self.elements[0].act_perm.len();
        let est_dim = self.elements[0].est_perm.len();

        let check_sp = |perm: &[usize], sign: &[f64], dim: usize, label: &str| -> Result<(), IrError> {
            if perm.len() != dim || sign.len() != dim {
                return Err(IrError::InvalidSymmetry(format!(
                    "{label}: length mismatch (perm={}, sign={}, expected={dim})",
                    perm.len(),
                    sign.len()
                )));
            }
            let mut seen = vec![false; dim];
            for (&p, &s) in perm.iter().zip(sign.iter()) {
                if p >= dim || seen[p] {
                    return Err(IrError::InvalidSymmetry(format!(
                        "{label}: not a valid bijection"
                    )));
                }
                seen[p] = true;
                if (s.abs() - 1.0).abs() > 1e-12 {
                    return Err(IrError::InvalidSymmetry(format!(
                        "{label}: sign {s} is not +/-1"
                    )));
                }
            }
            Ok(())
        };

        for el in &self.elements {
            check_sp(&el.obs_perm, &el.obs_sign, obs_dim, &format!("{}.obs", el.name))?;
            check_sp(&el.act_perm, &el.act_sign, act_dim, &format!("{}.act", el.name))?;
            check_sp(&el.est_perm, &el.est_sign, est_dim, &format!("{}.est", el.name))?;
        }

        // Check group closure under composition
        for a in &self.elements {
            for b in &self.elements {
                let prod = a.compose(b, "prod");
                if !self.elements.iter().any(|e| e.same_action(&prod)) {
                    return Err(IrError::InvalidSymmetry(format!(
                        "group is not closed under composition {} * {}",
                        a.name, b.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Evaluate the exact Reynolds group-averaged equivariant action:
    /// `bar_mu(o) = (1/|G|) sum_{g in G} K_g^{-1} mu(L_g o)`.
    pub fn reynolds_average<F>(&self, obs: &[f64], mut policy_fn: F) -> Vec<f64>
    where
        F: FnMut(&[f64]) -> Vec<f64>,
    {
        if self.elements.is_empty() {
            return policy_fn(obs);
        }
        let n = self.elements.len() as f64;
        let mut acc: Option<Vec<f64>> = None;
        for g in &self.elements {
            let obs_g = g.apply_obs(obs);
            let act_g = policy_fn(&obs_g);
            let mapped = g.apply_act_inv(&act_g);
            match &mut acc {
                Some(sum) => {
                    for (a, &b) in sum.iter_mut().zip(mapped.iter()) {
                        *a += b;
                    }
                }
                None => acc = Some(mapped),
            }
        }
        let mut out = acc.unwrap_or_default();
        for v in &mut out {
            *v /= n;
        }
        out
    }
}
