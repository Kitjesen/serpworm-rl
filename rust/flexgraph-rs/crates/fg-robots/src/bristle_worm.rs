//! Archetype 2: `BristleWorm` — wheelless anisotropic ventral-scale / bristle limbless crawler
//! with bidirectional Klein four-group symmetry ($\mathbb{Z}_2 \times \mathbb{Z}_2$, order 4).

use crate::serpworm::{build_serpworm, ParametersJson, SerpWormConfig, SerpWormPreset};
use fg_ir::{
    ContactMode, ContactSiteSpec, RigidFlexGraph, SymmetryElement, SymmetryGroupKind,
    SymmetryGroupSpec,
};
use fg_math::Vec3;
use serde::{Deserialize, Serialize};

/// Configuration for the wheelless anisotropic-scale `BristleWorm` crawler.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BristleWormConfig {
    pub n_segments: usize,
    pub elements_per_strip: usize,
    pub mu_fwd: f64,
    pub mu_rev: f64,
    pub mu_lat: f64,
    pub pivot_mu: f64,
    pub static_mu_ratio: f64,
    pub scale_radius_m: f64,
    pub patch_radius_m: f64,
    pub preferred_dt: f64,
    pub material_profile_id: usize,
}

impl Default for BristleWormConfig {
    fn default() -> Self {
        Self {
            n_segments: 5,
            elements_per_strip: 10,
            mu_fwd: 0.14,
            mu_rev: 0.88,
            mu_lat: 0.92,
            pivot_mu: 0.50,
            static_mu_ratio: 1.15,
            scale_radius_m: 0.010,
            patch_radius_m: 0.014,
            preferred_dt: 0.001,
            material_profile_id: 0,
        }
    }
}

/// Build the Klein four-group $\mathbb{Z}_2 \times \mathbb{Z}_2$ (`|G| = 4`: identity,
/// left-right mirror, front-back reversal, and their composition) for a reversible crawler.
pub fn make_z2xz2_bidirectional_symmetry(n_segments: usize) -> SymmetryGroupSpec {
    let n_t = 2 * n_segments;
    let n_y = n_segments.saturating_sub(1);
    let n_a = n_t + n_y;

    // Generator 1: left-right reflection (swap L/R tendon pairs, negate yaw)
    let mut p_lr = Vec::with_capacity(n_a);
    let mut s_lr = Vec::with_capacity(n_a);
    for seg in 0..n_segments {
        p_lr.push(2 * seg + 1);
        p_lr.push(2 * seg);
        s_lr.push(1.0);
        s_lr.push(1.0);
    }
    for j in 0..n_y {
        p_lr.push(n_t + j);
        s_lr.push(-1.0);
    }

    // Generator 2: 180-deg Z-rotation (head-tail reversal, det = +1):
    // reverses segment order, swaps L/R tendons, and reverses connector orientation (a,b)->(b,a),
    // which maps R_a^T R_b -> (R_a^T R_b)^T = R_z(-yaw), so yaw sign is -1.0.
    // (Composition lr * fb then has yaw sign (-1)*(-1) = +1.0, matching YZ mirror reflection.)
    let mut p_fb = Vec::with_capacity(n_a);
    let mut s_fb = Vec::with_capacity(n_a);
    for seg in 0..n_segments {
        let rev = n_segments - 1 - seg;
        p_fb.push(2 * rev + 1);
        p_fb.push(2 * rev);
        s_fb.push(1.0);
        s_fb.push(1.0);
    }
    for j in 0..n_y {
        p_fb.push(n_t + (n_y - 1 - j));
        s_fb.push(-1.0);
    }

    let g_lr = SymmetryElement {
        name: "lr".to_string(),
        obs_perm: p_lr.clone(),
        obs_sign: s_lr.clone(),
        act_perm: p_lr,
        act_sign: s_lr,
        est_perm: (0..7).collect(),
        est_sign: vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
    };
    let g_fb = SymmetryElement {
        name: "fb".to_string(),
        obs_perm: p_fb.clone(),
        obs_sign: s_fb.clone(),
        act_perm: p_fb,
        act_sign: s_fb,
        est_perm: (0..7).collect(),
        est_sign: vec![-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    };

    SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::Z2xZ2Bidirectional,
        n_a,
        n_a,
        7,
        &[g_lr, g_fb],
        4,
    )
    .expect("Z2xZ2 symmetry closure must succeed")
}

/// Build the wheelless `BristleWorm` crawler graph (`SchurRoute::BlockDiagonalPairs`, `Z2xZ2`).
pub fn build_bristle_worm(
    cfg: &BristleWormConfig,
    params_opt: Option<&ParametersJson>,
) -> RigidFlexGraph {
    let mut sw_cfg = SerpWormConfig::from_preset(
        SerpWormPreset::Ds3218Default,
        cfg.n_segments,
        cfg.elements_per_strip,
    );
    sw_cfg.preferred_dt = cfg.preferred_dt;
    sw_cfg.ground_contact = true;

    let mut graph = build_serpworm(&sw_cfg, params_opt);
    graph.name = format!("bristle_worm_N{}", cfg.n_segments);
    graph.symmetry = make_z2xz2_bidirectional_symmetry(cfg.n_segments);

    // Replace passive ratchet wheels with wheelless anisotropic ventral scales
    // preserving exact 1 mm ground clearance at rest (drop = 0.018 - scale_radius_m).
    let z_shift = crate::serpworm::WHEEL_RADIUS_M - cfg.scale_radius_m;
    graph.contact_sites = graph
        .contact_sites
        .into_iter()
        .enumerate()
        .map(|(idx, site)| ContactSiteSpec {
            id: idx,
            name: format!("ventral_scale_{idx}"),
            attachment: site.attachment,
            local_pos: Vec3::new(
                site.local_pos.x(),
                site.local_pos.y(),
                site.local_pos.z() - z_shift,
            ),
            local_frame: site.local_frame,
            radius: cfg.scale_radius_m,
            patch_radius: cfg.patch_radius_m,
            mode: ContactMode::AnisotropicScale {
                mu_fwd: cfg.mu_fwd,
                mu_rev: cfg.mu_rev,
                mu_lat: cfg.mu_lat,
                pivot_mu: cfg.pivot_mu,
                static_mu_ratio: cfg.static_mu_ratio,
                // Robot crawls along -X (matching wheel forward = cross(+Z, +Y) = -X)
                fwd_axis_local: Vec3::new(-1.0, 0.0, 0.0),
            },
            material_profile_id: cfg.material_profile_id,
            solref_imp: None,
        })
        .collect();

    graph
}
