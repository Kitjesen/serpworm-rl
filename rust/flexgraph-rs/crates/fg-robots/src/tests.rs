use crate::{
    build_archetype_by_name, build_archetype_default_world, build_bristle_worm,
    build_endoscope_spline_tube_sdf, build_euler_elastica_rod, build_flexspine_centipede,
    build_rigid_flex_grid, build_rigid_flex_grid_4x4, build_rigid_flex_grid_6x6, build_serpworm,
    build_soft_octopus_arm, build_tendon_endoscope, build_tensegrity_6bar,
    cantilever_small_deflection, default_endoscope_lumen_waypoints, elastica_cantilever_tip,
    euler_buckling_critical_load, load_graph_from_json_str, load_graph_from_mjcf_str,
    load_graph_from_urdf_flex_str, load_graph_from_yaml_str,
    tensegrity_6bar_equilibrium_residual, BristleWormConfig, CentipedeConfig, EndoscopeConfig,
    FlexGridConfig, OctopusArmConfig, ParametersJson, SerpWormConfig, SerpWormPreset,
    TensegrityConfig,
};
#[cfg(feature = "ros-urdf")]
use crate::load_graph_from_ros_urdf_str;
use fg_ir::{BeamFormulation, ColliderGeometry, JointType, SchurRoute, SymmetryGroupKind};
use fg_math::{BeamSection, Vec3};
use fg_sdf::SdfField;

#[test]
fn test_serpworm_n1_n3_n5_n7_geometry_and_block_diagonal_pairs() {
    let params = ParametersJson::load_repo_or_embedded();
    assert_eq!(params.strip_count, 8);
    assert!((params.youngs_modulus_pa - 200e9).abs() < 1e-3);

    for &(n_seg, elems) in &[(1, 16), (3, 16), (5, 16), (7, 16)] {
        let cfg = SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, n_seg, elems);
        let g = build_serpworm(&cfg, Some(&params));
        g.validate().expect("serpworm graph must validate");
        let topo = g.analyze().expect("serpworm graph must analyze");

        assert_eq!(topo.route, SchurRoute::BlockDiagonalPairs);
        assert_eq!(topo.num_bodies, 2 * n_seg);
        assert_eq!(topo.num_free_bodies, 2 * n_seg);
        assert_eq!(topo.pair_blocks.len(), n_seg);
        assert_eq!(topo.single_blocks.len(), 0);
        assert_eq!(topo.cyclomatic_number, 0);
        assert_eq!(g.beams.len(), 8 * n_seg);
        assert_eq!(g.tendons.len(), 4 * n_seg);
        assert_eq!(g.contact_sites.len(), 4 * n_seg);
        assert_eq!(topo.indexing.num_beam_nodes, 8 * n_seg * (elems - 1));
        assert_eq!(topo.indexing.total_nodes, 2 * n_seg + 8 * n_seg * (elems - 1));
        assert_eq!(g.symmetry.kind, SymmetryGroupKind::Z2Bilateral);
        assert_eq!(g.symmetry.elements.len(), 2);

        // Verify exact z0 resting clearance above ground: z0 + mount_z - wheel_r == +0.001 m
        let z_plate0 = g.bodies[0].rest_pose.pos.z();
        let wheel_z = g.contact_sites[0].local_pos.z();
        let wheel_r = g.contact_sites[0].radius;
        assert!((z_plate0 + wheel_z - wheel_r - 0.001).abs() < 1e-12);
    }
}

#[test]
fn test_bristle_worm_z2xz2_and_anisotropic_scales() {
    let cfg = BristleWormConfig {
        n_segments: 3,
        elements_per_strip: 8,
        ..Default::default()
    };
    let g = build_bristle_worm(&cfg, None);
    g.validate().unwrap();
    let topo = g.analyze().unwrap();
    assert_eq!(topo.route, SchurRoute::BlockDiagonalPairs);
    assert_eq!(g.symmetry.kind, SymmetryGroupKind::Z2xZ2Bidirectional);
    assert_eq!(g.symmetry.elements.len(), 4);
    // Verify exact geometric yaw signs across Z2xZ2 elements:
    // e: +1, lr (XZ mirror): -1, fb (180-deg Z-rotation): -1, lr*fb (YZ mirror): +1
    let yaw_idx = 2 * cfg.n_segments;
    for el in &g.symmetry.elements {
        match el.name.as_str() {
            "e" => assert!((el.act_sign[yaw_idx] - 1.0).abs() < 1e-12),
            "e_lr" => assert!((el.act_sign[yaw_idx] + 1.0).abs() < 1e-12),
            "e_fb" => assert!((el.act_sign[yaw_idx] + 1.0).abs() < 1e-12),
            _ => assert!((el.act_sign[yaw_idx] - 1.0).abs() < 1e-12),
        }
    }
    assert_eq!(g.contact_sites.len(), 12);
    // Resting ground clearance preserved at exactly 1 mm
    let z_p0 = g.bodies[0].rest_pose.pos.z();
    let c0 = &g.contact_sites[0];
    assert!((z_p0 + c0.local_pos.z() - c0.radius - 0.001).abs() < 1e-12);
}

#[test]
fn test_tendon_endoscope_3d_tree_chain_d4_and_cantilever() {
    let cfg = EndoscopeConfig::default();
    let g = build_tendon_endoscope(&cfg);
    g.validate().unwrap();
    let topo = g.analyze().unwrap();
    assert_eq!(topo.route, SchurRoute::TreeChain);
    assert_eq!(topo.num_bodies, 5);
    assert_eq!(topo.num_free_bodies, 4);
    assert!(g.bodies[0].is_fixed);
    assert_eq!(g.tendons.len(), 4);
    assert_eq!(g.tendons[0].waypoints.len(), 5);
    assert_eq!(g.symmetry.kind, SymmetryGroupKind::D4Dihedral);
    assert_eq!(g.symmetry.elements.len(), 8);

    let lumen = default_endoscope_lumen_waypoints(&cfg);
    assert_eq!(lumen.control_points.len(), 4);
    assert_eq!(lumen.radii_m.len(), 4);
    let tube = build_endoscope_spline_tube_sdf(&cfg);
    // Base and proximal spacer discs sit inside the 3D curved SplineTubeSdf lumen
    assert!(tube.query(g.bodies[0].rest_pose.pos).phi > 0.0);
    assert!(tube.query(g.bodies[1].rest_pose.pos).phi > 0.0);

    // Free cantilever tip variant (body_b == None on final beam)
    let cant_cfg = EndoscopeConfig {
        free_cantilever_tip: true,
        ..cfg
    };
    let gc = build_tendon_endoscope(&cant_cfg);
    gc.validate().unwrap();
    let topoc = gc.analyze().unwrap();
    assert_eq!(topoc.route, SchurRoute::TreeChain);
    assert!(gc.beams.last().unwrap().body_b.is_none());
    assert_eq!(
        gc.beams.last().unwrap().num_free_beam_nodes(),
        cant_cfg.elements_per_section
    );
}

#[test]
fn test_flexspine_centipede_tree_chain_and_bekker_feet() {
    let cfg = CentipedeConfig::default();
    let g = build_flexspine_centipede(&cfg);
    g.validate().unwrap();
    let topo = g.analyze().unwrap();
    assert_eq!(topo.route, SchurRoute::TreeChain);
    assert_eq!(topo.cyclomatic_number, 0);
    assert_eq!(g.bodies.len(), 9);
    assert_eq!(g.beams.len(), 4);
    assert_eq!(g.joints.len(), 6);
    assert_eq!(g.contact_sites.len(), 6);
    assert!(g.contact_sites.iter().all(|c| c.material_profile_id == 2));
    assert_eq!(g.symmetry.elements.len(), 2);
}

#[test]
fn test_tensegrity_6bar_cyclic_sparse_general_and_exact_self_equilibrium() {
    let cfg = TensegrityConfig::default();
    let g = build_tensegrity_6bar(&cfg);
    g.validate().unwrap();
    let topo = g.analyze().unwrap();

    // Classic 6-bar icosahedral tensegrity = K_{2,2,2} octahedron graph:
    // V = 6 free struts, E = 12 undirected strut pairs, C = 1 component -> beta_1 = 7!
    assert_eq!(topo.route, SchurRoute::SparseGeneral);
    assert_eq!(topo.num_free_bodies, 6);
    assert_eq!(topo.num_rigid_edges, 12);
    assert_eq!(topo.components.len(), 1);
    assert_eq!(topo.cyclomatic_number, 7);
    assert_eq!(g.beams.len(), 24);
    assert_eq!(g.tendons.len(), 24);
    assert_eq!(g.contact_sites.len(), 12);
    assert_eq!(topo.sparse_pattern.n_blocks, 6);
    assert_eq!(topo.sparse_pattern.nnz_offdiag_a, 12);

    // Verify all 24 cables have identical Euclidean length sqrt(6)*b = sqrt(6)/4 * L
    let expected_len = 6.0_f64.sqrt() * 0.25 * cfg.strut_length_m;
    for beam in &g.beams {
        let total_l: f64 = beam.resolved_lengths().iter().sum();
        assert!(
            (total_l - expected_len).abs() < 1e-12,
            "cable {} length {} != expected {}",
            beam.name,
            total_l,
            expected_len
        );
    }

    // Verify exact static force & torque self-equilibrium under uniform cable tension!
    let (max_f, max_t) = tensegrity_6bar_equilibrium_residual(&g);
    assert!(
        max_f < 1e-13,
        "tensegrity 6-bar force equilibrium residual too large: {max_f}"
    );
    assert!(
        max_t < 1e-13,
        "tensegrity 6-bar torque equilibrium residual too large: {max_t}"
    );
}

#[test]
fn test_declarative_json_yaml_and_urdf_flex_loaders() {
    // 1. JSON & YAML round-trip across all 5 archetypes
    for name in [
        "serpworm_3",
        "bristle_worm",
        "tendon_endoscope_3d",
        "flexspine_centipede",
        "tensegrity_6bar",
    ] {
        let g = build_archetype_by_name(name).unwrap();
        let world = build_archetype_default_world(name).unwrap();
        assert!(world.query_world(g.bodies[0].rest_pose.pos).phi.is_finite());
        let json = g.to_json_string().unwrap();
        let g_json = load_graph_from_json_str(&json).unwrap();
        assert_eq!(g_json.analyze().unwrap().route, g.analyze().unwrap().route);

        let yaml = g.to_yaml_string().unwrap();
        let g_yaml = load_graph_from_yaml_str(&yaml).unwrap();
        assert_eq!(g_yaml.bodies.len(), g.bodies.len());
        assert_eq!(g_yaml.beams.len(), g.beams.len());
    }

    // 2. Compact YAML archetype envelope
    let compact_yaml = "archetype: tensegrity6bar\n";
    let g_env = load_graph_from_yaml_str(compact_yaml).unwrap();
    assert_eq!(g_env.analyze().unwrap().route, SchurRoute::SparseGeneral);

    // 3. URDF-Flex XML parser (links, revolute joint, flexible beam, tendon, contact site, fixed base)
    let urdf_xml = r#"<?xml version="1.0"?>
    <robot name="mini_flex_arm">
      <!-- Fixed base + 2 articulated/compliant links -->
      <gravity xyz="0 0 -9.81"/>
      <solver_hints dt="0.001" rayleigh_alpha="0.08" rayleigh_beta="0.001"/>
      <fixed_base link="base"/>
      <link name="base" xyz="0 0 0.15" collision_radius="0.02">
        <inertial>
          <mass value="0.5"/>
          <inertia ixx="0.001" iyy="0.001" izz="0.001"/>
        </inertial>
      </link>
      <link name="mid" xyz="0.10 0 0.15">
        <inertial>
          <mass value="0.1"/>
          <inertia ixx="0.0002" iyy="0.0002" izz="0.0002"/>
        </inertial>
      </link>
      <link name="tip">
        <inertial>
          <mass value="0.08"/>
          <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
        </inertial>
      </link>
      <joint name="wrist" type="revolute" kp="6.0" kd="0.1" implicit_coupling="true">
        <parent link="mid"/>
        <child link="tip"/>
        <origin xyz="0.08 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.0" upper="1.0" effort="1.5" velocity="5.0"/>
      </joint>
      <joint name="stay_cable" type="cable_spring" anchor_a="0 0 0.02" anchor_b="0 0 0.02" stiffness="600" damping="1.5" compression_allowed="false">
        <parent link="base"/>
        <child link="tip"/>
      </joint>
      <beam name="flex_link" body_a="base" body_b="mid" n_elem="4" section="circle" radius="0.0015" E="65e9" nu="0.33" rho="6450"/>
      <beam name="whisker" body_a="tip" n_elem="3" length="0.05" dir="1 0 0" section="circle" radius="0.0008" E="65e9"/>
      <tendon name="pull_y" mode="linear" stiffness="2500" damping="1.0" capstan_mu_theta="0.15" waypoints="base:0,0.01,0;mid:0,0.01,0;tip:0,0.01,0"/>
      <contact_site name="tip_pad" body="tip" pos="0 0 -0.01" mode="isotropic_pad" radius="0.01" mu="0.6"/>
    </robot>"#;

    let g_urdf = load_graph_from_urdf_flex_str(urdf_xml).unwrap();
    assert_eq!(g_urdf.name, "mini_flex_arm");
    assert_eq!(g_urdf.bodies.len(), 3);
    assert!(g_urdf.bodies[0].is_fixed);
    assert!((g_urdf.bodies[2].rest_pose.pos.x() - 0.18).abs() < 1e-12);
    assert_eq!(g_urdf.beams.len(), 2);
    assert!(g_urdf.beams[1].body_b.is_none()); // free cantilever whisker
    assert_eq!(g_urdf.tendons.len(), 1);
    assert_eq!(g_urdf.tendons[0].waypoints.len(), 3);
    assert!(g_urdf.tendons[0].stick_slip_enabled);
    assert_eq!(g_urdf.joints.len(), 2);
    assert!(matches!(
        g_urdf.joints[1].joint_type,
        JointType::CableSpring {
            stiffness,
            compression_allowed: false,
            ..
        } if (stiffness - 600.0).abs() < 1e-12
    ));
    assert_eq!(g_urdf.contact_sites.len(), 1);
    let topo = g_urdf.analyze().unwrap();
    assert_eq!(topo.num_free_bodies, 2);
}

#[test]
fn test_soft_octopus_arm_der_and_elastica_analytical() {
    let cfg = OctopusArmConfig::default();
    let g = build_soft_octopus_arm(&cfg);
    g.validate().unwrap();
    let topo = g.analyze().unwrap();
    assert_eq!(topo.route, SchurRoute::TreeChain);
    assert_eq!(topo.num_bodies, cfg.n_sections + 1);
    assert_eq!(topo.num_free_bodies, cfg.n_sections);
    assert!(g.bodies[0].is_fixed);
    assert_eq!(g.beams.len(), cfg.n_sections);
    assert!(g
        .beams
        .iter()
        .all(|b| b.formulation == BeamFormulation::DiscreteElasticRod));
    // Verify strictly monotonic tapering of circular cross-sections from base to tip
    let radii: Vec<f64> = g
        .beams
        .iter()
        .map(|b| match b.section {
            BeamSection::Circular { radius } => radius,
            _ => panic!("expected circular section on octopus arm"),
        })
        .collect();
    for w in radii.windows(2) {
        assert!(w[0] > w[1], "octopus arm radius must taper: {} > {}", w[0], w[1]);
    }
    assert_eq!(g.tendons.len(), 4);
    assert_eq!(g.symmetry.kind, SymmetryGroupKind::D4Dihedral);
    assert_eq!(g.symmetry.elements.len(), 8);
    assert!(!g.diff_params.is_empty());

    // Free cantilever tip variant
    let cant_cfg = OctopusArmConfig {
        free_cantilever_tip: true,
        ..cfg
    };
    let gc = build_soft_octopus_arm(&cant_cfg);
    gc.validate().unwrap();
    assert_eq!(gc.analyze().unwrap().route, SchurRoute::TreeChain);
    assert!(gc.beams.last().unwrap().body_b.is_none());

    // Archetype registry + JSON/YAML roundtrip
    let g_reg = build_archetype_by_name("soft_octopus_arm_der").unwrap();
    assert_eq!(g_reg.beams.len(), 4);
    let json = g_reg.to_json_string().unwrap();
    let g_rt = load_graph_from_json_str(&json).unwrap();
    assert_eq!(g_rt.beams.len(), 4);
    let world = build_archetype_default_world("soft_octopus_arm_der").unwrap();
    assert!(world.query_world(g_reg.bodies[0].rest_pose.pos).phi.is_finite());

    // Canonical Euler/Elastica rod builder + analytical solutions
    let l = 0.5;
    let r = 0.005;
    let e_pa = 20e6;
    let rod = build_euler_elastica_rod(
        l,
        r,
        e_pa,
        1100.0,
        12,
        BeamFormulation::DiscreteElasticRod,
        Vec3::new(0.0, 0.0, -9.81),
        Some(0.02),
    );
    rod.validate().unwrap();
    // 1 fixed base + 1 free tip body -> single-body component classified as BlockDiagonalPairs
    assert_eq!(rod.analyze().unwrap().route, SchurRoute::BlockDiagonalPairs);

    let p_cr = euler_buckling_critical_load(e_pa, r, l);
    let ei = e_pa * 0.25 * std::f64::consts::PI * r.powi(4);
    assert!((p_cr - std::f64::consts::PI.powi(2) * ei / (4.0 * l * l)).abs() < 1e-14);

    // Small-load limit: nonlinear Elastica agrees with linear FL^3/(3EI) to < 1e-4
    let f_small = 1e-3 * ei / (l * l);
    let d_lin = cantilever_small_deflection(f_small, e_pa, r, l);
    let (x_s, dz_s, th_s) = elastica_cantilever_tip(f_small, e_pa, r, l);
    assert!(((dz_s - d_lin) / d_lin).abs() < 1e-4);
    assert!((x_s - l).abs() < 1e-5 * l);
    assert!(((th_s - f_small * l * l / (2.0 * ei)) / th_s).abs() < 1e-4);

    // Large nonlinear load: Bisshopp-Drucker benchmark FL^2/(EI) = 4.0.
    // Cross-validate elliptic-integral solution directly against RK4 integration of the
    // exact Elastica ODE (theta' = kappa, kappa' = alpha*cos(theta), kappa(0) = -sqrt(2*alpha*sin(phi_L))):
    let f_large = 4.0 * ei / (l * l);
    let (x_l, dz_l, th_l) = elastica_cantilever_tip(f_large, e_pa, r, l);
    let alpha = f_large / ei;
    let mut th = 0.0_f64;
    let mut kp = -(2.0 * alpha * th_l.sin()).sqrt();
    let mut xr = 0.0_f64;
    let mut zr = 0.0_f64;
    let n_rk = 400usize;
    let ds = l / (n_rk as f64);
    let rhs = |t: f64, k: f64| (k, alpha * t.cos(), t.cos(), t.sin());
    for _ in 0..n_rk {
        let (dt1, dk1, dx1, dz1) = rhs(th, kp);
        let (dt2, dk2, dx2, dz2) = rhs(th + 0.5 * ds * dt1, kp + 0.5 * ds * dk1);
        let (dt3, dk3, dx3, dz3) = rhs(th + 0.5 * ds * dt2, kp + 0.5 * ds * dk2);
        let (dt4, dk4, dx4, dz4) = rhs(th + ds * dt3, kp + ds * dk3);
        th += (ds / 6.0) * (dt1 + 2.0 * dt2 + 2.0 * dt3 + dt4);
        kp += (ds / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4);
        xr += (ds / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
        zr += (ds / 6.0) * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4);
    }
    assert!(kp.abs() < 1e-8, "free-tip curvature must vanish, got {kp}");
    assert!((th + th_l).abs() < 1e-8, "ODE tip angle mismatch");
    assert!((xr - x_l).abs() < 1e-8, "ODE x_tip mismatch: {xr} vs {x_l}");
    assert!((-zr - dz_l).abs() < 1e-8, "ODE dz_tip mismatch: {} vs {dz_l}", -zr);
    assert!((x_l / l - 0.671059).abs() < 1e-4, "x_l/L = {}", x_l / l);
    assert!((dz_l / l - 0.66996).abs() < 1e-3, "dz_l/L = {}", dz_l / l);
    assert!((x_l * x_l + dz_l * dz_l).sqrt() < l);
}

#[test]
fn test_rigid_flex_grid_4x4_and_6x6_cyclic_sparse_general() {
    // 4x4 lattice: 16 free bodies, 24 beams, cyclomatic number beta_1 = 24 - 16 + 1 = 9
    let g4 = build_rigid_flex_grid_4x4();
    g4.validate().unwrap();
    let t4 = g4.analyze().unwrap();
    assert_eq!(t4.route, SchurRoute::SparseGeneral);
    assert_eq!(t4.num_free_bodies, 16);
    assert_eq!(g4.beams.len(), 24);
    assert_eq!(t4.num_rigid_edges, 24);
    assert_eq!(t4.cyclomatic_number, 9);
    assert_eq!(g4.tendons.len(), 4);
    assert_eq!(g4.contact_sites.len(), 16);
    assert_eq!(g4.symmetry.kind, SymmetryGroupKind::Z2Bilateral);
    assert_eq!(g4.symmetry.elements.len(), 2);

    // 6x6 lattice: 36 free bodies, 60 beams, cyclomatic number beta_1 = 60 - 36 + 1 = 25
    let g6 = build_rigid_flex_grid_6x6();
    g6.validate().unwrap();
    let t6 = g6.analyze().unwrap();
    assert_eq!(t6.route, SchurRoute::SparseGeneral);
    assert_eq!(t6.num_free_bodies, 36);
    assert_eq!(g6.beams.len(), 60);
    assert_eq!(t6.num_rigid_edges, 60);
    assert_eq!(t6.cyclomatic_number, 25);

    // Pinned-corner 4x4 lattice: 12 free bodies, 4 fixed corners
    let g4_pin = build_rigid_flex_grid(&FlexGridConfig {
        pin_corners: true,
        ..Default::default()
    });
    g4_pin.validate().unwrap();
    let t4_pin = g4_pin.analyze().unwrap();
    assert_eq!(t4_pin.route, SchurRoute::SparseGeneral);
    assert_eq!(t4_pin.num_free_bodies, 12);

    // Registry + YAML roundtrip
    let g_reg = build_archetype_by_name("rigid_flex_grid_4x4").unwrap();
    assert_eq!(g_reg.beams.len(), 24);
    let g_reg6 = build_archetype_by_name("rigid_flex_grid_6x6").unwrap();
    assert_eq!(g_reg6.beams.len(), 60);
    let yaml = g_reg.to_yaml_string().unwrap();
    let g_rt = load_graph_from_yaml_str(&yaml).unwrap();
    assert_eq!(g_rt.analyze().unwrap().cyclomatic_number, 9);
}

#[cfg(feature = "ros-urdf")]
#[test]
fn test_ros_urdf_importer_standard_urdf() {
    let std_urdf = r#"<?xml version="1.0"?>
    <robot name="ros_test_manipulator">
      <link name="world"/>
      <link name="base_link">
        <inertial>
          <origin xyz="0 0 0.05" rpy="0 0 0"/>
          <mass value="2.0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.015"/>
        </inertial>
        <collision name="base_box">
          <origin xyz="0 0 0.05" rpy="0 0 0"/>
          <geometry><box size="0.20 0.20 0.10"/></geometry>
        </collision>
      </link>
      <link name="upper_arm">
        <inertial>
          <origin xyz="0.15 0 0" rpy="0 0 0"/>
          <mass value="0.8"/>
          <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.006" iyz="0" izz="0.006"/>
        </inertial>
        <collision name="arm_cyl">
          <origin xyz="0.15 0 0" rpy="0 1.57079632679 0"/>
          <geometry><cylinder radius="0.03" length="0.30"/></geometry>
        </collision>
      </link>
      <link name="forearm">
        <inertial>
          <mass value="0.5"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
        </inertial>
        <collision name="forearm_cap">
          <geometry><capsule radius="0.025" length="0.22"/></geometry>
        </collision>
      </link>
      <link name="slider_carriage">
        <inertial>
          <mass value="0.3"/>
          <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/>
        </inertial>
        <collision name="tip_sphere">
          <geometry><sphere radius="0.02"/></geometry>
        </collision>
      </link>
      <link name="gripper_jaw">
        <inertial>
          <mass value="0.1"/>
          <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/>
        </inertial>
        <collision name="jaw_mesh">
          <geometry><mesh filename="package://demo/meshes/jaw.stl" scale="1.0 1.0 1.0"/></geometry>
        </collision>
      </link>
      <joint name="world_to_base" type="fixed">
        <parent link="world"/>
        <child link="base_link"/>
        <origin xyz="0 0 0.20" rpy="0 0 0"/>
      </joint>
      <joint name="shoulder_yaw" type="revolute">
        <parent link="base_link"/>
        <child link="upper_arm"/>
        <origin xyz="0 0 0.10" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-2.5" upper="2.5" effort="12.0" velocity="4.0"/>
        <dynamics damping="0.25" friction="0.05"/>
      </joint>
      <joint name="elbow_spin" type="continuous">
        <parent link="upper_arm"/>
        <child link="forearm"/>
        <origin xyz="0.30 0 0" rpy="0 0 1.5707963267948966"/>
        <axis xyz="1 0 0"/>
        <dynamics damping="0.1"/>
      </joint>
      <joint name="wrist_slide" type="prismatic">
        <parent link="forearm"/>
        <child link="slider_carriage"/>
        <origin xyz="0 -0.22 0" rpy="0 0 -1.5707963267948966"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.05" upper="0.05" effort="8.0" velocity="1.0"/>
      </joint>
      <joint name="jaw_mimic" type="revolute">
        <parent link="slider_carriage"/>
        <child link="gripper_jaw"/>
        <origin xyz="0.04 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-0.6" upper="0.6" effort="3.0" velocity="2.0"/>
        <mimic joint="shoulder_yaw" multiplier="0.5" offset="0.1"/>
      </joint>
    </robot>"#;

    let g = load_graph_from_ros_urdf_str(std_urdf).unwrap();
    g.validate().unwrap();
    assert_eq!(g.name, "ros_test_manipulator");
    assert_eq!(g.bodies.len(), 6);
    // world link is fixed
    assert!(g.bodies[0].is_fixed);
    // Kinematic rest pose propagation: base_link at z=0.20, upper_arm at z=0.30, forearm at x=0.30, z=0.30
    let forearm = g.bodies.iter().find(|b| b.name == "forearm").unwrap();
    assert!((forearm.rest_pose.pos.x() - 0.30).abs() < 1e-12);
    assert!((forearm.rest_pose.pos.z() - 0.30).abs() < 1e-12);
    let jaw = g.bodies.iter().find(|b| b.name == "gripper_jaw").unwrap();
    assert!((jaw.rest_pose.pos.x() - 0.56).abs() < 1e-12);

    // 5 joints mapped cleanly (weld, revolute, continuous, prismatic, mimic revolute)
    assert_eq!(g.joints.len(), 5);
    assert!(matches!(g.joints[0].joint_type, JointType::Weld { .. }));
    assert!(matches!(g.joints[1].joint_type, JointType::Revolute { .. }));
    assert!(matches!(
        g.joints[1].limits,
        Some([lo, hi]) if (lo + 2.5).abs() < 1e-12 && (hi - 2.5).abs() < 1e-12
    ));
    assert!(matches!(g.joints[2].joint_type, JointType::Revolute { .. }));
    // Verify URDF <axis> is rotated by joint <origin rpy="0 0 pi/2">: [1,0,0] -> [0,1,0] in parent frame!
    if let JointType::Revolute { axis_local_a, .. } = g.joints[2].joint_type {
        assert!(
            (axis_local_a - Vec3::EY).norm() < 1e-10,
            "rotated URDF joint axis expected +Y, got {:?}",
            axis_local_a
        );
    }
    assert!(g.joints[2].limits.is_none());
    assert!(matches!(g.joints[3].joint_type, JointType::Prismatic { .. }));
    assert!(matches!(
        g.joints[3].limits,
        Some([lo, hi]) if (lo + 0.05).abs() < 1e-12 && (hi - 0.05).abs() < 1e-12
    ));

    // All 5 collision shapes populated into mesh_colliders
    assert_eq!(g.mesh_colliders.len(), 5);
    assert!(matches!(
        g.mesh_colliders[0].geometry,
        ColliderGeometry::Box { half_extents } if (half_extents.x() - 0.10).abs() < 1e-12
    ));
    assert!(matches!(
        g.mesh_colliders[1].geometry,
        ColliderGeometry::Cylinder { radius, half_height }
            if (radius - 0.03).abs() < 1e-12 && (half_height - 0.15).abs() < 1e-12
    ));
    assert!(matches!(
        g.mesh_colliders[2].geometry,
        ColliderGeometry::Capsule { radius, half_height }
            if (radius - 0.025).abs() < 1e-12 && (half_height - 0.11).abs() < 1e-12
    ));
    assert!(matches!(
        g.mesh_colliders[3].geometry,
        ColliderGeometry::Sphere { radius } if (radius - 0.02).abs() < 1e-12
    ));
    assert!(matches!(
        g.mesh_colliders[4].geometry,
        ColliderGeometry::ExternalMesh { ref uri, .. } if uri.contains("jaw.stl")
    ));
    assert!(!g.contact_sites.is_empty());
    assert_eq!(g.analyze().unwrap().route, SchurRoute::TreeChain);
}

#[test]
fn test_mjcf_importer_ant_swimmer_tendon_finger_and_flexcomp() {
    // 1. MuJoCo Ant XML (freejoint, sphere torso, 4x2 fromto capsule legs, 8 hinges, motors, solref/solimp)
    let ant_xml = r#"<mujoco model="ant">
      <compiler angle="degree" coordinate="local"/>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <default>
        <joint armature="0.01" damping="1.0" limited="true"/>
        <geom conaffinity="0" condim="3" density="500.0" friction="1.0 0.5 0.5" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2.0"/>
      </default>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="40 40 40"/>
        <body name="torso" pos="0 0 0.75">
          <freejoint name="root"/>
          <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
          <body name="front_left_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
            <body name="aux_1" pos="0.2 0.2 0">
              <joint axis="0 0 1" name="hip_1" pos="0 0 0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
              <body pos="0.2 0.2 0" name="ankle_1_body">
                <joint axis="-1 1 0" name="ankle_1" pos="0 0 0" range="30 70" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
              </body>
            </body>
          </body>
          <body name="front_right_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
            <body name="aux_2" pos="-0.2 0.2 0">
              <joint axis="0 0 1" name="hip_2" pos="0 0 0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
              <body pos="-0.2 0.2 0" name="ankle_2_body">
                <joint axis="1 1 0" name="ankle_2" pos="0 0 0" range="-70 -30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
      </actuator>
    </mujoco>"#;

    let g_ant = load_graph_from_mjcf_str(ant_xml).unwrap();
    g_ant.validate().unwrap();
    assert_eq!(g_ant.name, "ant");
    assert!((g_ant.solver_hints.preferred_dt - 0.002).abs() < 1e-12);
    // Floor plane skipped as world SDF; torso is free floating (freejoint)
    assert!(!g_ant.bodies[0].is_fixed);
    assert!(g_ant.bodies.len() >= 7);
    assert!(g_ant.joints.len() >= 4);
    assert!(g_ant.mesh_colliders.len() >= 7);
    // Verify degree -> radian conversion on hip_1 range [-30, 30] deg
    let hip1 = g_ant.joints.iter().find(|j| j.name == "hip_1").unwrap();
    assert!(matches!(hip1.joint_type, JointType::Revolute { .. }));
    let [lo, hi] = hip1.limits.expect("hip_1 must have limits");
    assert!((lo + 30.0_f64.to_radians()).abs() < 1e-12);
    assert!((hi - 30.0_f64.to_radians()).abs() < 1e-12);
    // Verify SolRefImp populated from <default><geom solref/solimp>
    assert!(g_ant.contact_sites.iter().all(|c| c.solref_imp.is_some()));
    let sri = g_ant.contact_sites[0].solref_imp.unwrap();
    assert!((sri.timeconst - 0.02).abs() < 1e-12);
    assert!((sri.dmin - 0.90).abs() < 1e-12);

    // 2. MuJoCo Swimmer XML (3-link planar capsule chain)
    let swimmer_xml = r#"<mujoco model="swimmer">
      <compiler angle="degree"/>
      <option timestep="0.004" gravity="0 0 0"/>
      <worldbody>
        <body name="torso" pos="0 0 0.1">
          <freejoint name="root"/>
          <geom name="torso_cap" fromto="1.0 0 0 0 0 0" size="0.1" type="capsule" density="1000"/>
          <body name="mid" pos="0 0 0">
            <joint axis="0 0 1" name="rot2" pos="0 0 0" range="-100 100" type="hinge"/>
            <geom name="mid_cap" fromto="0 0 0 -1.0 0 0" size="0.1" type="capsule" density="1000"/>
            <body name="back" pos="-1.0 0 0">
              <joint axis="0 0 1" name="rot3" pos="0 0 0" range="-100 100" type="hinge"/>
              <geom name="back_cap" fromto="0 0 0 -1.0 0 0" size="0.1" type="capsule" density="1000"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="rot2" gear="150"/>
        <motor joint="rot3" gear="150"/>
      </actuator>
    </mujoco>"#;

    let g_swim = load_graph_from_mjcf_str(swimmer_xml).unwrap();
    g_swim.validate().unwrap();
    assert_eq!(g_swim.bodies.len(), 3);
    assert_eq!(g_swim.joints.len(), 2);
    assert_eq!(g_swim.analyze().unwrap().route, SchurRoute::TreeChain);
    // Verify Swimmer capsule along X has axial inertia on I_xx (strictly smaller than yaw I_zz!)
    // and COM centered at x = 0.5 (not shifted about body origin).
    let swim_torso = &g_swim.bodies[0];
    assert!(
        (swim_torso.com_local - Vec3::new(0.5, 0.0, 0.0)).norm() < 1e-12,
        "swimmer torso COM expected (0.5,0,0), got {:?}",
        swim_torso.com_local
    );
    let ixx = swim_torso.inertia.0[0][0];
    let iyy = swim_torso.inertia.0[1][1];
    let izz = swim_torso.inertia.0[2][2];
    assert!(
        ixx < 0.1 * izz && (iyy - izz).abs() < 1e-10,
        "swimmer X-aligned capsule must satisfy I_xx << I_yy == I_zz, got ({ixx}, {iyy}, {izz})"
    );

    // Verify MJCF Cylinder mass excludes spherical endcaps (strictly less than Capsule of same r, half_height)
    let cyl_vs_cap_xml = r#"<mujoco model="cyl_vs_cap">
      <worldbody>
        <body name="b_cyl" pos="0 0 0.5"><freejoint/><geom type="cylinder" size="0.1 0.2" density="1000"/></body>
        <body name="b_cap" pos="1 0 0.5"><freejoint/><geom type="capsule" size="0.1 0.2" density="1000"/></body>
      </worldbody>
    </mujoco>"#;
    let g_cc = load_graph_from_mjcf_str(cyl_vs_cap_xml).unwrap();
    let expected_cyl_m = 1000.0 * std::f64::consts::PI * 0.1_f64.powi(2) * 0.4;
    let expected_cap_m = expected_cyl_m + 1000.0 * (4.0 / 3.0) * std::f64::consts::PI * 0.1_f64.powi(3);
    assert!((g_cc.bodies[0].mass - expected_cyl_m).abs() < 1e-10);
    assert!((g_cc.bodies[1].mass - expected_cap_m).abs() < 1e-10);

    // 3. MuJoCo Tendon-Finger + <flexcomp> flexible rod + <site> + <tendon><spatial> + <position>
    let finger_xml = r#"<mujoco model="tendon_finger_flex">
      <compiler angle="radian"/>
      <option timestep="0.001" gravity="0 0 -9.81"/>
      <worldbody>
        <body name="palm" pos="0 0 0.25">
          <inertial pos="0 0 0" mass="0.4" diaginertia="0.001 0.001 0.001"/>
          <geom name="palm_box" type="box" size="0.04 0.03 0.015" solref="-4000 -40"/>
          <site name="s_palm" pos="0.04 0 -0.015"/>
          <body name="proximal" pos="0.04 0 0">
            <inertial pos="0.03 0 0" mass="0.05" diaginertia="2e-5 2e-5 2e-5"/>
            <joint name="mcp" type="hinge" axis="0 1 0" range="-0.2 1.4" damping="0.02"/>
            <geom name="prox_cap" type="capsule" fromto="0 0 0 0.06 0 0" size="0.01"/>
            <site name="s_prox" pos="0.03 0 -0.012"/>
            <body name="distal" pos="0.06 0 0">
              <inertial pos="0.025 0 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
              <joint name="pip" type="hinge" axis="0 1 0" range="0.0 1.4" damping="0.01"/>
              <geom name="dist_cap" type="capsule" fromto="0 0 0 0.05 0 0" size="0.009"/>
              <site name="s_dist" pos="0.045 0 -0.01"/>
            </body>
          </body>
        </body>
        <flexcomp name="elastic_spine" type="cable" count="6 1 1" spacing="0.02 0.02 0.02" pos="0 0.05 0.25" radius="0.002" mass="0.04" young="2e7"/>
      </worldbody>
      <tendon>
        <spatial name="flexor" stiffness="1500" damping="1.2" solreflimit="0.01 1.0" solimplimit="0.9 0.95 0.001 0.5 2.0">
          <site site="s_palm"/>
          <site site="s_prox"/>
          <site site="s_dist"/>
        </spatial>
      </tendon>
      <actuator>
        <position name="flexor_act" tendon="flexor" kp="2000" kv="2.0" ctrlrange="-0.03 0.03"/>
        <general name="mcp_servo" joint="mcp" gainprm="12.0" biasprm="0 -12.0 -0.1"/>
      </actuator>
    </mujoco>"#;

    let g_finger = load_graph_from_mjcf_str(finger_xml).unwrap();
    g_finger.validate().unwrap();
    assert_eq!(g_finger.name, "tendon_finger_flex");
    // palm has no freejoint -> fixed base!
    assert!(g_finger.bodies[0].is_fixed);
    assert_eq!(g_finger.tendons.len(), 1);
    assert_eq!(g_finger.tendons[0].waypoints.len(), 3);
    assert!(g_finger.tendons[0].solref_imp.is_some());
    assert!((g_finger.tendons[0].stiffness - 2000.0).abs() < 1e-12);
    // <flexcomp> generated a flexible beam + anchor body
    assert_eq!(g_finger.beams.len(), 1);
    assert_eq!(g_finger.beams[0].n_elements, 5);
    // Negative direct-stiffness solref on palm_box preserved on MeshColliderSpec
    let palm_c = g_finger
        .mesh_colliders
        .iter()
        .find(|c| c.name == "palm_box")
        .unwrap();
    assert!((palm_c.solref_imp.unwrap().timeconst + 4000.0).abs() < 1e-12);
    // <general> gainprm/biasprm wired onto mcp joint
    let mcp = g_finger.joints.iter().find(|j| j.name == "mcp").unwrap();
    assert!((mcp.kp - 12.0).abs() < 1e-12);
    assert!((mcp.kd - 0.1).abs() < 1e-12);
}
