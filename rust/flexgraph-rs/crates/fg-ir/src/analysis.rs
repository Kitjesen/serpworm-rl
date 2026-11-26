//! Compile-time bipartite graph sparsity analysis, automatic Schur solver routing,
//! fill-reducing Minimum-Degree permutation, Liu's elimination tree, and exact symbolic
//! lower-triangular block-CSC pattern for general Sparse Block-$LDL^\top$ factorization.

use crate::schema::{IrError, RigidFlexGraph};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Automatic rigid-interface Schur complement solver route selected from the topology of $\mathcal{G}_R$.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchurRoute {
    /// Every connected component of free rigid bodies has size `<= 2`
    /// (e.g. SerpWorm with explicit inter-segment yaw -> batched 12x12 Cholesky/LU).
    BlockDiagonalPairs,
    /// Acyclic forest/tree with max component size `>= 3`
    /// (e.g. TendonEndoscope-3D, FlexSpine-Centipede -> $O(N_b)$ post-order Block-Tree $LDL^\top$).
    TreeChain,
    /// Cyclic rigid-interface graph (`cyclomatic_number > 0`)
    /// (e.g. Tensegrity-6Bar, closed-loop origami -> symbolic Sparse Block-$LDL^\top$).
    SparseGeneral,
}

/// Global node index mapping for a single beam edge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeamDofMap {
    pub beam_id: usize,
    /// Global node index of endpoint A (`0..num_bodies`).
    pub node_a: usize,
    /// Global node index of endpoint B (`Some(b)` if clamped-clamped, `None` if cantilever free tip).
    pub node_b: Option<usize>,
    /// Global node indices of the free beam nodes (`num_bodies..total_nodes`):
    /// - Clamped-clamped: `n_elements - 1` interior nodes.
    /// - Cantilever: `n_elements` free nodes (last entry is the free tip).
    pub free_node_ids: Vec<usize>,
    /// Full sequence of `n_elements + 1` global node indices along this beam.
    pub chain_node_ids: Vec<usize>,
}

/// Global flat node layout (`0..num_bodies` are rigid bodies, `num_bodies..total_nodes` are beam nodes).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeIndexing {
    pub num_bodies: usize,
    pub num_beam_nodes: usize,
    pub total_nodes: usize,
    pub beam_maps: Vec<BeamDofMap>,
}

/// Symbolic sparse block-CSC pattern of the strict lower triangle of $L$ in `P H P^T = L D L^T`,
/// where every block entry corresponds to a 6x6 spatial rigid-body block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseBlockPattern {
    /// Dimension in 6x6 blocks (`num_free_bodies`).
    pub n_blocks: usize,
    /// Permutation `perm[new_col] = free_body_slot` (`0..n_blocks`).
    pub perm: Vec<usize>,
    /// Inverse permutation `inv_perm[free_body_slot] = new_col`.
    pub inv_perm: Vec<usize>,
    /// Elimination tree parent in permuted coordinates (`etree[j] = Some(p)` with `p > j`, or `None` for root).
    pub etree: Vec<Option<usize>>,
    /// Column pointers into `row_idx` (length `n_blocks + 1`) for strict lower triangle of `L`.
    pub col_ptr: Vec<usize>,
    /// Sorted strict-lower row indices (`row > col`) for each column `col` of `L`.
    pub row_idx: Vec<usize>,
    /// Number of off-diagonal 6x6 nonzero blocks in `L`.
    pub nnz_offdiag_l: usize,
    /// Number of off-diagonal 6x6 nonzero blocks in the original matrix `A`.
    pub nnz_offdiag_a: usize,
    /// Exact fill-in count `nnz_offdiag_l - nnz_offdiag_a`.
    pub fill_in_blocks: usize,
}

/// Complete compile-time topological analysis of a `RigidFlexGraph`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTopologyAnalysis {
    pub route: SchurRoute,
    pub num_bodies: usize,
    pub num_free_bodies: usize,
    /// Mapping `free_slot -> original body_id`.
    pub free_body_ids: Vec<usize>,
    /// Mapping `body_id -> Some(free_slot)` or `None` if fixed Dirichlet base.
    pub body_to_free_slot: Vec<Option<usize>>,
    /// Connected components in `free_slot` coordinates.
    pub components: Vec<Vec<usize>>,
    /// Undirected edge count in $\mathcal{G}_R$ over free rigid bodies.
    pub num_rigid_edges: usize,
    /// Independent cycle count `E - V + C` of $\mathcal{G}_R$.
    pub cyclomatic_number: usize,
    /// Free-slot pairs `[u, v]` for `BlockDiagonalPairs` fast path.
    pub pair_blocks: Vec<[usize; 2]>,
    /// Isolated free-slot singletons (e.g., cantilever-only rigid body) for `BlockDiagonalPairs` fast path.
    pub single_blocks: Vec<usize>,
    /// Post-order elimination sequence over `free_slot` (leaves first, roots last) for `TreeChain`.
    pub tree_postorder: Vec<usize>,
    /// Parent in `free_slot` coordinates for `TreeChain` (`None` at component roots).
    pub tree_parent: Vec<Option<usize>>,
    /// Exact symbolic sparse block-$LDL^\top$ pattern (always populated for all routes).
    pub sparse_pattern: SparseBlockPattern,
    /// Global node indexing for state vectors `x[total_nodes]`, `v[total_nodes]`.
    pub indexing: NodeIndexing,
}

/// Compute global node indexing (`0..num_bodies` rigid bodies, followed by free beam nodes).
fn build_node_indexing(graph: &RigidFlexGraph) -> NodeIndexing {
    let num_bodies = graph.bodies.len();
    let mut next_node = num_bodies;
    let mut beam_maps = Vec::with_capacity(graph.beams.len());
    for beam in &graph.beams {
        let mut free_node_ids = Vec::with_capacity(beam.num_free_beam_nodes());
        let mut chain_node_ids = Vec::with_capacity(beam.n_elements + 1);
        chain_node_ids.push(beam.body_a);
        if let Some(bb) = beam.body_b {
            for _ in 1..beam.n_elements {
                free_node_ids.push(next_node);
                chain_node_ids.push(next_node);
                next_node += 1;
            }
            chain_node_ids.push(bb);
        } else {
            for _ in 1..=beam.n_elements {
                free_node_ids.push(next_node);
                chain_node_ids.push(next_node);
                next_node += 1;
            }
        }
        beam_maps.push(BeamDofMap {
            beam_id: beam.id,
            node_a: beam.body_a,
            node_b: beam.body_b,
            free_node_ids,
            chain_node_ids,
        });
    }
    NodeIndexing {
        num_bodies,
        num_beam_nodes: next_node - num_bodies,
        total_nodes: next_node,
        beam_maps,
    }
}

/// Exact Minimum-Degree fill-reducing elimination ordering on an undirected graph `adj`.
fn minimum_degree_ordering(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut elim = vec![false; n];
    let mut cur_adj: Vec<Vec<usize>> = adj.to_vec();
    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Pick uneliminated vertex with minimum current degree (tie-break by smallest index).
        let mut best = None;
        let mut best_deg = usize::MAX;
        for (u, nbrs) in cur_adj.iter().enumerate() {
            if elim[u] {
                continue;
            }
            let deg = nbrs.len();
            if deg < best_deg {
                best_deg = deg;
                best = Some(u);
            }
        }
        let pivot = best.expect("uneliminated vertex must exist");
        elim[pivot] = true;
        perm.push(pivot);

        // Form clique among uneliminated neighbors of `pivot`.
        let nbrs: Vec<usize> = cur_adj[pivot]
            .iter()
            .copied()
            .filter(|&v| !elim[v])
            .collect();
        for &u in &nbrs {
            cur_adj[u].retain(|&w| w != pivot);
        }
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                let u = nbrs[i];
                let v = nbrs[j];
                if !cur_adj[u].contains(&v) {
                    cur_adj[u].push(v);
                }
                if !cur_adj[v].contains(&u) {
                    cur_adj[v].push(u);
                }
            }
        }
    }
    perm
}

/// Build Liu's elimination tree `etree` on permuted graph `adj_perm` (Tim Davis, Ch. 4).
fn compute_etree(adj_perm: &[Vec<usize>]) -> Vec<Option<usize>> {
    let n = adj_perm.len();
    let mut parent = vec![None; n];
    let mut ancestor = vec![None; n];
    for (k, nbrs) in adj_perm.iter().enumerate() {
        for &i in nbrs {
            if i >= k {
                continue;
            }
            let mut node = i;
            while node != k {
                let next = ancestor[node];
                ancestor[node] = Some(k);
                match next {
                    None => {
                        parent[node] = Some(k);
                        break;
                    }
                    Some(p) => node = p,
                }
            }
        }
    }
    parent
}

/// DFS post-order of an elimination forest (`parent[i] = Some(p)` with `p > i`).
fn postorder_etree(parent: &[Option<usize>]) -> Vec<usize> {
    let n = parent.len();
    let mut children = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (u, &p) in parent.iter().enumerate() {
        match p {
            Some(par) => children[par].push(u),
            None => roots.push(u),
        }
    }
    let mut order = Vec::with_capacity(n);
    for root in roots {
        let mut stack: Vec<(usize, bool)> = vec![(root, false)];
        while let Some((u, expanded)) = stack.pop() {
            if expanded {
                order.push(u);
            } else {
                stack.push((u, true));
                // Push children in reverse so smallest-index child is visited first.
                for &c in children[u].iter().rev() {
                    stack.push((c, false));
                }
            }
        }
    }
    order
}

/// Build symbolic sparse block-$LDL^\top$ pattern given initial permutation `base_perm`.
fn build_sparse_pattern(adj: &[Vec<usize>], base_perm: Vec<usize>) -> SparseBlockPattern {
    let n = adj.len();
    if n == 0 {
        return SparseBlockPattern {
            n_blocks: 0,
            perm: Vec::new(),
            inv_perm: Vec::new(),
            etree: Vec::new(),
            col_ptr: vec![0],
            row_idx: Vec::new(),
            nnz_offdiag_l: 0,
            nnz_offdiag_a: 0,
            fill_in_blocks: 0,
        };
    }

    // Permute adjacency according to `base_perm` and compute elimination tree, then post-order etree.
    let permute_adj = |p: &[usize]| -> (Vec<usize>, Vec<Vec<usize>>) {
        let mut inv = vec![0; n];
        for (new_idx, &old_idx) in p.iter().enumerate() {
            inv[old_idx] = new_idx;
        }
        let mut ap = vec![Vec::new(); n];
        for (old_u, nbrs) in adj.iter().enumerate() {
            let nu = inv[old_u];
            for &old_v in nbrs {
                ap[nu].push(inv[old_v]);
            }
            ap[nu].sort_unstable();
            ap[nu].dedup();
        }
        (inv, ap)
    };

    let (_, adj_base) = permute_adj(&base_perm);
    let etree_base = compute_etree(&adj_base);
    let post = postorder_etree(&etree_base);

    // Refine permutation by post-ordering the elimination tree (preserves fill-in, improves locality).
    let perm: Vec<usize> = post.iter().map(|&idx| base_perm[idx]).collect();
    let (inv_perm, adj_perm) = permute_adj(&perm);
    let etree = compute_etree(&adj_perm);

    // Exact symbolic lower-triangular pattern of L via Tim Davis's `ereach`:
    // For each row `k`, walk from each `i < k` in `adj_perm[k]` up `etree` to `k`.
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut mark = vec![usize::MAX; n];
    let mut nnz_offdiag_a = 0usize;

    for (k, nbrs) in adj_perm.iter().enumerate() {
        mark[k] = k;
        for &i in nbrs {
            if i >= k {
                continue;
            }
            nnz_offdiag_a += 1;
            let mut node = i;
            while node < k && mark[node] != k {
                mark[node] = k;
                col_rows[node].push(k);
                match etree[node] {
                    Some(p) => node = p,
                    None => break,
                }
            }
        }
    }

    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for rows in &col_rows {
        row_idx.extend_from_slice(rows);
        col_ptr.push(row_idx.len());
    }
    let nnz_offdiag_l = row_idx.len();
    let fill_in_blocks = nnz_offdiag_l.saturating_sub(nnz_offdiag_a);

    SparseBlockPattern {
        n_blocks: n,
        perm,
        inv_perm,
        etree,
        col_ptr,
        row_idx,
        nnz_offdiag_l,
        nnz_offdiag_a,
        fill_in_blocks,
    }
}

/// Analyze a `RigidFlexGraph` and generate the compile-time Schur execution plan.
pub fn analyze_graph(graph: &RigidFlexGraph) -> Result<GraphTopologyAnalysis, IrError> {
    graph.validate()?;

    let num_bodies = graph.bodies.len();
    let mut free_body_ids = Vec::new();
    let mut body_to_free_slot = vec![None; num_bodies];
    for (id, body) in graph.bodies.iter().enumerate() {
        if !body.is_fixed {
            let slot = free_body_ids.len();
            free_body_ids.push(id);
            body_to_free_slot[id] = Some(slot);
        }
    }
    let n_free = free_body_ids.len();

    // Build undirected adjacency graph G_R over free rigid bodies.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_free];
    let mut add_edge = |ba: usize, bb: usize| {
        if let (Some(u), Some(v)) = (body_to_free_slot[ba], body_to_free_slot[bb]) {
            if u != v {
                if !adj[u].contains(&v) {
                    adj[u].push(v);
                }
                if !adj[v].contains(&u) {
                    adj[v].push(u);
                }
            }
        }
    };

    for beam in &graph.beams {
        if let Some(bb) = beam.body_b {
            add_edge(beam.body_a, bb);
        }
    }
    for joint in &graph.joints {
        if joint.implicit_coupling {
            add_edge(joint.body_a, joint.body_b);
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
    }

    let num_rigid_edges: usize = adj.iter().map(Vec::len).sum::<usize>() / 2;

    // Connected components + BFS tree orientation (leaves-first post-order for trees).
    let mut visited = vec![false; n_free];
    let mut components = Vec::new();
    let mut tree_parent = vec![None; n_free];
    let mut tree_postorder = Vec::with_capacity(n_free);

    for start in 0..n_free {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            comp.push(u);
            for &v in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    tree_parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        // Reverse BFS order guarantees every child precedes its parent (valid post-order!).
        for &u in comp.iter().rev() {
            tree_postorder.push(u);
        }
        comp.sort_unstable();
        components.push(comp);
    }

    let num_components = components.len();
    let cyclomatic_number = (num_rigid_edges + num_components).saturating_sub(n_free);
    let max_comp_size = components.iter().map(Vec::len).max().unwrap_or(0);

    let detected_route = if n_free == 0 || (cyclomatic_number == 0 && max_comp_size <= 2) {
        SchurRoute::BlockDiagonalPairs
    } else if cyclomatic_number == 0 {
        SchurRoute::TreeChain
    } else {
        SchurRoute::SparseGeneral
    };
    let route = graph.solver_hints.force_route.unwrap_or(detected_route);

    let mut pair_blocks = Vec::new();
    let mut single_blocks = Vec::new();
    for comp in &components {
        match comp.len() {
            1 => single_blocks.push(comp[0]),
            2 => pair_blocks.push([comp[0], comp[1]]),
            _ => {}
        }
    }

    // Choose fill-reducing base permutation:
    // - For trees/pairs: post-order eliminates with ZERO fill-in!
    // - For cyclic graphs: Minimum-Degree heuristic minimizes fill-in!
    let base_perm = if cyclomatic_number == 0 {
        tree_postorder.clone()
    } else {
        minimum_degree_ordering(&adj)
    };
    let sparse_pattern = build_sparse_pattern(&adj, base_perm);
    let indexing = build_node_indexing(graph);

    Ok(GraphTopologyAnalysis {
        route,
        num_bodies,
        num_free_bodies: n_free,
        free_body_ids,
        body_to_free_slot,
        components,
        num_rigid_edges,
        cyclomatic_number,
        pair_blocks,
        single_blocks,
        tree_postorder,
        tree_parent,
        sparse_pattern,
        indexing,
    })
}
