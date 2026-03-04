use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nvidia::ComputeCapability;
use crate::scan::DepGraph;

/// Threshold for collapsing homogeneous node groups into clusters.
const COLLAPSE_THRESHOLD: usize = 25;

/// Pre-computed, renderer-agnostic view of the dependency graph.
///
/// All the O(N+E) visibility, bridging, clustering, and host-dep work is done
/// in Rust so the browser only maps this to Cytoscape elements.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphView {
    /// Individual (non-clustered) visible nodes.
    pub nodes: Vec<ViewNode>,
    /// Collapsed clusters of homogeneous nodes.
    pub clusters: Vec<ViewCluster>,
    /// Edges between visible elements (node-to-node, node-to-cluster, cluster-to-cluster).
    pub edges: Vec<ViewEdge>,
    /// Synthetic host dependency nodes (unresolved .so from host).
    pub host_nodes: Vec<HostNode>,
    /// Edges from host nodes to visible elements.
    pub host_edges: Vec<ViewEdge>,
    /// Mapping from soname to cluster ID (for expansion lookups).
    pub node_to_cluster: HashMap<String, String>,
    /// All SM architecture values present in the graph, sorted ascending.
    pub sm_values: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewNode {
    pub id: String,
    pub role: NodeRole,
    pub priority: crate::image::BinaryPriority,
    pub path: String,
    pub cubins: Vec<ComputeCapability>,
    pub ptx: Vec<ComputeCapability>,
    pub size: u64,
    pub deps: Vec<String>,
    pub unresolved: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    Cuda,
    Bridge,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewCluster {
    pub id: String,
    pub dir_key: String,
    pub label: String,
    pub members: Vec<String>,
    pub cubin_vals: Vec<u32>,
    pub ptx_vals: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HostNode {
    pub id: String,
    pub soname: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ViewEdge {
    pub source: String,
    pub target: String,
}

impl GraphView {
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            clusters: Vec::new(),
            edges: Vec::new(),
            host_nodes: Vec::new(),
            host_edges: Vec::new(),
            node_to_cluster: HashMap::new(),
            sm_values: Vec::new(),
        }
    }
}

/// Convert a ComputeCapability to its SM integer (e.g. 7.0 -> 70).
fn sm_value(cc: &ComputeCapability) -> u32 {
    cc.major * 10 + cc.minor
}

/// Build a sorted, deduplicated CC key string for grouping (e.g. "70,80+90").
fn cc_key(cubins: &[ComputeCapability], ptx: &[ComputeCapability]) -> String {
    let mut c: Vec<u32> = cubins.iter().map(sm_value).collect();
    c.sort_unstable();
    let c_str: Vec<String> = c.iter().map(|v| v.to_string()).collect();

    let mut p: Vec<u32> = ptx.iter().map(sm_value).collect();
    p.sort_unstable();

    if p.is_empty() {
        c_str.join(",")
    } else {
        let p_str: Vec<String> = p.iter().map(|v| v.to_string()).collect();
        format!("{}+{}", c_str.join(","), p_str.join(","))
    }
}

/// Extract the directory grouping key from a binary path.
///
/// For Python packages (site-packages/foo or dist-packages/foo), returns the
/// top-level package name. Otherwise returns the parent directory name.
fn dir_key(path: &str) -> &str {
    let parts: Vec<&str> = path.split('/').collect();

    // Look for site-packages or dist-packages
    let sp_idx = parts
        .iter()
        .position(|&p| p == "site-packages" || p == "dist-packages");

    if let Some(idx) = sp_idx {
        if idx + 1 < parts.len() {
            return parts[idx + 1];
        }
    }

    // Fall back to parent directory
    if parts.len() >= 2 {
        let parent = parts[parts.len() - 2];
        if parent.is_empty() {
            return "/";
        }
        return parent;
    }

    "/"
}

/// Identify all visible nodes and their roles (CUDA or bridge).
///
/// A node is visible if:
/// 1. It has CUDA content (cubins or ptx), OR
/// 2. It's a "bridge": a non-CUDA node that is depended on by at least one CUDA
///    node AND itself depends on at least one CUDA node.
///
/// Uses a reverse-dep index for O(N+E) bridge detection.
fn find_visible_nodes(graph: &DepGraph) -> HashMap<String, NodeRole> {
    let mut visible = HashMap::new();

    // All CUDA nodes are visible
    for (soname, node) in &graph.nodes {
        if node.has_cuda() {
            visible.insert(soname.clone(), NodeRole::Cuda);
        }
    }

    // Build reverse dependency index: dep -> [dependants]
    let mut reverse_deps: HashMap<&str, Vec<&str>> = HashMap::new();
    for (soname, node) in &graph.nodes {
        for dep in &node.deps {
            reverse_deps
                .entry(dep.as_str())
                .or_default()
                .push(soname.as_str());
        }
    }

    // Bridge detection: non-CUDA node depended on by a CUDA node AND depends on a CUDA node
    for (soname, node) in &graph.nodes {
        if node.has_cuda() {
            continue;
        }

        // Does this node depend on any CUDA node?
        let depends_on_cuda = node
            .deps
            .iter()
            .any(|d| graph.nodes.get(d).is_some_and(|n| n.has_cuda()));

        if !depends_on_cuda {
            continue;
        }

        // Is this node depended on by any CUDA node?
        let depended_by_cuda = reverse_deps.get(soname.as_str()).is_some_and(|dependants| {
            dependants
                .iter()
                .any(|d| graph.nodes.get(*d).is_some_and(|n| n.has_cuda()))
        });

        if depended_by_cuda {
            visible.insert(soname.clone(), NodeRole::Bridge);
        }
    }

    visible
}

/// Group visible nodes by (dir_key, cc_key) and collapse large groups into clusters.
///
/// Returns:
/// - individual nodes (sonames of nodes that remain unclustered)
/// - clusters
/// - node-to-cluster mapping
fn build_clusters(
    visible: &HashMap<String, NodeRole>,
    graph: &DepGraph,
) -> (HashSet<String>, Vec<ViewCluster>, HashMap<String, String>) {
    // Group by (dir_key, cc_key)
    let mut group_map: HashMap<String, Vec<String>> = HashMap::new();
    for soname in visible.keys() {
        let node = match graph.nodes.get(soname) {
            Some(n) => n,
            None => continue,
        };
        let gk = format!(
            "{}|{}",
            dir_key(&node.path),
            cc_key(&node.cubins, &node.ptx)
        );
        group_map.entry(gk).or_default().push(soname.clone());
    }

    let mut individual_nodes = HashSet::new();
    let mut clusters = Vec::new();
    let mut node_to_cluster = HashMap::new();

    for (gk, members) in &group_map {
        if members.len() > COLLAPSE_THRESHOLD {
            let dir = gk.split('|').next().unwrap_or("");
            let cluster_id = format!(
                "cluster_{}",
                gk.chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect::<String>()
            );

            // Sample CC values from the first member (they're homogeneous)
            let sample = members.first().and_then(|s| graph.nodes.get(s.as_str()));

            let (cubin_vals, ptx_vals) = match sample {
                Some(node) => {
                    let mut cv: Vec<u32> = node.cubins.iter().map(sm_value).collect();
                    cv.sort_unstable();
                    let mut pv: Vec<u32> = node.ptx.iter().map(sm_value).collect();
                    pv.sort_unstable();
                    (cv, pv)
                }
                None => (Vec::new(), Vec::new()),
            };

            let label = format!("{} in {}", members.len(), dir);

            for m in members {
                node_to_cluster.insert(m.clone(), cluster_id.clone());
            }

            clusters.push(ViewCluster {
                id: cluster_id,
                dir_key: dir.to_string(),
                label,
                members: members.clone(),
                cubin_vals,
                ptx_vals,
            });
        } else {
            for m in members {
                individual_nodes.insert(m.clone());
            }
        }
    }

    (individual_nodes, clusters, node_to_cluster)
}

/// Build deduplicated edges between visible elements.
///
/// Handles node-to-node, node-to-cluster, cluster-to-node, and cluster-to-cluster edges.
/// Edge direction: dependency -> dependant (arrow points from dep to the node that needs it).
fn build_edges(
    individual_nodes: &HashSet<String>,
    node_to_cluster: &HashMap<String, String>,
    graph: &DepGraph,
) -> Vec<ViewEdge> {
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    let mut edges = Vec::new();

    let mut add_edge = |src: String, tgt: String| {
        let key = (src.clone(), tgt.clone());
        if edge_set.insert(key) {
            edges.push(ViewEdge {
                source: src,
                target: tgt,
            });
        }
    };

    // Edges between individual nodes
    for soname in individual_nodes {
        let node = match graph.nodes.get(soname) {
            Some(n) => n,
            None => continue,
        };
        for dep in &node.deps {
            if individual_nodes.contains(dep) {
                add_edge(dep.clone(), soname.clone());
            }
        }
    }

    // Edges involving clusters (cluster members -> deps)
    for (cluster_id, members) in cluster_members(node_to_cluster) {
        let mut seen_deps = HashSet::new();
        for member in &members {
            let node = match graph.nodes.get(member.as_str()) {
                Some(n) => n,
                None => continue,
            };
            for dep in &node.deps {
                if !seen_deps.insert(dep.clone()) {
                    continue;
                }
                if individual_nodes.contains(dep) {
                    add_edge(dep.clone(), cluster_id.clone());
                } else if let Some(dep_cluster) = node_to_cluster.get(dep) {
                    if dep_cluster != &cluster_id {
                        add_edge(dep_cluster.clone(), cluster_id.clone());
                    }
                }
            }
        }
    }

    // Edges from individual nodes to clusters they depend on
    for soname in individual_nodes {
        let node = match graph.nodes.get(soname) {
            Some(n) => n,
            None => continue,
        };
        for dep in &node.deps {
            if let Some(cluster_id) = node_to_cluster.get(dep) {
                add_edge(cluster_id.clone(), soname.clone());
            }
        }
    }

    edges
}

/// Invert node_to_cluster into cluster_id -> [members].
fn cluster_members(node_to_cluster: &HashMap<String, String>) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    for (soname, cluster_id) in node_to_cluster {
        result
            .entry(cluster_id.clone())
            .or_default()
            .push(soname.clone());
    }
    result
}

/// Collect all unresolved sonames from the entire graph and build host nodes + edges.
///
/// For each visible element (individual node or cluster), we BFS through its transitive
/// deps to find which unresolved host libs it ultimately needs, then create edges.
fn build_host_view(
    graph: &DepGraph,
    individual_nodes: &HashSet<String>,
    node_to_cluster: &HashMap<String, String>,
) -> (Vec<HostNode>, Vec<ViewEdge>) {
    // Collect ALL unresolved sonames from every node in the graph
    let mut all_unresolved: BTreeSet<String> = BTreeSet::new();
    for node in graph.nodes.values() {
        for u in &node.unresolved {
            all_unresolved.insert(u.clone());
        }
    }

    let host_nodes: Vec<HostNode> = all_unresolved
        .iter()
        .map(|soname| HostNode {
            id: format!("host_{soname}"),
            soname: soname.clone(),
        })
        .collect();

    // BFS to find transitive unresolved deps for a starting node
    let collect_transitive_unresolved = |start: &str| -> HashSet<String> {
        let mut unresolved = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(cur) = queue.pop_front() {
            if !visited.insert(cur.clone()) {
                continue;
            }
            if let Some(node) = graph.nodes.get(&cur) {
                for u in &node.unresolved {
                    unresolved.insert(u.clone());
                }
                for dep in &node.deps {
                    queue.push_back(dep.clone());
                }
            }
        }

        unresolved
    };

    let mut host_edge_set: HashSet<(String, String)> = HashSet::new();
    let mut host_edges = Vec::new();

    let mut add_host_edge = |src: String, tgt: String| {
        let key = (src.clone(), tgt.clone());
        if host_edge_set.insert(key) {
            host_edges.push(ViewEdge {
                source: src,
                target: tgt,
            });
        }
    };

    // Host edges from individual nodes
    for soname in individual_nodes {
        for u in collect_transitive_unresolved(soname) {
            add_host_edge(format!("host_{u}"), soname.clone());
        }
    }

    // Host edges from clusters
    let clusters = cluster_members(node_to_cluster);
    for (cluster_id, members) in &clusters {
        let mut cluster_unresolved = HashSet::new();
        for member in members {
            for u in collect_transitive_unresolved(member) {
                cluster_unresolved.insert(u);
            }
        }
        for u in cluster_unresolved {
            add_host_edge(format!("host_{u}"), cluster_id.clone());
        }
    }

    (host_nodes, host_edges)
}

/// Collect all SM values present in the graph, sorted ascending.
fn collect_sm_values(graph: &DepGraph) -> Vec<u32> {
    let mut sm_set: BTreeSet<u32> = BTreeSet::new();
    for node in graph.nodes.values() {
        for c in &node.cubins {
            sm_set.insert(sm_value(c));
        }
        for p in &node.ptx {
            sm_set.insert(sm_value(p));
        }
    }
    sm_set.into_iter().collect()
}

/// Build the complete pre-computed graph view from a dependency graph.
pub fn build_graph_view(graph: &DepGraph) -> GraphView {
    let visible = find_visible_nodes(graph);
    let (individual_nodes, clusters, node_to_cluster) = build_clusters(&visible, graph);

    let view_nodes: Vec<ViewNode> = individual_nodes
        .iter()
        .filter_map(|soname| {
            let node = graph.nodes.get(soname)?;
            let role = visible.get(soname).copied().unwrap_or(NodeRole::Cuda);
            Some(ViewNode {
                id: soname.clone(),
                role,
                priority: node.priority,
                path: node.path.clone(),
                cubins: node.cubins.clone(),
                ptx: node.ptx.clone(),
                size: node.size,
                deps: node.deps.clone(),
                unresolved: node.unresolved.clone(),
                package: node.package.clone(),
            })
        })
        .collect();

    let edges = build_edges(&individual_nodes, &node_to_cluster, graph);
    let (host_nodes, host_edges) = build_host_view(graph, &individual_nodes, &node_to_cluster);
    let sm_values = collect_sm_values(graph);

    GraphView {
        nodes: view_nodes,
        clusters,
        edges,
        host_nodes,
        host_edges,
        node_to_cluster,
        sm_values,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::image::BinaryPriority;
    use crate::nvidia::ComputeCapability;
    use crate::scan::{DepGraph, DepNode};

    use super::*;

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
    }

    fn make_node(
        soname: &str,
        path: &str,
        cubins: Vec<ComputeCapability>,
        deps: Vec<&str>,
    ) -> (String, DepNode) {
        (
            soname.to_string(),
            DepNode {
                path: path.to_string(),
                soname: soname.to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 1024,
                cubins,
                ptx: vec![],
                deps: deps.iter().map(|s| s.to_string()).collect(),
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        )
    }

    fn make_node_with_unresolved(
        soname: &str,
        path: &str,
        cubins: Vec<ComputeCapability>,
        deps: Vec<&str>,
        unresolved: Vec<&str>,
    ) -> (String, DepNode) {
        let (name, mut node) = make_node(soname, path, cubins, deps);
        node.unresolved = unresolved.iter().map(|s| s.to_string()).collect();
        (name, node)
    }

    #[test]
    fn cuda_nodes_are_visible() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "libcuda.so".to_string(),
            DepNode {
                path: "/lib/libcuda.so".to_string(),
                soname: "libcuda.so".to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 1024,
                cubins: vec![cc(7, 0)],
                ptx: vec![],
                deps: vec![],
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        );
        let graph = DepGraph {
            roots: vec!["libcuda.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible.get("libcuda.so"), Some(&NodeRole::Cuda));
    }

    #[test]
    fn non_cuda_nodes_without_bridge_role_are_hidden() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "libplain.so".to_string(),
            DepNode {
                path: "/lib/libplain.so".to_string(),
                soname: "libplain.so".to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 512,
                cubins: vec![],
                ptx: vec![],
                deps: vec![],
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        );
        let graph = DepGraph {
            roots: vec!["libplain.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        assert!(visible.is_empty());
    }

    #[test]
    fn bridge_detection() {
        // A -> bridge -> B, where A and B have CUDA, bridge does not
        let mut nodes = HashMap::new();
        let (k, a) = make_node("a.so", "/lib/a.so", vec![cc(7, 0)], vec!["bridge.so"]);
        nodes.insert(k, a);

        let (k, bridge) = make_node("bridge.so", "/lib/bridge.so", vec![], vec!["b.so"]);
        nodes.insert(k, bridge);

        let (k, b) = make_node("b.so", "/lib/b.so", vec![cc(8, 0)], vec![]);
        nodes.insert(k, b);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        assert_eq!(visible.len(), 3);
        assert_eq!(visible.get("a.so"), Some(&NodeRole::Cuda));
        assert_eq!(visible.get("bridge.so"), Some(&NodeRole::Bridge));
        assert_eq!(visible.get("b.so"), Some(&NodeRole::Cuda));
    }

    #[test]
    fn non_bridge_hidden() {
        // A (cuda) -> plain (no cuda) -> nothing
        // plain depends on no cuda node, so not a bridge
        let mut nodes = HashMap::new();
        let (k, a) = make_node("a.so", "/lib/a.so", vec![cc(7, 0)], vec!["plain.so"]);
        nodes.insert(k, a);

        let (k, plain) = make_node("plain.so", "/lib/plain.so", vec![], vec![]);
        nodes.insert(k, plain);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        assert_eq!(visible.len(), 1);
        assert!(visible.contains_key("a.so"));
        assert!(!visible.contains_key("plain.so"));
    }

    #[test]
    fn clustering_groups_large_homogeneous_sets() {
        let mut nodes = HashMap::new();
        // Create 30 nodes with same CC profile and same directory -> should cluster
        for i in 0..30 {
            let soname = format!("lib{i}.so");
            let (k, n) = make_node(
                &soname,
                &format!("/opt/site-packages/mylib/lib{i}.so"),
                vec![cc(7, 0), cc(8, 0)],
                vec![],
            );
            nodes.insert(k, n);
        }

        let graph = DepGraph {
            roots: vec![],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        let (individual, clusters, node_to_cluster) = build_clusters(&visible, &graph);

        assert!(individual.is_empty(), "all 30 should be clustered");
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 30);
        assert_eq!(node_to_cluster.len(), 30);
    }

    #[test]
    fn small_groups_stay_individual() {
        let mut nodes = HashMap::new();
        // Create 5 nodes (below threshold) -> should stay individual
        for i in 0..5 {
            let soname = format!("lib{i}.so");
            let (k, n) = make_node(
                &soname,
                &format!("/opt/site-packages/mylib/lib{i}.so"),
                vec![cc(7, 0)],
                vec![],
            );
            nodes.insert(k, n);
        }

        let graph = DepGraph {
            roots: vec![],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        let (individual, clusters, node_to_cluster) = build_clusters(&visible, &graph);

        assert_eq!(individual.len(), 5);
        assert!(clusters.is_empty());
        assert!(node_to_cluster.is_empty());
    }

    #[test]
    fn edge_deduplication() {
        let mut nodes = HashMap::new();
        let (k, a) = make_node("a.so", "/lib/a.so", vec![cc(7, 0)], vec!["b.so"]);
        nodes.insert(k, a);
        let (k, b) = make_node("b.so", "/lib/b.so", vec![cc(8, 0)], vec![]);
        nodes.insert(k, b);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        let (individual, _, node_to_cluster) = build_clusters(&visible, &graph);
        let edges = build_edges(&individual, &node_to_cluster, &graph);

        // Should have exactly one edge: b.so -> a.so
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, "b.so");
        assert_eq!(edges[0].target, "a.so");
    }

    #[test]
    fn host_nodes_collect_all_unresolved() {
        let mut nodes = HashMap::new();
        let (k, n) = make_node_with_unresolved(
            "a.so",
            "/lib/a.so",
            vec![cc(7, 0)],
            vec!["b.so"],
            vec!["libhost1.so"],
        );
        nodes.insert(k, n);
        let (k, n) = make_node_with_unresolved(
            "b.so",
            "/lib/b.so",
            vec![cc(8, 0)],
            vec![],
            vec!["libhost2.so"],
        );
        nodes.insert(k, n);
        // A non-visible node with unresolved deps
        let (k, n) = make_node_with_unresolved(
            "hidden.so",
            "/lib/hidden.so",
            vec![],
            vec![],
            vec!["libhost3.so"],
        );
        nodes.insert(k, n);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        let (individual, _, node_to_cluster) = build_clusters(&visible, &graph);
        let (host_nodes, _) = build_host_view(&graph, &individual, &node_to_cluster);

        let host_sonames: HashSet<&str> = host_nodes.iter().map(|h| h.soname.as_str()).collect();
        assert!(host_sonames.contains("libhost1.so"));
        assert!(host_sonames.contains("libhost2.so"));
        assert!(host_sonames.contains("libhost3.so"));
    }

    #[test]
    fn transitive_unresolved_bfs() {
        // a.so -> b.so (unresolved: host.so)
        // a.so should transitively pick up host.so
        let mut nodes = HashMap::new();
        let (k, n) = make_node("a.so", "/lib/a.so", vec![cc(7, 0)], vec!["b.so"]);
        nodes.insert(k, n);
        let (k, n) =
            make_node_with_unresolved("b.so", "/lib/b.so", vec![cc(8, 0)], vec![], vec!["host.so"]);
        nodes.insert(k, n);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        let (individual, _, node_to_cluster) = build_clusters(&visible, &graph);
        let (_, host_edges) = build_host_view(&graph, &individual, &node_to_cluster);

        // a.so should have an edge from host_host.so
        let a_host_edges: Vec<&ViewEdge> = host_edges
            .iter()
            .filter(|e| e.target == "a.so" && e.source == "host_host.so")
            .collect();
        assert!(
            !a_host_edges.is_empty(),
            "a.so should transitively depend on host.so"
        );
    }

    #[test]
    fn empty_graph_view() {
        let gv = GraphView::empty();
        assert!(gv.nodes.is_empty());
        assert!(gv.clusters.is_empty());
        assert!(gv.edges.is_empty());
        assert!(gv.host_nodes.is_empty());
        assert!(gv.host_edges.is_empty());
        assert!(gv.node_to_cluster.is_empty());
        assert!(gv.sm_values.is_empty());
    }

    #[test]
    fn build_graph_view_empty_graph() {
        let graph = DepGraph {
            roots: vec![],
            nodes: HashMap::new(),
        };
        let gv = build_graph_view(&graph);
        assert!(gv.nodes.is_empty());
        assert!(gv.clusters.is_empty());
        assert!(gv.edges.is_empty());
    }

    #[test]
    fn roundtrip_serde() {
        let mut nodes = HashMap::new();
        let (k, n) = make_node("a.so", "/lib/a.so", vec![cc(7, 0)], vec![]);
        nodes.insert(k, n);

        let graph = DepGraph {
            roots: vec!["a.so".to_string()],
            nodes,
        };

        let gv = build_graph_view(&graph);
        let json = serde_json::to_string(&gv).expect("serialize");
        let gv2: GraphView = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(gv.nodes.len(), gv2.nodes.len());
        assert_eq!(gv.clusters.len(), gv2.clusters.len());
        assert_eq!(gv.edges.len(), gv2.edges.len());
        assert_eq!(gv.host_nodes.len(), gv2.host_nodes.len());
        assert_eq!(gv.sm_values, gv2.sm_values);
    }

    #[test]
    fn sm_values_collected_and_sorted() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "a.so".to_string(),
            DepNode {
                path: "/lib/a.so".to_string(),
                soname: "a.so".to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 100,
                cubins: vec![cc(9, 0), cc(7, 0)],
                ptx: vec![cc(10, 0)],
                deps: vec![],
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        );
        nodes.insert(
            "b.so".to_string(),
            DepNode {
                path: "/lib/b.so".to_string(),
                soname: "b.so".to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 100,
                cubins: vec![cc(8, 0)],
                ptx: vec![],
                deps: vec![],
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        );

        let graph = DepGraph {
            roots: vec![],
            nodes,
        };

        let sm = collect_sm_values(&graph);
        assert_eq!(sm, vec![70, 80, 90, 100]);
    }

    #[test]
    fn dir_key_python_packages() {
        assert_eq!(dir_key("/opt/site-packages/torch/lib/foo.so"), "torch");
        assert_eq!(
            dir_key("/usr/lib/python3/dist-packages/numpy/core/foo.so"),
            "numpy"
        );
    }

    #[test]
    fn dir_key_regular_paths() {
        assert_eq!(dir_key("/usr/lib/foo.so"), "lib");
        assert_eq!(dir_key("/foo.so"), "/");
    }

    #[test]
    fn cc_key_formatting() {
        assert_eq!(cc_key(&[cc(7, 0), cc(8, 0)], &[]), "70,80");
        assert_eq!(cc_key(&[cc(7, 0)], &[cc(9, 0)]), "70+90");
        assert_eq!(cc_key(&[], &[]), "");
    }

    #[test]
    fn load_real_scan_and_verify_clusters() {
        let path = std::path::Path::new("wiimi-scan-docker.io_vllm_vllm-openai_v0.16.0.json");
        if !path.exists() {
            // Skip if the scan file isn't present (CI, etc.)
            return;
        }
        let content = std::fs::read_to_string(path).unwrap();
        let result: crate::scan::ScanResult = serde_json::from_str(&content).unwrap();
        let graph = result.dep_graph.as_ref().unwrap();
        let gv = build_graph_view(graph);

        // vLLM v0.16.0 has ~770 visible CUDA nodes, mostly in flashinfer_jit_cache
        // With 384 + 254 + 59 nodes sharing dir+cc profiles, we expect clusters
        assert!(
            !gv.clusters.is_empty(),
            "expected clusters for vLLM image, got 0 (nodes: {}, all in graph: {})",
            gv.nodes.len(),
            graph.nodes.len()
        );

        let total_clustered: usize = gv.clusters.iter().map(|c| c.members.len()).sum();
        assert!(
            total_clustered > 100,
            "expected >100 clustered nodes, got {total_clustered}"
        );

        // Individual (unclustered) nodes should be much fewer than total visible
        assert!(
            gv.nodes.len() < 200,
            "expected <200 individual nodes after clustering, got {}",
            gv.nodes.len()
        );

        eprintln!(
            "vLLM graph_view: {} individual, {} clusters ({} clustered), {} edges",
            gv.nodes.len(),
            gv.clusters.len(),
            total_clustered,
            gv.edges.len()
        );
    }

    #[test]
    fn dir_key_dist_packages() {
        assert_eq!(
            dir_key("/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/foo/foo.so"),
            "flashinfer_jit_cache"
        );
    }

    #[test]
    fn clustering_dist_packages_large_group() {
        // Simulate 30 flashinfer JIT .so files under dist-packages with same CC
        let mut nodes = HashMap::new();
        for i in 0..30 {
            let soname = format!("batch_op_{i}_sm90.so");
            let path = format!(
                "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/batch_op_{i}_sm90/batch_op_{i}_sm90.so"
            );
            let (k, n) = make_node(&soname, &path, vec![cc(9, 0), cc(10, 0), cc(12, 0)], vec![]);
            nodes.insert(k, n);
        }

        let graph = DepGraph {
            roots: vec![],
            nodes,
        };

        let visible = find_visible_nodes(&graph);
        assert_eq!(visible.len(), 30);

        let (individual, clusters, node_to_cluster) = build_clusters(&visible, &graph);
        assert!(
            individual.is_empty(),
            "all 30 dist-packages nodes should be clustered, got {} individual",
            individual.len()
        );
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 30);
        assert_eq!(clusters[0].dir_key, "flashinfer_jit_cache");
        assert_eq!(node_to_cluster.len(), 30);
    }
}
