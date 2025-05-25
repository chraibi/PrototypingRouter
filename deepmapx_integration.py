import math
import networkx as nx
from typing import Tuple


def depthmapx_dvalue(k: int) -> float:
    """Return depthmapX diamond normalization value."""
    if k <= 2:
        raise ValueError("dvalue is undefined for k <= 2")
    return (
        2.0 * (k * (math.log2((k + 2.0) / 3.0) - 1.0) + 1.0) / ((k - 1.0) * (k - 2.0))
    )


def calculate_depthmapx_integration(
    graph: nx.Graph, node_id: int
) -> Tuple[float, float, float, float]:
    """Calculate Integration [HH] matching depthmapX's algorithm.

    Returns a tuple of (integration_hh, mean_depth, ra, rra_d).
    Values are -1.0 when integration cannot be computed (e.g. small graphs).
    """
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return -1.0, -1.0, -1.0, -1.0

    lengths = nx.single_source_shortest_path_length(graph, node_id)
    total_depth = sum(dist for target, dist in lengths.items() if target != node_id)
    mean_depth = total_depth / (node_count - 1)

    if node_count > 2 and mean_depth > 1.0:
        ra = 2.0 * (mean_depth - 1.0) / (node_count - 2)
        rra_d = ra / depthmapx_dvalue(node_count)
        integration_hh = 1.0 / rra_d
    else:
        ra = -1.0
        rra_d = -1.0
        integration_hh = -1.0

    return integration_hh, mean_depth, ra, rra_d
