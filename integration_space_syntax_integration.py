import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional

def calculate_space_syntax_integration(graph: nx.Graph, node_id: int) -> Tuple[float, float, float, float]:
    """
    Calculate canonical Space Syntax integration measures for a given node.
    Based on Hillier & Hanson methodology: Integration = 1/RRA
    
    Returns:
        integration_hh: Integration [HH] = 1/RRA_D (diamond normalization)
        mean_depth: Average topological distance to all other nodes  
        relative_asymmetry: RA = 2(MD-1)/(k-2) for axial analysis
        real_relative_asymmetry: RRA_D = RA/D (diamond normalization)
    """
    k = len(graph.nodes())
    
    if k <= 2:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        # Calculate shortest path lengths to all other nodes
        shortest_paths = nx.single_source_shortest_path_length(graph, node_id)
        
        # Remove self-distance (which is 0)
        distances = [dist for target, dist in shortest_paths.items() if target != node_id]
        
        if not distances:
            return 0.0, 0.0, 0.0, 0.0
        
        # Mean Depth (MD) = Σd(x,i) / (|V|-1)
        mean_depth = sum(distances) / len(distances)
        
        # Relative Asymmetry (RA) = 2(MD-1) / (k-2) 
        # This is the correct RA formula for axial analysis from the sources
        relative_asymmetry = 2 * (mean_depth - 1) / (k - 2) if k > 2 else 0
        
        # Real Relative Asymmetry (RRA) with Diamond normalization
        # RRA_D = RA / D_k where D_k is the diamond value for k nodes
        diamond_value = calculate_diamond_value(k)
        real_relative_asymmetry = relative_asymmetry / diamond_value if diamond_value > 0 else 0
        
        # Integration [HH] = 1/RRA_D (canonical Space Syntax integration)
        integration_hh = 1 / real_relative_asymmetry if real_relative_asymmetry > 0 else float('inf')
        
        return integration_hh, mean_depth, relative_asymmetry, real_relative_asymmetry
        
    except Exception as e:
        return 0.0, float('inf'), 0.0, 0.0

def calculate_diamond_value(k: int) -> float:
    """
    Calculate Diamond's normalization value for a k-node system.
    This represents the RA value for the most integrated node in a diamond-shaped graph.
    """
    if k < 3:
        return 1.0
    
    if k % 2 == 1:  # odd number of nodes
        return 2 * (k - 1) / (3 * (k - 2))
    else:  # even number of nodes
        return 2 * (k - 1) * (k - 2) / (3 * (k - 2) * (k - 2))

def calculate_angular_integration(graph: nx.Graph, node_id: int, 
                                angular_weights: Optional[Dict] = None) -> float:
    """
    Calculate angular integration considering direction changes.
    This is more suitable for pedestrian routing as it accounts for cognitive load.
    
    Args:
        graph: Network graph with angular information
        node_id: Target node
        angular_weights: Dictionary of edge weights representing angular costs
    """
    if angular_weights is None:
        # Use uniform weights if no angular data provided
        angular_weights = {edge: 1.0 for edge in graph.edges()}
    
    try:
        # Calculate weighted shortest paths using angular costs
        shortest_paths = nx.single_source_dijkstra_path_length(
            graph, node_id, weight=lambda u, v, d: angular_weights.get((u, v), 1.0)
        )
        
        if len(shortest_paths) <= 1:
            return 0.0
        
        # Calculate mean angular depth
        total_angular_cost = sum(cost for target, cost in shortest_paths.items() 
                               if target != node_id)
        mean_angular_depth = total_angular_cost / (len(shortest_paths) - 1)
        
        # Angular integration (inverse relationship)
        angular_integration = 1 / (mean_angular_depth + 1)
        
        return angular_integration
        
    except Exception:
        return 0.0

def calculate_local_integration(graph: nx.Graph, node_id: int, radius: int = 3) -> float:
    """
    Calculate local integration within a specified topological radius.
    Often more predictive of pedestrian movement than global measures.
    
    Args:
        graph: Network graph
        node_id: Target node
        radius: Maximum topological distance to consider (typically 3-5)
    """
    try:
        # Get all nodes within the specified radius
        local_nodes = set([node_id])
        current_layer = set([node_id])
        
        for step in range(radius):
            next_layer = set()
            for node in current_layer:
                neighbors = set(graph.neighbors(node))
                next_layer.update(neighbors - local_nodes)
            
            if not next_layer:
                break
                
            local_nodes.update(next_layer)
            current_layer = next_layer
        
        # Create subgraph for local analysis
        local_subgraph = graph.subgraph(local_nodes)
        
        # Calculate integration within local area
        integration, _, _, _ = calculate_space_syntax_integration(local_subgraph, node_id)
        
        return integration
        
    except Exception:
        return 0.0

# Example usage and comparison
def compare_integration_measures(graph: nx.Graph, node_id: int):
    """
    Compare different integration measures for pedestrian routing analysis.
    """
    # Standard Space Syntax measures
    global_int, mean_depth, ra, rra = calculate_space_syntax_integration(graph, node_id)
    
    # Local integration (radius 3 - good for pedestrian scale)
    local_int_r3 = calculate_local_integration(graph, node_id, radius=3)
    
    # Angular integration (if angular data available)
    angular_int = calculate_angular_integration(graph, node_id)
    
    results = {
        'node_id': node_id,
        'global_integration': global_int,
        'mean_depth': mean_depth,
        'relative_asymmetry': ra,
        'real_relative_asymmetry': rra,
        'local_integration_r3': local_int_r3,
        'angular_integration': angular_int
    }
    
    return results

# Test with a simple example
if __name__ == "__main__":
    # Create a simple test graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 5), (5, 6)])
    
    # Calculate integration for node 2 (should be highly integrated)
    results = compare_integration_measures(G, 2)
    
    print("Integration Analysis Results:")
    for measure, value in results.items():
        print(f"{measure}: {value:.4f}")
