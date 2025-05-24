from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np
import pedpy
import time
import pickle
import json

# import shapely
# from collections import defaultdict, deque
import networkx as nx

# from scipy.spatial.distance import pdist, squareform
# import math
import pandas as pd
import tqdm


def cast_rays(polygon: Polygon, origin: Point, n_rays=720):
    """Cast rays from origin point and find intersections with polygon boundaries."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ray_length = 1000
    intersections = []

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        ray = LineString(
            [origin, (origin.x + dx * ray_length, origin.y + dy * ray_length)]
        )

        min_dist = float("inf")
        closest = None

        # Check intersection with exterior boundary
        for seg_start, seg_end in zip(
            polygon.exterior.coords[:-1], polygon.exterior.coords[1:]
        ):
            edge = LineString([seg_start, seg_end])
            if ray.intersects(edge):
                ip = ray.intersection(edge)
                if ip.geom_type == "Point":
                    dist = origin.distance(ip)
                    if dist < min_dist:
                        min_dist = dist
                        closest = ip

        # Check intersection with interior boundaries (obstacles/holes)
        for interior in polygon.interiors:
            for seg_start, seg_end in zip(interior.coords[:-1], interior.coords[1:]):
                edge = LineString([seg_start, seg_end])
                if ray.intersects(edge):
                    ip = ray.intersection(edge)
                    if ip.geom_type == "Point":
                        dist = origin.distance(ip)
                        if dist < min_dist:
                            min_dist = dist
                            closest = ip

        if closest:
            intersections.append((closest.x, closest.y))

    return Polygon(intersections) if intersections else None


def calculate_visual_connectivity(point: Point, all_points: list, isovist_polygon: Polygon):
    """
    Calculate visual connectivity - number of other grid points that are visible from this point.
    Uses the isovist polygon to determine visibility.
    
    Args:
        point: The observation point
        all_points: List of all grid points to check visibility to
        isovist_polygon: The isovist (visibility polygon) from the observation point
    
    Returns:
        int: Number of other points that are visually connected (visible)
    """
    if not isovist_polygon:
        return 0
    
    connected_count = 0
    for other_point in all_points:
        if point.equals(other_point):  # Skip self
            continue
            
        # Check if the other point is within the isovist polygon
        if isovist_polygon.contains(other_point):
            connected_count += 1
    
    return connected_count


def calculate_integration_depth(graph: nx.Graph, node_id: int):
    """
    Calculate integration (inverse of mean depth) using shortest path analysis.
    Lower mean depth = higher integration.
    """
    if len(graph.nodes()) <= 1:
        return 0, float("inf")

    try:
        shortest_paths = nx.single_source_shortest_path_length(graph, node_id)
        total_depth = sum(shortest_paths.values())
        mean_depth = total_depth / (len(shortest_paths) - 1)  # Exclude self
        integration = 1 / (mean_depth + 1)  # Add 1 to avoid division by zero
        return integration, mean_depth
    except:
        return 0, float("inf")


def calculate_through_vision(polygon: Polygon, point: Point, n_rays=36):
    """
    Calculate through-vision - longest unobstructed sight line from the point.
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    max_distance = 0

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        ray = LineString([point, (point.x + dx * 1000, point.y + dy * 1000)])

        min_dist = float("inf")

        # Check intersection with exterior boundary
        for seg_start, seg_end in zip(
            polygon.exterior.coords[:-1], polygon.exterior.coords[1:]
        ):
            edge = LineString([seg_start, seg_end])
            if ray.intersects(edge):
                ip = ray.intersection(edge)
                if ip.geom_type == "Point":
                    dist = point.distance(ip)
                    if dist < min_dist:
                        min_dist = dist

        # Check intersection with interior boundaries
        for interior in polygon.interiors:
            for seg_start, seg_end in zip(interior.coords[:-1], interior.coords[1:]):
                edge = LineString([seg_start, seg_end])
                if ray.intersects(edge):
                    ip = ray.intersection(edge)
                    if ip.geom_type == "Point":
                        dist = point.distance(ip)
                        if dist < min_dist:
                            min_dist = dist

        if min_dist != float("inf"):
            max_distance = max(max_distance, min_dist)

    return max_distance


def create_visibility_graph_from_isovists(points: list, isovists: list):
    """
    Create a graph connecting points that have mutual visibility based on isovist analysis.
    Two points are connected if they can see each other (both points are in each other's isovists).
    """
    graph = nx.Graph()

    # Add nodes
    for i, point in enumerate(points):
        graph.add_node(i, pos=(point.x, point.y), point=point)

    # Add edges for mutually visible connections
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            isovist1, isovist2 = isovists[i], isovists[j]
            
            # Check mutual visibility: p2 should be visible from p1 AND p1 should be visible from p2
            if (isovist1 and isovist1.contains(p2)) and (isovist2 and isovist2.contains(p1)):
                distance = p1.distance(p2)
                graph.add_edge(i, j, weight=distance)

    return graph


def calculate_route_score(
    point: Point,
    points: list,
    graph: nx.Graph,
    node_id: int,
    polygon: Polygon,
    isovist_polygon: Polygon,
    weights: dict = None,
):
    """
    Calculate comprehensive route score based on Space Syntax metrics.
    """
    if weights is None:
        weights = {
            "connectivity": 0.3,
            "integration": 0.3,
            "through_vision": 0.3,
            "depth_penalty": 0.1,
        }

    # Calculate metrics using proper visual connectivity
    visual_connectivity = calculate_visual_connectivity(point, points, isovist_polygon)
    integration, mean_depth = calculate_integration_depth(graph, node_id)
    through_vision = calculate_through_vision(polygon, point)
    depth_penalty = mean_depth

    # Normalize metrics
    max_connectivity = len(points) - 1
    norm_connectivity = visual_connectivity / max_connectivity if max_connectivity > 0 else 0
    norm_through_vision = min(through_vision / 50.0, 1.0)  # Normalize to max expected vision
    norm_depth_penalty = min(depth_penalty / 10.0, 1.0)  # Normalize depth penalty

    # Calculate weighted score
    score = (
        weights["connectivity"] * norm_connectivity
        + weights["integration"] * integration
        + weights["through_vision"] * norm_through_vision
        - weights["depth_penalty"] * norm_depth_penalty
    )

    return {
        "score": score,
        "visual_connectivity": visual_connectivity,
        "integration": integration,
        "through_vision": through_vision,
        "mean_depth": mean_depth,
        "normalized": {
            "connectivity": norm_connectivity,
            "integration": integration,
            "through_vision": norm_through_vision,
            "depth_penalty": norm_depth_penalty,
        },
    }


def generate_grid_points(polygon: Polygon, grid_spacing: float = 1.0):
    """
    Generate points on a regular grid within the polygon.

    Args:
        polygon: The polygon to generate points within
        grid_spacing: Distance between grid points

    Returns:
        list: List of Point objects on the grid that are inside the polygon
    """
    minx, miny, maxx, maxy = polygon.bounds

    # Create grid coordinates
    x_coords = np.arange(minx, maxx + grid_spacing, grid_spacing)
    y_coords = np.arange(miny, maxy + grid_spacing, grid_spacing)

    points = []
    grid_info = []  # Store grid indices for each point

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(x, y)
            if polygon.contains(point):
                points.append(point)
                grid_info.append((i, j, x, y))

    print(f"Generated {len(points)} grid points with spacing {grid_spacing}")
    print(f"Grid dimensions: {len(x_coords)} x {len(y_coords)}")

    return points, grid_info


def generate_adaptive_grid_points(polygon: Polygon, target_density: int = 100):
    """
    Generate grid points with adaptive spacing based on polygon size.

    Args:
        polygon: The polygon to generate points within
        target_density: Approximate number of points desired

    Returns:
        list: List of Point objects on the grid
    """
    minx, miny, maxx, maxy = polygon.bounds
    area = polygon.area

    # Calculate grid spacing based on target density
    approx_spacing = np.sqrt(area / target_density)

    return generate_grid_points(polygon, approx_spacing)


def plot_space_syntax_analysis_grid(
    polygon: Polygon,
    points: list,
    scores: list,
    graph: nx.Graph,
    grid_info: list = None,
):
    """
    Visualize the space syntax analysis with color-coded route scores for grid points.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Extract metrics for visualization
    route_scores = [s["score"] for s in scores]
    visual_connectivities = [s["visual_connectivity"] for s in scores]
    integrations = [s["integration"] for s in scores]
    through_visions = [s["through_vision"] for s in scores]

    # Plot 1: Overall Route Scores
    ax1.plot(*polygon.exterior.xy, color="black", linewidth=2)
    for interior in polygon.interiors:
        ax1.fill(*interior.xy, color="gray", alpha=0.7)
        ax1.plot(*interior.xy, color="black", linewidth=1)

    scatter1 = ax1.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c=route_scores,
        cmap="RdYlGn",
        s=50,  # Smaller points for grid
        alpha=0.8,
        marker="s",  # Square markers for grid points
    )
    ax1.set_title("Overall Route Scores (Grid)")
    ax1.set_aspect("equal")
    plt.colorbar(scatter1, ax=ax1, label="Route Score")

    # Plot 2: Visual Connectivity (corrected)
    ax2.plot(*polygon.exterior.xy, color="black", linewidth=2)
    for interior in polygon.interiors:
        ax2.fill(*interior.xy, color="gray", alpha=0.7)
        ax2.plot(*interior.xy, color="black", linewidth=1)

    scatter2 = ax2.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c=visual_connectivities,
        cmap="Blues",
        s=50,
        alpha=0.8,
        marker="s",
    )
    ax2.set_title("Visual Connectivity (Grid - Isovist-based)")
    ax2.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax2, label="Visually Connected Points")

    # Plot 3: Integration with Graph
    ax3.plot(*polygon.exterior.xy, color="black", linewidth=2)
    for interior in polygon.interiors:
        ax3.fill(*interior.xy, color="gray", alpha=0.7)
        ax3.plot(*interior.xy, color="black", linewidth=1)

    # Draw graph edges (but make them lighter for grid)
    pos = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_edges(
        graph, pos, ax=ax3, alpha=0.1, edge_color="lightblue", width=0.5
    )

    scatter3 = ax3.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c=integrations,
        cmap="Oranges",
        s=50,
        alpha=0.8,
        marker="s",
    )
    ax3.set_title("Integration (Grid with Visibility Graph)")
    ax3.set_aspect("equal")
    plt.colorbar(scatter3, ax=ax3, label="Integration Value")

    # Plot 4: Through Vision
    ax4.plot(*polygon.exterior.xy, color="black", linewidth=2)
    for interior in polygon.interiors:
        ax4.fill(*interior.xy, color="gray", alpha=0.7)
        ax4.plot(*interior.xy, color="black", linewidth=1)

    scatter4 = ax4.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c=through_visions,
        cmap="Purples",
        s=50,
        alpha=0.8,
        marker="s",
    )
    ax4.set_title("Through Vision (Grid - Max Sight Distance)")
    ax4.set_aspect("equal")
    plt.colorbar(scatter4, ax=ax4, label="Max Vision Distance")

    plt.tight_layout()
    plt.savefig("Metrics_Grid_Fixed.pdf")
    print(">> Metrics_Grid_Fixed.pdf")


def create_test_polygon_with_obstacles():
    """Create a test polygon with interior obstacles."""
    exterior = [(0, 0), (20, 0), (20, 15), (0, 15)]
    obstacle1 = [(3, 3), (7, 3), (7, 7), (3, 7)]
    obstacle2 = [(12, 8), (16, 8), (16, 12), (12, 12)]
    obstacle3 = [(2, 10), (5, 10), (5, 13), (2, 13)]

    return Polygon(exterior, [obstacle1, obstacle2, obstacle3])


def find_optimal_route(
    start_point: Point, end_point: Point, points: list, scores: list, graph: nx.Graph
):
    """
    Find optimal route considering space syntax scores.
    """
    # Find nearest points to start and end
    start_idx = min(range(len(points)), key=lambda i: points[i].distance(start_point))
    end_idx = min(range(len(points)), key=lambda i: points[i].distance(end_point))

    try:
        # Find shortest path in graph
        path_indices = nx.shortest_path(graph, start_idx, end_idx)

        # Calculate route quality based on space syntax scores
        route_quality = np.mean([scores[i]["score"] for i in path_indices])

        route_points = [points[i] for i in path_indices]

        return {
            "path_indices": path_indices,
            "route_points": route_points,
            "route_quality": route_quality,
            "total_distance": sum(
                points[path_indices[i]].distance(points[path_indices[i + 1]])
                for i in range(len(path_indices) - 1)
            ),
        }
    except nx.NetworkXNoPath:
        return None


def save_isovists_for_simulation(points: list, isovists: list, filename: str):
    """
    Save isovist polygons for later use in simulations to avoid recalculation.
    Saves both as pickle (for Python) and WKT format (for interoperability).
    """
    # Save as pickle for fast Python loading
    isovist_data = {
        'points': [(p.x, p.y) for p in points],
        'isovists_wkt': [isovist.wkt if isovist else None for isovist in isovists],
        'grid_size': len(points)
    }
    
    with open(filename.replace('.csv', '_isovists.pkl'), 'wb') as f:
        pickle.dump(isovist_data, f)
    
    # Also save as JSON with WKT strings for other languages
    with open(filename.replace('.csv', '_isovists.json'), 'w') as f:
        json.dump(isovist_data, f, indent=2)
    
    print(f"Saved isovists to {filename.replace('.csv', '_isovists.pkl')} and .json")
    return isovist_data


def export_grid_results_with_coordinates(
    points: list, scores: list, grid_info: list, isovists: list, filename: str
):
    """
    Export results with grid coordinates and isovist information for easier analysis.
    """
    results = []
    for i, (score_data, point, isovist) in enumerate(zip(scores, points, isovists)):
        result_dict = {
            "x": point.x,
            "y": point.y,
            "score": score_data["score"],
            "visual_connectivity": score_data["visual_connectivity"],
            "integration": score_data["integration"],
            "through_vision": score_data["through_vision"],
            "mean_depth": score_data["mean_depth"],
            "norm_connectivity": score_data["normalized"]["connectivity"],
            "norm_integration": score_data["normalized"]["integration"],
            "norm_through_vision": score_data["normalized"]["through_vision"],
            "norm_depth_penalty": score_data["normalized"]["depth_penalty"],
            "isovist_area": isovist.area if isovist else 0,
            "isovist_wkt": isovist.wkt if isovist else None,
        }

        # Add grid information if available
        if grid_info and i < len(grid_info):
            grid_i, grid_j, grid_x, grid_y = grid_info[i]
            result_dict.update(
                {"grid_i": grid_i, "grid_j": grid_j, "grid_x": grid_x, "grid_y": grid_y}
            )

        results.append(result_dict)

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved grid results to {filename}")
    
    # Save isovists separately for simulation use
    save_isovists_for_simulation(points, isovists, filename)
    
    return df


import argparse

parser = argparse.ArgumentParser(description="Run space syntax grid analysis with proper visual connectivity.")
parser.add_argument(
    "--wkt",
    type=str,
    default="files/gallery.wkt",
    help="Path to WKT file. (gallery.wkt)",
)
parser.add_argument(
    "--grid", type=float, default=1.0, help="Grid spacing in meters. (1.0)"
)
parser.add_argument(
    "--output",
    type=str,
    default="metrics_gallery_grid_fixed.csv",
    help="Output CSV filename. (metrics_gallery_grid_fixed.csv)",
)
parser.add_argument(
    "--rays", type=int, default=720, help="Number of rays for isovist calculation. (720)"
)
args = parser.parse_args()


if __name__ == "__main__":
    # Load or create test polygon
    try:
        with open(args.wkt, "r") as f:
            geometry = wkt.loads(f.read())
        walkable_area = pedpy.WalkableArea(geometry.geoms[0])
        polygon = walkable_area.polygon
        print("Loaded polygon from gallery.wkt")
    except FileNotFoundError:
        print("gallery.wkt not found, using test polygon with obstacles")
        polygon = create_test_polygon_with_obstacles()

    grid_spacing = args.grid
    n_rays = args.rays
    
    # Generate grid points
    start = time.time()
    points, grid_info = generate_grid_points(polygon, grid_spacing)
    print(f"Generated {len(points)} grid points in {time.time() - start:.2f} seconds.")

    # Calculate isovists for all grid points
    print("Calculating isovists for all grid points...")
    start = time.time()
    isovists = []
    for point in tqdm.tqdm(points, desc="Computing isovists"):
        isovist = cast_rays(polygon, point, n_rays)
        isovists.append(isovist)
    print(f"Calculated {len(isovists)} isovists in {time.time() - start:.2f} seconds.")

    # Create visibility graph based on mutual isovist visibility
    start = time.time()
    visibility_graph = create_visibility_graph_from_isovists(points, isovists)
    print(
        f"Created visibility graph with {len(visibility_graph.edges())} connections in {time.time() - start:.2f} seconds."
    )

    # Calculate space syntax metrics for each point
    scores = []
    start = time.time()
    print("Calculating space syntax metrics with proper visual connectivity...")
    for i, (point, isovist) in enumerate(tqdm.tqdm(zip(points, isovists), desc="Computing metrics")):
        score_data = calculate_route_score(point, points, visibility_graph, i, polygon, isovist)
        scores.append(score_data)

    # Print top scoring locations
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1]["score"], reverse=True)
    print("\nTop 5 grid locations by route score (with proper visual connectivity):")
    for i, (idx, score_data) in enumerate(sorted_scores[:5]):
        point = points[idx]
        print(
            f"{i + 1}. Grid Point ({point.x:.2f}, {point.y:.2f}): Score={score_data['score']:.3f}"
        )
        print(
            f"   Visual Connectivity: {score_data['visual_connectivity']}, Integration: {score_data['integration']:.3f}"
        )
        print(
            f"   Through-vision: {score_data['through_vision']:.2f}, Mean depth: {score_data['mean_depth']:.2f}"
        )

    # Generate visualization
    start = time.time()
    plot_space_syntax_analysis_grid(
        polygon, points, scores, visibility_graph, grid_info
    )
    print(f"Finished plot in {time.time() - start:.2f} seconds.")

    # Export results with grid information and isovists
    df = export_grid_results_with_coordinates(points, scores, grid_info, isovists, args.output)

    # Print some statistics
    print("\nFixed Grid Analysis Statistics:")
    print(f"Total grid points: {len(points)}")
    print(f"Grid spacing: {grid_spacing}")
    print(f"Rays per isovist: {n_rays}")
    print(f"Average route score: {df['score'].mean():.3f}")
    print(f"Max route score: {df['score'].max():.3f}")
    print(f"Min route score: {df['score'].min():.3f}")
    print(f"Average visual connectivity: {df['visual_connectivity'].mean():.1f}")
    print(f"Max visual connectivity: {df['visual_connectivity'].max()}")

    # Example route finding using grid points
    if len(points) >= 2:
        start_point = points[0]
        end_point = points[-1]
        route = find_optimal_route(
            start_point, end_point, points, scores, visibility_graph
        )

        if route:
            print("\nOptimal route found on grid:")
            print(
                f"From: ({start_point.x:.2f}, {start_point.y:.2f}) To: ({end_point.x:.2f}, {end_point.y:.2f})"
            )
            print(f"Route quality score: {route['route_quality']:.3f}")
            print(f"Total distance: {route['total_distance']:.2f}")
            print(f"Number of waypoints: {len(route['route_points'])}")
            
    print("\nFiles saved:")
    print(f"- {args.output} (CSV with all metrics)")
    print(f"- {args.output.replace('.csv', '_isovists.pkl')} (Isovist polygons for simulation)")
    print(f"- {args.output.replace('.csv', '_isovists.json')} (Isovist polygons as JSON)")
    print("- Metrics_Grid_Fixed.pdf (Visualization)")
