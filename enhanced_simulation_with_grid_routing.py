from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import pathlib
import jupedsim as jps
from scipy.spatial import cKDTree


class SpaceSyntaxData:
    """Class to handle pre-calculated Space Syntax data from CSV."""

    def __init__(self, csv_file_path: str):
        """Load and prepare space syntax data from CSV."""
        self.df = pd.read_csv(csv_file_path)
        # Create spatial index for fast nearest neighbor queries
        self.points = np.column_stack((self.df["x"].values, self.df["y"].values))
        self.kdtree = cKDTree(self.points)

    def get_nearest_point_data(self, point: Point, k_neighbors: int = 1):
        """Get data for the nearest point(s) to the given location."""
        query_point = np.array([point.x, point.y])
        distances, indices = self.kdtree.query(query_point, k=k_neighbors)

        if k_neighbors == 1:
            # Return as DataFrame with single row for consistent access
            return self.df.iloc[[indices]]
        else:
            return self.df.iloc[indices]

    def get_points_within_radius(self, point: Point, radius: float):
        """Get all points within a given radius of the query point."""
        query_point = np.array([point.x, point.y])
        indices = self.kdtree.query_ball_point(query_point, radius)
        return self.df.iloc[indices] if indices else pd.DataFrame()


def validate_and_adjust_target(
    best_point: Point,
    current_pos: Point,
    polygon: Polygon,
    step_back_distance: float = 0.1,
    max_attempts: int = 5,
):
    """
    Enhanced unstuck mechanism that validates target points and adjusts them if needed.

    Args:
        best_point: The computed target point
        current_pos: Current agent position
        polygon: The walkable area polygon
        step_back_distance: Distance to step back from invalid targets (meters)
        max_attempts: Maximum number of adjustment attempts

    Returns:
        tuple: (adjusted_point, adjustment_info)
    """
    adjustment_info = {
        "was_adjusted": False,
        "adjustment_type": None,
        "attempts": 0,
        "final_distance_from_original": 0.0,
    }

    # If the point is already valid, return it
    if polygon.contains(best_point):
        return best_point, adjustment_info

    adjustment_info["was_adjusted"] = True

    # Calculate direction from invalid target back to current position
    direction = np.array([current_pos.x - best_point.x, current_pos.y - best_point.y])
    norm = np.linalg.norm(direction)

    if norm < 0.01:  # Points are essentially the same
        print("Stuck: best point is current location")
        adjustment_info["adjustment_type"] = "stuck_at_current"
        return current_pos, adjustment_info

    # Normalize direction vector
    direction /= norm
    adjustment_info["attempts"] = 1

    # Try stepping back from the invalid target
    for attempt in range(max_attempts):
        step_distance = step_back_distance * (attempt + 1)
        adjusted = Point(
            best_point.x + direction[0] * step_distance,
            best_point.y + direction[1] * step_distance,
        )

        if polygon.contains(adjusted):
            adjustment_info["adjustment_type"] = "stepped_back"
            adjustment_info["attempts"] = attempt + 1
            adjustment_info["final_distance_from_original"] = step_distance
            return adjusted, adjustment_info

    # If stepping back didn't work, try moving towards goal from current position
    print("Stepping back failed, trying direct approach towards goal...")

    # This requires the goal to be passed, so we'll add it as a parameter
    # For now, fallback to current position
    adjustment_info["adjustment_type"] = "fallback_to_current"
    print("Using current position as fallback")
    return current_pos, adjustment_info


def get_fallback_target(
    current_pos: Point,
    goal: Point,
    polygon: Polygon,
    step_size: float = 0.5,
    max_steps: int = 10,
):
    """
    Generate a fallback target by taking small steps towards the goal.

    Args:
        current_pos: Current agent position
        goal: Target goal position
        polygon: Walkable area polygon
        step_size: Size of each step towards goal
        max_steps: Maximum number of steps to try

    Returns:
        Point: A valid target point
    """
    direction_to_goal = np.array([goal.x - current_pos.x, goal.y - current_pos.y])
    distance_to_goal = np.linalg.norm(direction_to_goal)

    if distance_to_goal < 0.01:
        return current_pos

    # Normalize direction
    direction_to_goal /= distance_to_goal

    # Try progressively smaller steps towards goal
    for i in range(1, max_steps + 1):
        step_distance = min(step_size * i, distance_to_goal * 0.8)
        candidate = Point(
            current_pos.x + direction_to_goal[0] * step_distance,
            current_pos.y + direction_to_goal[1] * step_distance,
        )

        if polygon.contains(candidate):
            print(f"✓ Found fallback target {step_distance:.2f}m towards goal")
            return candidate

    print("⚠ No valid fallback target found, staying at current position")
    return current_pos


def plot_isovist_from_csv(
    env_polygon: Polygon,
    current_pos: Point,
    goal: Point,
    best_point: Point,
    step: int,
    space_syntax_data: SpaceSyntaxData,
    search_radius: float = 2.0,
    adjustment_info: dict = None,
):
    """Plot the environment with isovist data from CSV and adjustment information."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot exterior boundary
    ax.plot(*env_polygon.exterior.xy, color="black", linewidth=2)

    # Plot obstacles (interior boundaries)
    for interior in env_polygon.interiors:
        ax.fill(*interior.xy, color="gray", alpha=0.7)
        ax.plot(*interior.xy, color="black", linewidth=1.5)

    # Get current position data and plot isovist
    current_data = space_syntax_data.get_nearest_point_data(current_pos)

    # Try to plot the computed isovist if available
    try:
        current_data = space_syntax_data.get_nearest_point_data(current_pos)

        if "isovist_wkt" in current_data.columns and pd.notna(
            current_data["isovist_wkt"].iloc[0]
        ):
            try:
                isovist = wkt.loads(current_data["isovist_wkt"].iloc[0])
                if isovist:
                    ax.fill(
                        *isovist.exterior.xy,
                        color="skyblue",
                        alpha=0.5,
                        label="Current Isovist",
                    )
                    ax.plot(*isovist.exterior.xy, color="blue", linewidth=1)
            except Exception as e:
                print(f"Error plotting isovist: {e}")
    except Exception as e:
        print(f"Error plotting computed isovist: {e}")

    # Plot nearby points colored by score
    nearby_points = space_syntax_data.get_points_within_radius(
        current_pos, search_radius
    )
    if not nearby_points.empty:
        scatter = ax.scatter(
            nearby_points["x"],
            nearby_points["y"],
            c=nearby_points["score"],
            cmap="viridis",
            s=30,
            alpha=0.7,
            label="Space Syntax Points",
        )
        plt.colorbar(scatter, ax=ax, label="Space Syntax Score")

    # Plot key points with different colors based on adjustment
    ax.plot(current_pos.x, current_pos.y, "ro", markersize=10, label="Agent", zorder=5)

    # Color the target point based on whether it was adjusted
    if adjustment_info and adjustment_info["was_adjusted"]:
        if adjustment_info["adjustment_type"] == "stepped_back":
            ax.plot(
                best_point.x,
                best_point.y,
                "go",
                markersize=10,
                label="Adjusted Target",
                zorder=5,
            )
        elif adjustment_info["adjustment_type"] == "fallback_to_current":
            ax.plot(
                best_point.x,
                best_point.y,
                "mo",
                markersize=10,
                label="Fallback Target",
                zorder=5,
            )
        else:
            ax.plot(
                best_point.x,
                best_point.y,
                "co",
                markersize=10,
                label="Stuck Target",
                zorder=5,
            )
    else:
        ax.plot(
            best_point.x,
            best_point.y,
            "go",
            markersize=10,
            label="Next Target",
            zorder=5,
        )

    ax.plot(goal.x, goal.y, "bx", markersize=12, label="Goal", zorder=5)

    ax.set_aspect("equal")
    ax.legend()

    # Add score and adjustment information to title
    score = current_data["score"].iloc[0] if "score" in current_data.columns else 0
    dist = current_pos.distance(goal)
    title = f"Step {step} – Distance to goal: {dist:.2f}m – Space Score: {score:.3f}"

    if adjustment_info and adjustment_info["was_adjusted"]:
        title += f" – Adjusted ({adjustment_info['adjustment_type']})"

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"isovist_step_{step:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def is_goal_reached(current, goal, threshold=0.5):
    """Check if the goal has been reached."""
    return current.distance(goal) < threshold


# def cast_rays(polygon: Polygon, origin: Point, n_rays=720):
#     """
#     Cast rays from origin point and find intersections with polygon boundaries.

#     including both exterior and interior boundaries (obstacles).
#     """
#     angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
#     ray_length = 1000
#     intersections = []

#     for angle in angles:
#         dx = np.cos(angle)
#         dy = np.sin(angle)
#         ray = LineString(
#             [origin, (origin.x + dx * ray_length, origin.y + dy * ray_length)]
#         )

#         min_dist = float("inf")
#         closest = None

#         # Check intersection with exterior boundary
#         for seg_start, seg_end in zip(
#             polygon.exterior.coords[:-1], polygon.exterior.coords[1:]
#         ):
#             edge = LineString([seg_start, seg_end])
#             if ray.intersects(edge):
#                 ip = ray.intersection(edge)
#                 if ip.geom_type == "Point":
#                     dist = origin.distance(ip)
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest = ip

#         # Check intersection with interior boundaries (obstacles/holes)
#         for interior in polygon.interiors:
#             for seg_start, seg_end in zip(interior.coords[:-1], interior.coords[1:]):
#                 edge = LineString([seg_start, seg_end])
#                 if ray.intersects(edge):
#                     ip = ray.intersection(edge)
#                     if ip.geom_type == "Point":
#                         dist = origin.distance(ip)
#                         if dist < min_dist:
#                             min_dist = dist
#                             closest = ip

#         if closest:
#             intersections.append((closest.x, closest.y))

#     return Polygon(intersections) if intersections else None


def compute_next_step(
    current: Point,
    goal: Point,
    polygon: Polygon,
    space_syntax_data: SpaceSyntaxData,
    weights: dict = None,
):
    """
    Compute next step using isovist-based candidate selection and Space Syntax scoring.

    Combines directional alignment, progress toward goal, and configurable space syntax metrics.
    """
    if weights is None:
        weights = {
            "connectivity": 0.3,
            "integration": 0.3,
            "through_vision": 0.3,
            "depth_penalty": 0.1,
        }

    # Get current position data and isovist
    current_data = space_syntax_data.get_nearest_point_data(current)
    isovist = None

    if "isovist_wkt" in current_data.columns and pd.notna(
        current_data["isovist_wkt"].iloc[0]
    ):
        try:
            isovist = wkt.loads(current_data["isovist_wkt"].iloc[0])
        except Exception as e:
            print(f"Error loading isovist from CSV: {e}")

    if not isovist:
        print("No valid isovist found, using fallback target")
        return get_fallback_target(current, goal, polygon), None

    # Check if goal is directly visible
    if isovist.contains(goal):
        return goal, None

    # Calculate scores for all isovist boundary points
    current_dist = current.distance(goal)
    best_score = -np.inf
    best_point = current
    best_data = None

    # Direction to goal for alignment calculation
    direction_to_goal = np.array([goal.x - current.x, goal.y - current.y])
    direction_to_goal_norm = np.linalg.norm(direction_to_goal)

    if direction_to_goal_norm > 0:
        direction_to_goal /= direction_to_goal_norm

    for x, y in isovist.exterior.coords:
        candidate = Point(x, y)
        new_dist = candidate.distance(goal)

        # Calculate directional alignment score
        direction_to_candidate = np.array([x - current.x, y - current.y])

        if np.linalg.norm(direction_to_candidate) < 1e-6:
            continue

        # Normalize vectors
        dist_to_canditate = np.linalg.norm(direction_to_candidate)
        direction_to_candidate /= dist_to_canditate

        alignment_score = (
            np.dot(direction_to_goal, direction_to_candidate)
            if direction_to_goal_norm > 0
            else 0
        )
        progress_score = (
            current_dist - new_dist
        ) * 2  # Weighted progress. TODO: hard-coded weight

        # Get Space Syntax metrics from CSV for this candidate point
        candidate_data = space_syntax_data.get_nearest_point_data(candidate)

        if not candidate_data.empty:
            # Calculate Space Syntax score using configurable weights
            norm_connectivity = (
                candidate_data["norm_connectivity"].iloc[0]
                if "norm_connectivity" in candidate_data.columns
                else 0
            )
            norm_integration = (
                candidate_data["norm_integration"].iloc[0]
                if "norm_integration" in candidate_data.columns
                else 0
            )
            norm_through_vision = (
                candidate_data["norm_through_vision"].iloc[0]
                if "norm_through_vision" in candidate_data.columns
                else 0
            )
            norm_depth_penalty = (
                candidate_data["norm_depth_penalty"].iloc[0]
                if "norm_depth_penalty" in candidate_data.columns
                else 0
            )

            space_syntax_score = (
                weights["connectivity"] * norm_connectivity
                + weights["integration"] * norm_integration
                + weights["through_vision"] * norm_through_vision
                - weights["depth_penalty"] * norm_depth_penalty
            )
        else:
            space_syntax_score = 0.0

        # Combine all scores. dist_to_candidate favores long tentacles.
        total_score = (
            alignment_score + progress_score + space_syntax_score + dist_to_canditate
        )

        if total_score > best_score:
            best_score = total_score
            best_point = candidate
            best_data = (
                candidate_data.iloc[0].to_dict() if not candidate_data.empty else None
            )

    return best_point, best_data


def initialize_simulation(model, output_file, geometry, exit_polygon):
    """Initialize the JuPedSim simulation."""
    simulation = jps.Simulation(
        model=model,
        geometry=geometry.polygon,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(output_file), every_nth_frame=10
        ),
    )
    exit_id = simulation.add_exit_stage(exit_polygon)
    direct_steering_stage = simulation.add_direct_steering_stage()

    direct_journey = jps.JourneyDescription([direct_steering_stage])
    exit_journey = jps.JourneyDescription([exit_id])
    direct_journey_id = simulation.add_journey(direct_journey)
    exit_journey_id = simulation.add_journey(exit_journey)
    return (
        simulation,
        direct_journey_id,
        exit_journey_id,
        direct_steering_stage,
        exit_id,
    )


def add_agents(
    simulation,
    direct_journey_id,
    exit_journey_id,
    stage_id,
    exit_id,
    agent_positions,
    agent_parameters,
):
    """Add agents to the simulation."""
    for i, pos in enumerate(agent_positions):
        if i == 0:  # First agent uses direct steering with space syntax
            simulation.add_agent(
                agent_parameters(
                    journey_id=direct_journey_id,
                    stage_id=stage_id,
                    position=pos,
                    desired_speed=1.0,
                    radius=0.15,
                )
            )
        else:  # Other agents use standard exit behavior
            simulation.add_agent(
                agent_parameters(
                    journey_id=exit_journey_id,
                    stage_id=exit_id,
                    position=pos,
                    desired_speed=1.0,
                    radius=0.15,
                )
            )


if __name__ == "__main__":
    # Configuration
    CSV_FILE_PATH = (
        "metrics_gallery_grid_fixed.csv"  # Update this path to your CSV file
    )
    VISUALIZATION_FREQUENCY = 100  # Plot every N steps

    # Load Space Syntax data
    print("Loading Space Syntax data from CSV...")
    space_syntax_data = SpaceSyntaxData(CSV_FILE_PATH)
    print(f"Loaded {len(space_syntax_data.df)} space syntax points")

    # Load geometry
    with open("files/gallery.wkt", "r") as f:
        geometry = wkt.loads(f.read())
    walkable_area = pedpy.WalkableArea(geometry.geoms[0])
    polygon = walkable_area.polygon

    # Define areas
    exit_polygon = Polygon([[21, 12], [24, 12], [24, 10], [21, 10]])
    spawning_area = Polygon([[-16, -1], [-12, -1], [-12, -4], [-16, -4]])

    # Initialize simulation
    output_file = "traj.sqlite"
    collision_free_model = jps.CollisionFreeSpeedModel()
    simulation, direct_journey_id, exit_journey_id, stage_id, exit_id = (
        initialize_simulation(
            collision_free_model, output_file, walkable_area, exit_polygon
        )
    )

    # Distribute agents
    pos_in_spawning_area = jps.distribute_by_number(
        polygon=spawning_area,
        number_of_agents=2,
        distance_to_agents=0.30,
        distance_to_polygon=0.15,
        seed=1,
    )

    add_agents(
        simulation,
        direct_journey_id=direct_journey_id,
        exit_journey_id=exit_journey_id,
        stage_id=stage_id,
        exit_id=exit_id,
        agent_positions=pos_in_spawning_area,
        agent_parameters=jps.CollisionFreeSpeedModelAgentParameters,
    )

    # Simulation parameters
    goal = exit_polygon.centroid
    print(f"Goal position: {goal}")

    # Main simulation loop
    while simulation.agent_count() > 0 and simulation.iteration_count() < 10000:
        step = simulation.iteration_count()

        for i, agent in enumerate(simulation.agents()):
            if (
                i == 0 and step % VISUALIZATION_FREQUENCY == 0
            ):  # Only the first agent uses space syntax guidance
                print(f"\nUpdating route for agent {i} at step: {step}")
                current_pos = Point(agent.position)

                # Compute next step using Space Syntax data with configurable weights
                space_syntax_weights = {
                    "connectivity": 0.3,
                    "integration": 0.3,
                    "through_vision": 0.3,
                    "depth_penalty": 0.1,
                }

                best_point, best_data = compute_next_step(
                    current_pos, goal, polygon, space_syntax_data, space_syntax_weights
                )

                # Enhanced unstuck mechanism
                adjusted_point, adjustment_info = validate_and_adjust_target(
                    best_point, current_pos, polygon
                )

                # If basic adjustment failed, try fallback target
                if (
                    adjustment_info["adjustment_type"] == "fallback_to_current"
                    and adjusted_point == current_pos
                ):
                    print("⚠ Trying fallback target generation...")
                    adjusted_point = get_fallback_target(current_pos, goal, polygon)
                    adjustment_info["adjustment_type"] = "fallback_towards_goal"

                best_point = adjusted_point

                # Set agent target
                agent.target = (best_point.x, best_point.y)

                # Visualization
                if step % VISUALIZATION_FREQUENCY == 0:
                    plot_isovist_from_csv(
                        polygon,
                        current_pos,
                        goal,
                        best_point,
                        step,
                        space_syntax_data,
                        adjustment_info=adjustment_info,
                    )

                # Log information
                current_dist = current_pos.distance(goal)
                space_score = (
                    best_data.get("score", 0.0) if best_data is not None else 0.0
                )

                print(f"Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                print(f"Next target: ({best_point.x:.2f}, {best_point.y:.2f})")
                print(f"Space Syntax score: {space_score:.3f}")

                # Log adjustment information
                if adjustment_info["was_adjusted"]:
                    print(f"Target adjustment: {adjustment_info['adjustment_type']}")

                if best_data is not None:
                    print(f"Connectivity: {best_data.get('norm_connectivity', 0):.3f}")
                    print(f"Integration: {best_data.get('norm_integration', 0):.3f}")
                    print(
                        f"Through Vision: {best_data.get('norm_through_vision', 0):.3f}"
                    )
                    print(
                        f"Depth Penalty: {best_data.get('norm_depth_penalty', 0):.3f}"
                    )

                print(f"Distance to goal: {current_dist:.2f}m")
                # Check if goal is reached
                if is_goal_reached(current_pos, goal):
                    simulation.mark_agent_for_removal(agent.id)
                    print(f"✓ Agent {i} reached the goal!")

        simulation.iterate()

    print("\n🏁 Simulation completed!")
    print(f"Total evacuation time: {simulation.elapsed_time():.2f} seconds")
    print(f"Total iterations: {simulation.iteration_count()}")
