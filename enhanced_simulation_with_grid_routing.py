"""Enhanced Space Syntax Pedestrian Simulation with Grid Routing.

This module provides a pedestrian simulation system that uses Space Syntax theory
to guide agent movement through complex environments. Agents make routing decisions
based on pre-computed spatial metrics including connectivity, integration, and
visibility analysis.
"""

from __future__ import annotations
import os
import pathlib
from typing import Dict, List, Optional, Tuple
import warnings
from collections import defaultdict, deque
import random
import numpy as np
import pandas as pd
import typer
from shapely import wkt
from shapely.geometry import Point, Polygon

from utils import (
    is_goal_reached,
    plot_isovist_from_csv,
    SpaceSyntaxData,
    AdjustmentInfo,
    calculate_score,
)

try:
    import pedpy
    import jupedsim as jps
except ImportError as e:
    typer.echo(f"Error: Required simulation libraries not found: {e}", err=True)
    typer.echo("Please install: pip install jupedsim pedpy", err=True)
    raise typer.Exit(1)

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

grid_usage: Dict[Tuple[int, int], int] = defaultdict(int)

# Enhanced progress tracking
progress_tracker = defaultdict(lambda: deque(maxlen=50))  # Longer history
position_history = defaultdict(lambda: deque(maxlen=100))  # Position history
stuck_counter = defaultdict(int)  # Count consecutive stuck detections
exploration_mode = defaultdict(bool)  # Flag for exploration mode
forbidden_zones = defaultdict(list)  # Areas to avoid when stuck

# Short-term memory for recently visited positions
recent_positions = defaultdict(lambda: deque(maxlen=20))  # Increased memory


def is_agent_stuck(
    agent_id, current_pos, current_dist, threshold_distance=0.2, threshold_position=1.5
):
    """Enhanced stuck detection with both distance and position criteria"""
    progress_tracker[agent_id].append(current_dist)
    position_history[agent_id].append(current_pos)

    # Check distance progress
    distance_stuck = False
    if len(progress_tracker[agent_id]) >= 30:
        recent_distances = list(progress_tracker[agent_id])[-30:]
        recent_progress = max(recent_distances) - min(recent_distances)
        if recent_progress < threshold_distance:
            distance_stuck = True

    # Check position oscillation
    position_stuck = False
    if len(position_history[agent_id]) >= 20:
        recent_positions_list = list(position_history[agent_id])[-20:]
        max_distance_between_positions = 0
        for i in range(len(recent_positions_list)):
            for j in range(i + 1, len(recent_positions_list)):
                dist = recent_positions_list[i].distance(recent_positions_list[j])
                max_distance_between_positions = max(
                    max_distance_between_positions, dist
                )

        if max_distance_between_positions < threshold_position:
            position_stuck = True

    is_stuck = distance_stuck or position_stuck

    if is_stuck:
        stuck_counter[agent_id] += 1
        # Add current area to forbidden zones
        forbidden_zones[agent_id].append((current_pos, 2.0))  # Avoid 2m radius
        # Limit forbidden zones to prevent too many restrictions
        if len(forbidden_zones[agent_id]) > 5:
            forbidden_zones[agent_id].pop(0)
    else:
        stuck_counter[agent_id] = 0

    return is_stuck


def choose_stochastic_best(candidates_with_scores, agent_id):
    """Enhanced selection with different strategies based on stuck level."""
    if not candidates_with_scores:
        return None

    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

    # If heavily stuck, be more random in selection
    if stuck_counter[agent_id] > 1:
        # More random selection from top candidates
        top_k = candidates_with_scores[: min(8, len(candidates_with_scores))]
        weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1][: len(top_k)]
        return random.choices([pt for pt, _ in top_k], weights=weights, k=1)[0]
    else:
        # Normal selection
        top_k = candidates_with_scores[: min(3, len(candidates_with_scores))]
        scores = [score for _, score in top_k]
        if max(scores) - min(scores) < 1e-4:
            return random.choice([pt for pt, _ in top_k])
        else:
            weights = [0.6, 0.3, 0.1][: len(top_k)]
            return random.choices([pt for pt, _ in top_k], weights=weights, k=1)[0]


def find_best_escape_direction(current_pos, goal, polygon, agent_id, max_distance=5.0):
    """Find a good escape direction when stuck."""
    # Try directions perpendicular to goal direction
    to_goal = np.array([goal.x - current_pos.x, goal.y - current_pos.y])
    to_goal_norm = to_goal / np.linalg.norm(to_goal)

    # Perpendicular directions
    perp1 = np.array([-to_goal_norm[1], to_goal_norm[0]])
    perp2 = np.array([to_goal_norm[1], -to_goal_norm[0]])

    # Also try some angled directions
    angles = [45, -45, 90, -90, 135, -135]  # degrees
    directions = []

    for angle in angles:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotated = np.array(
            [
                cos_a * to_goal_norm[0] - sin_a * to_goal_norm[1],
                sin_a * to_goal_norm[0] + cos_a * to_goal_norm[1],
            ]
        )
        directions.append(rotated)

    directions.extend([perp1, perp2])

    best_point = current_pos
    best_score = -np.inf

    for direction in directions:
        for distance in [2.0, 3.0, 4.0, max_distance]:
            candidate = Point(
                current_pos.x + direction[0] * distance,
                current_pos.y + direction[1] * distance,
            )

            if polygon.contains(candidate):
                # Check if it's away from forbidden zones
                valid = True
                for forbidden_center, forbidden_radius in forbidden_zones[agent_id]:
                    if candidate.distance(forbidden_center) < forbidden_radius:
                        valid = False
                        break

                if valid:
                    # Score based on distance from current and not being in recent memory
                    score = distance
                    if any(
                        candidate.distance(pos) < 2.0
                        for pos in recent_positions[agent_id]
                    ):
                        score *= 0.5

                    if score > best_score:
                        best_score = score
                        best_point = candidate

    return best_point


def validate_and_adjust_target(
    best_point: Point,
    current_pos: Point,
    polygon: Polygon,
    step_back_distance: float = 0.2,
    max_attempts: int = 5,
) -> Tuple[Point, AdjustmentInfo]:
    """
    Adjust target points that are on boundaries (walls) to be inside the walkable area.

    This is NORMAL behavior - isovist boundary points are always on geometry boundaries,
    but JuPedSim requires targets to be inside the walkable area, not on walls.

    Args:
        best_point: The computed target point (likely on a wall)
        current_pos: Current agent position
        polygon: The walkable area polygon
        step_back_distance: Distance to step back from walls (meters)
        max_attempts: Maximum number of adjustment attempts

    Returns:
        Tuple of (adjusted_point, adjustment_info)
    """
    adjustment_info = AdjustmentInfo()

    # If the point is already inside (rare case), return it
    if polygon.contains(best_point):
        return best_point, adjustment_info

    # This is the normal case - step back from the wall
    adjustment_info.was_adjusted = True

    # Calculate direction from wall point back toward current position
    direction = np.array([current_pos.x - best_point.x, current_pos.y - best_point.y])
    norm = np.linalg.norm(direction)

    if norm < 0.01:  # Points are essentially the same
        adjustment_info.adjustment_type = "stuck_at_current"
        return current_pos, adjustment_info

    # Normalize direction vector
    direction /= norm

    # Try stepping back from the wall toward current position
    for attempt in range(max_attempts):
        step_distance = step_back_distance * (attempt + 1)
        adjusted = Point(
            best_point.x + direction[0] * step_distance,
            best_point.y + direction[1] * step_distance,
        )

        if polygon.contains(adjusted):
            adjustment_info.adjustment_type = "stepped_back"
            adjustment_info.attempts = attempt + 1
            adjustment_info.final_distance_from_original = step_distance
            return adjusted, adjustment_info

    # If stepping back didn't work, fallback to current position
    adjustment_info.adjustment_type = "fallback_to_current"
    return current_pos, adjustment_info


def compute_next_step(
    current: Point,
    goal: Point,
    polygon: Polygon,
    space_syntax_data: SpaceSyntaxData,
    agent_id,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Point, Optional[Dict]]:
    """
    Compute next step using isovist-based candidate selection and Space Syntax scoring.

    Combines directional alignment, progress toward goal, and configurable space syntax metrics.

    Args:
        current: Current agent position
        goal: Target goal position
        polygon: Walkable area polygon
        space_syntax_data: Space syntax data handler
        weights: Dictionary of metric weights

    Returns:
        Tuple of (next_target_point, best_point_data)
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
            typer.echo(f"⚠ Error loading isovist from CSV: {e}")

    if not isovist:
        return current, None

    # Check if goal is directly visible within the isovist
    if isovist.contains(goal):
        return goal, None

    # Calculate scores for all isovist boundary points
    current_dist = current.distance(goal)
    best_point = current
    best_data = None

    # Check if agent is stuck
    if is_agent_stuck(agent_id, current, current_dist):
        exploration_mode[agent_id] = True
        print(
            f"🔄 Agent {agent_id} stuck (count: {stuck_counter[agent_id]}). Entering exploration mode."
        )

        # For heavily stuck agents, try escape direction
        if stuck_counter[agent_id] > 0:
            print(f"🚨 Agent {agent_id} heavily stuck. Trying escape direction.")
            escape_point = find_best_escape_direction(current, goal, polygon, agent_id)
            if escape_point != current:
                return escape_point, None
    else:
        exploration_mode[agent_id] = False

    # Direction to goal for alignment calculation
    direction_to_goal = np.array([goal.x - current.x, goal.y - current.y])
    direction_to_goal_norm = np.linalg.norm(direction_to_goal)

    if direction_to_goal_norm > 0:
        direction_to_goal /= direction_to_goal_norm

    candidates_with_scores = []
    coords = np.array(list(isovist.exterior.coords))  # shape (N,2)
    dx = coords[:, 0] - current.x
    dy = coords[:, 1] - current.y
    ray_lengths = np.hypot(dx, dy)
    max_ray_length = float(ray_lengths.max()) if ray_lengths.size else 0.0
    for x, y in isovist.exterior.coords:
        candidate = Point(x, y)
        # Calculate directional alignment score
        direction_to_candidate = np.array([x - current.x, y - current.y])
        dist_to_candidate = np.linalg.norm(direction_to_candidate)

        if dist_to_candidate < 1e-6:
            continue

        # Normalize direction vector
        direction_to_candidate /= dist_to_candidate
        # Get Space Syntax metrics from CSV for this candidate point
        candidate_data = space_syntax_data.get_nearest_point_data(candidate)
        # Exploration bonus when stuck

        score = calculate_score(
            current,
            goal,
            current_dist,
            direction_to_goal,
            candidate,
            candidate_data.iloc[0] if not candidate_data.empty else pd.Series(),
            weights,
            grid_usage,
            max_ray_length,
        )
        candidates_with_scores.append((candidate, score))

    best_point = choose_stochastic_best(candidates_with_scores, agent_id)
    best_data = space_syntax_data.get_nearest_point_data(best_point)
    return best_point, best_data


def initialize_simulation(
    model: jps.CollisionFreeSpeedModel,
    output_file: str,
    geometry: pedpy.WalkableArea,
    exit_polygon: Polygon,
) -> Tuple[jps.Simulation, int, int, int, int]:
    """Initialize the JuPedSim simulation with required stages and journeys."""
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
    simulation: jps.Simulation,
    direct_journey_id: int,
    exit_journey_id: int,
    stage_id: int,
    exit_id: int,
    agent_positions: List[Tuple[float, float]],
    agent_parameters: type,
    num_space_syntax_agents: int = 1,
) -> None:
    """Add agents to the simulation with specified behavior types."""
    for i, pos in enumerate(agent_positions):
        if i < num_space_syntax_agents:  # First N agents use space syntax guidance
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


def main(
    csv_file: str = typer.Option(
        "metrics_gallery_grid_fixed.csv",
        "--csv-file",
        "-c",
        help="Path to Space Syntax CSV file",
    ),
    geometry_file: str = typer.Option(
        "files/gallery.wkt", "--geometry", "-g", help="Path to WKT geometry file"
    ),
    output_file: str = typer.Option(
        "traj.sqlite", "--output", "-o", help="Output trajectory file"
    ),
    output_dir: str = typer.Option(
        "output", "--output-dir", "-d", help="Directory for visualization output"
    ),
    max_iterations: int = typer.Option(
        10000, "--max-iter", "-i", help="Maximum simulation iterations"
    ),
    update_freq: int = typer.Option(
        200, "--update-freq", "-v", help="Update frequency (every N steps)"
    ),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Total number of agents"),
    space_syntax_agents: int = typer.Option(
        1, "--ss-agents", "-s", help="Number of agents using space syntax guidance"
    ),
    connectivity_weight: float = typer.Option(
        0.3, "--w-conn", help="Weight for connectivity metric"
    ),
    integration_weight: float = typer.Option(
        0.3, "--w-int", help="Weight for integration metric"
    ),
    through_vision_weight: float = typer.Option(
        0.3, "--w-vision", help="Weight for through vision metric"
    ),
    depth_penalty_weight: float = typer.Option(
        0.1, "--w-depth", help="Weight for depth penalty metric"
    ),
    goal_threshold: float = typer.Option(
        0.5, "--goal-threshold", help="Distance threshold for goal achievement"
    ),
) -> None:
    """
    Run enhanced pedestrian simulation with Space Syntax guidance.

    This simulation uses pre-computed Space Syntax metrics to guide agent movement
    through complex environments, combining spatial analysis with pedestrian dynamics.
    """

    typer.echo("🚀 Starting Enhanced Space Syntax Pedestrian Simulation")

    # Validate inputs
    if space_syntax_agents > num_agents:
        typer.echo(
            "Error: Number of space syntax agents cannot exceed total agents", err=True
        )
        raise typer.Exit(1)

    # Load Space Syntax data
    try:
        space_syntax_data = SpaceSyntaxData(csv_file)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error loading Space Syntax data: {e}", err=True)
        raise typer.Exit(1)

    # Load geometry
    try:
        with open(geometry_file, "r") as f:
            geometry = wkt.loads(f.read())
        walkable_area = pedpy.WalkableArea(geometry.geoms[0])
        polygon = walkable_area.polygon
        typer.echo(f"✓ Loaded geometry from {geometry_file}")
    except (FileNotFoundError, IndexError) as e:
        typer.echo(f"Error loading geometry: {e}", err=True)
        raise typer.Exit(1)

    # Define areas (these could be made configurable in future versions)
    exit_polygon = Polygon([[21, 12], [24, 12], [24, 10], [21, 10]])
    spawning_area = Polygon([[-16, -1], [-12, -1], [-12, -4], [-16, -4]])

    # Initialize simulation
    collision_free_model = jps.CollisionFreeSpeedModel()
    simulation, direct_journey_id, exit_journey_id, stage_id, exit_id = (
        initialize_simulation(
            collision_free_model, output_file, walkable_area, exit_polygon
        )
    )

    # Distribute agents
    pos_in_spawning_area = jps.distribute_by_number(
        polygon=spawning_area,
        number_of_agents=num_agents,
        distance_to_agents=0.30,
        distance_to_polygon=0.15,
        seed=42,  # Fixed seed for reproducibility
    )

    add_agents(
        simulation=simulation,
        direct_journey_id=direct_journey_id,
        exit_journey_id=exit_journey_id,
        stage_id=stage_id,
        exit_id=exit_id,
        agent_positions=pos_in_spawning_area,
        agent_parameters=jps.CollisionFreeSpeedModelAgentParameters,
        num_space_syntax_agents=space_syntax_agents,
    )

    # Simulation parameters
    goal = exit_polygon.centroid
    space_syntax_weights = {
        "connectivity": connectivity_weight,
        "integration": integration_weight,
        "through_vision": through_vision_weight,
        "depth_penalty": depth_penalty_weight,
    }

    typer.echo(f"🎯 Goal position: ({goal.x:.2f}, {goal.y:.2f})")
    typer.echo(f"🔧 Space Syntax weights: {space_syntax_weights}")
    typer.echo(
        f"👥 Agents: {num_agents} total, {space_syntax_agents} using space syntax"
    )

    # Main simulation loop
    typer.echo("\n🏃 Starting simulation...")

    while (
        simulation.agent_count() > 0 and simulation.iteration_count() < max_iterations
    ):
        step = simulation.iteration_count()
        for cell in list(grid_usage):
            grid_usage[cell] = max(0, grid_usage[cell] - 1)

        for i, agent in enumerate(simulation.agents()):
            grid_data = space_syntax_data.get_nearest_point_data(Point(agent.position))
            ci, cj = grid_data["grid_i"].iloc[0], grid_data["grid_j"].iloc[0]
            # Increment that cell's usage
            grid_usage[(ci, cj)] += 1

            if (
                i < space_syntax_agents and step % update_freq == 0
            ):  # Space syntax guided agents
                current_pos = Point(agent.position)
                agent_id = agent.id
                # Compute next step using Space Syntax data
                best_point, best_data = compute_next_step(
                    current_pos,
                    goal,
                    polygon,
                    space_syntax_data,
                    agent_id,
                    space_syntax_weights,
                )

                # Enhanced wall-stepping mechanism (this is normal behavior)
                adjusted_point, adjustment_info = validate_and_adjust_target(
                    best_point, current_pos, polygon
                )
                best_point = adjusted_point

                # Set agent target
                agent.target = (best_point.x, best_point.y)

                # Visualization and logging
                agent_dir = os.path.join(output_dir, f"agent_{agent_id:02d}")
                os.makedirs(agent_dir, exist_ok=True)
                plot_isovist_from_csv(
                    polygon,
                    current_pos,
                    goal,
                    best_point,
                    step,
                    space_syntax_data,
                    adjustment_info=adjustment_info,
                    output_dir=agent_dir,
                )

                # Log detailed information
                current_dist = current_pos.distance(goal)

                if best_data is not None and not best_data.empty:
                    space_score = best_data["score"].iloc[0]
                    norm_conn = best_data["norm_connectivity"].iloc[0]
                    norm_int = best_data["norm_integration"].iloc[0]
                    norm_through = best_data["norm_through_vision"].iloc[0]
                    norm_depth = best_data["norm_depth_penalty"].iloc[0]
                    iso_area = best_data["isovist_area"].iloc[0]

                else:
                    space_score = 0
                    norm_conn = 0
                    norm_int = 0
                    norm_through = 0
                    norm_depth = 0
                    iso_area = 0.0

                typer.echo(f"\n📍 Step {step} - Agent {i}")
                typer.echo(f"   Position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                typer.echo(f"   Target: ({best_point.x:.2f}, {best_point.y:.2f})")
                typer.echo(f"   Distance to goal: {current_dist:.2f}m")
                typer.echo(f"   Space score: {space_score:.3f}")
                typer.echo(f"   Norm connectivity:   {norm_conn:.3f}")
                typer.echo(f"   Norm integration:    {norm_int:.3f}")
                typer.echo(f"   Norm through-vision: {norm_through:.3f}")
                typer.echo(f"   Norm depth-penalty:  {norm_depth:.3f}")
                typer.echo(f"   Isovist area:        {iso_area:.3f}")
                # if adjustment_info.was_adjusted:
                #     if adjustment_info.adjustment_type == "stepped_back":
                #         typer.echo(
                #             f"   → Normal wall stepping ({adjustment_info.final_distance_from_original:.3f}m)"
                #         )
                #     else:
                #         typer.echo(
                #             f"   ⚠ Special case: {adjustment_info.adjustment_type}"
                #         )

            # Check if goal is reached
            if is_goal_reached(current_pos, goal, goal_threshold):
                simulation.mark_agent_for_removal(agent.id)
                typer.echo(f"✅ Agent {i} reached the goal at step {step}!")

        simulation.iterate()

    # Simulation completed
    typer.echo("\n🏁 Simulation completed!")
    typer.echo(f"⏱️  Total evacuation time: {simulation.elapsed_time():.2f} seconds")
    typer.echo(f"🔄 Total iterations: {simulation.iteration_count()}")
    typer.echo(f"📊 Trajectory saved to: {output_file}")
    typer.echo(f"🖼️  Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    typer.run(main)
