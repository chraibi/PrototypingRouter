from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np
import pedpy
import pathlib
import jupedsim as jps


from jupedsim.internal.notebook_utils import animate, read_sqlite_file


def cast_rays(polygon: Polygon, origin: Point, n_rays=720):
    """
    Cast rays from origin point and find intersections with polygon boundaries.

    including both exterior and interior boundaries (obstacles).
    """
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


def plot_isovist(
    env_polygon: Polygon,
    isovist: Polygon,
    origin: Point,
    goal: Point,
    best_point: Point,
    step: int,
):
    """Plot the environment polygon with obstacles and the computed isovist."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot exterior boundary
    ax.plot(*env_polygon.exterior.xy, color="black", linewidth=2, label="")

    for interior in env_polygon.interiors:
        ax.fill(*interior.xy, color="gray", alpha=0.7)
        ax.plot(*interior.xy, color="black", linewidth=1.5)

    if isovist:
        ax.fill(*isovist.exterior.xy, color="skyblue", alpha=0.5, label="Isovist")
        ax.plot(*isovist.exterior.xy, color="blue", linewidth=1)

    ax.plot(origin.x, origin.y, "ro", markersize=8, label="Agent")
    ax.plot(best_point.x, best_point.y, "go", markersize=8, label="Next")
    ax.plot(goal.x, goal.y, "bx", markersize=8, label="Goal")

    ax.set_aspect("equal")
    ax.legend()
    ax.grid(False, alpha=0.3)
    dist = origin.distance(goal)
    ax.set_title(f"Step {step} – Distance to goal: {dist:.2f} m")

    plt.savefig(f"isovist_step_{step:03d}.png")


def is_goal_reached(current, goal, threshold=0.5):
    return current.distance(goal) < threshold


def compute_next_step(current, goal, polygon):
    isovist = cast_rays(polygon, current)
    if isovist.contains(goal):
        return goal, isovist

    if not isovist:
        return current, None

    # Prioritize reducing remaining distance to goal
    current_dist = current.distance(goal)

    best_score = -np.inf
    best_point = current

    for x, y in isovist.exterior.coords:
        candidate = Point(x, y)
        new_dist = candidate.distance(goal)

        # Calculate directional alignment score
        direction_to_goal = np.array([goal.x - current.x, goal.y - current.y])
        direction_to_candidate = np.array([x - current.x, y - current.y])

        if np.linalg.norm(direction_to_candidate) < 1e-6:
            continue

        # Normalize vectors
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        direction_to_candidate /= np.linalg.norm(direction_to_candidate)

        # Combined score: alignment + progress incentive
        alignment_score = np.dot(direction_to_goal, direction_to_candidate)
        progress_score = (current_dist - new_dist) * 2  # Weighted progress
        total_score = alignment_score + progress_score

        if total_score > best_score:
            best_score = total_score
            best_point = candidate

    return best_point, isovist


def initialize_simulation(model, output_file, geometry, exit_polygon):
    simulation = jps.Simulation(
        model=model,
        geometry=geometry.polygon,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(output_file), every_nth_frame=1
        ),
    )
    exit_id = simulation.add_exit_stage(exit_polygon)
    journey = jps.JourneyDescription([exit_id])
    journey_id = simulation.add_journey(journey)
    return simulation, journey_id, exit_id


def add_agents(
    simulation,
    journey_id,
    exit_id,
    agent_positions,
    agent_parameters,
):
    for pos in agent_positions:
        simulation.add_agent(
            agent_parameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=pos,
                desired_speed=1.0,
                radius=0.15,
            )
        )


def run_simulation(simulation, output_file, max_iterations=5000):
    while (
        simulation.agent_count() > 0 and simulation.iteration_count() < max_iterations
    ):
        simulation.iterate()
    print(f"Evacuation time: {simulation.elapsed_time():.2f} s")
    trajectory_data, walkable_area = read_sqlite_file(output_file)
    anim = animate(trajectory_data, walkable_area, every_nth_frame=50)
    return anim, simulation.elapsed_time()


if __name__ == "__main__":
    # Load geometry from file or use test polygon
    with open("files/gallery.wkt", "r") as f:
        geometry = wkt.loads(f.read())
    walkable_area = pedpy.WalkableArea(geometry.geoms[0])
    polygon = walkable_area.polygon
    exit_polygon = Polygon([[21, 12], [24, 12], [24, 10], [21, 10]])
    spawning_area = Polygon([-16, -1], [-12, -1], [-12, -4], [-16, -4])
    # ----- init simulation -------
    output_file = "traj.sqlite"
    collision_free_model = jps.CollisionFreeSpeedModel()
    simulation, journey_id, exit_id = initialize_simulation(
        collision_free_model, output_file, walkable_area, exit_polygon
    )

    pos_in_spawning_area = jps.distribute_by_number(
        polygon=spawning_area,
        number_of_agents=1,
        distance_to_agents=0.30,
        distance_to_polygon=0.15,
        seed=1,
    )
    add_agents(
        simulation,
        journey_id=journey_id,
        exit_id=exit_id,
        agent_positions=pos_in_spawning_area,
        agent_parameters=jps.CollisionFreeSpeedModelAgentParameters,
    )
    anim, time_elapsed = run_simulation(simulation, output_file, max_iterations=5000)
    # ---------------------------------
    viewpoint = Point(5, 5)  # Center point for test polygon
    goal = Point(15, 11)
    # Compute isovist
    # isovist = cast_rays(polygon, viewpoint)
    steps = 15
    speed = 1.0  # meters per step
    step = 0
    while step <= steps:
        step += 1
        best_point, isovist = compute_next_step(viewpoint, goal, polygon)
        direction = np.array([best_point.x - viewpoint.x, best_point.y - viewpoint.y])
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            print("Stuck: best point is current location.")
            break

        if isovist:
            print(step)
            print(f"Isovist area: {isovist.area:.2f}")
            print(f"Isovist perimeter: {isovist.length:.2f}")
            plot_isovist(polygon, isovist, viewpoint, goal, best_point, step)

        else:
            print("No valid isovist computed")

        step_fraction = min(speed, distance) / distance
        viewpoint = Point(
            viewpoint.x + direction[0] * step_fraction,
            viewpoint.y + direction[1] * step_fraction,
        )
        print(f"Distance to goal: {viewpoint.distance(goal)}")

        if is_goal_reached(viewpoint, goal, threshold=0.1):
            print("🎯 Goal reached!")
            break
