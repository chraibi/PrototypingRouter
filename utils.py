from shapely import Point, Polygon, wkt
import pandas as pd
from typing import Optional, Dict
import numpy as np
from scipy.spatial import cKDTree
import typer
import matplotlib.pyplot as plt
import pathlib


class AdjustmentInfo:
    """Information about target point adjustments made during unstick operations."""

    def __init__(self) -> None:
        self.was_adjusted: bool = False
        self.adjustment_type: Optional[str] = None
        self.attempts: int = 0
        self.final_distance_from_original: float = 0.0


class SpaceSyntaxData:
    """Handles pre-calculated Space Syntax data from CSV files with spatial indexing."""

    def __init__(self, csv_file_path: str) -> None:
        """
        Load and prepare space syntax data from CSV.

        Args:
            csv_file_path: Path to CSV file containing space syntax metrics

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            self.df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Space syntax CSV file not found: {csv_file_path}")

        # Validate required columns
        required_cols = ["x", "y"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # Create spatial index for fast nearest neighbor queries
        self.points = np.column_stack((self.df["x"].values, self.df["y"].values))
        self.kdtree = cKDTree(self.points)

        typer.echo(f"✓ Loaded {len(self.df)} space syntax points from {csv_file_path}")

    def get_nearest_point_data(
        self, point: Point, k_neighbors: int = 1
    ) -> pd.DataFrame:
        """
        Get data for the nearest point(s) to the given location.

        Args:
            point: Query point
            k_neighbors: Number of nearest neighbors to return

        Returns:
            DataFrame with nearest point(s) data
        """
        query_point = np.array([point.x, point.y])
        distances, indices = self.kdtree.query(query_point, k=k_neighbors)

        if k_neighbors == 1:
            return (
                self.df.iloc[[indices]]
                if isinstance(indices, int)
                else self.df.iloc[indices[:1]]
            )
        else:
            return self.df.iloc[indices]

    def get_points_within_radius(self, point: Point, radius: float) -> pd.DataFrame:
        """
        Get all points within a given radius of the query point.

        Args:
            point: Center point for radius search
            radius: Search radius in meters

        Returns:
            DataFrame with points within radius
        """
        query_point = np.array([point.x, point.y])
        indices = self.kdtree.query_ball_point(query_point, radius)
        return self.df.iloc[indices] if indices else pd.DataFrame()


def calculate_score(
    current: Point,
    goal: Point,
    current_dist: float,
    direction_to_goal: np.ndarray,
    candidate: Point,
    candidate_data: pd.Series,
    weights: Dict[str, float],
    grid_usage,
    max_ray_length,
    USAGE_THRESHOLD=10,
    USAGE_WEIGHT=2,
) -> float:
    """
    Compute the combined score for a candidate point:
    - directional alignment toward goal
    - progress toward goal
    - space syntax metrics
    - visibility distance to candidate
    """
    # Vector and distance to candidate
    dx, dy = candidate.x - current.x, candidate.y - current.y
    dist_to_candidate = np.hypot(dx, dy)
    if dist_to_candidate < 1e-6:
        return -np.inf

    dir_to_candidate = np.array([dx, dy]) / dist_to_candidate

    # Alignment score
    alignment_score = (
        np.dot(direction_to_goal, dir_to_candidate)
        if np.linalg.norm(direction_to_goal) > 0
        else 0.0
    )

    # Progress score
    new_dist = candidate.distance(goal)
    progress_score = (current_dist - new_dist) * 2

    # Space Syntax score
    if not candidate_data.empty:
        norm_conn = candidate_data.get("norm_connectivity", 0.0)
        norm_int = candidate_data.get("norm_integration", 0.0)
        norm_through = candidate_data.get("norm_through_vision", 0.0)
        norm_depth = candidate_data.get("norm_depth_penalty", 0.0)
        space_syntax_score = (
            weights["connectivity"] * norm_conn
            + weights["integration"] * norm_int
            + weights["through_vision"] * norm_through
            - weights["depth_penalty"] * norm_depth
        )
        ci, cj = candidate_data["grid_i"], candidate_data["grid_j"]
        used = grid_usage[(ci, cj)]
        if used > 2:
            diff = USAGE_THRESHOLD - used
        else:
            diff = 0
        usage_score = USAGE_WEIGHT * diff
    else:
        space_syntax_score = 0.0
        usage_score = 0

    # Combined total score
    w_dist = 0.0  # TODO should be in weights
    distance_score = w_dist * (dist_to_candidate) * max(0, alignment_score)
    total_score = (
        0.0 * alignment_score
        + progress_score
        + space_syntax_score
        + distance_score
        + usage_score
    )
    print("====================================")
    print(candidate)
    print("candidate data")
    print(f"{progress_score =}")
    print(candidate_data)
    print("total score")
    print(total_score)
    return total_score


def is_goal_reached(current: Point, goal: Point, threshold: float = 0.5) -> bool:
    """Check if the goal has been reached within the given threshold."""
    return current.distance(goal) < threshold


def plot_isovist_from_csv(
    env_polygon: Polygon,
    current_pos: Point,
    goal: Point,
    best_point: Point,
    best_score: float,
    step: int,
    space_syntax_data: SpaceSyntaxData,
    search_radius: float = 2.0,
    adjustment_info: Optional[AdjustmentInfo] = None,
    output_dir: str = "output",
) -> None:
    """Plot the environment with isovist data from CSV and adjustment information."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot exterior boundary
    ax.plot(*env_polygon.exterior.xy, color="black", linewidth=2)

    # Plot obstacles (interior boundaries)
    for interior in env_polygon.interiors:
        ax.fill(*interior.xy, color="gray", alpha=0.7)
        ax.plot(*interior.xy, color="black", linewidth=1.5)

    # Get current position data and plot isovist if available
    try:
        current_data = space_syntax_data.get_nearest_point_data(current_pos)

        if "isovist_wkt" in current_data.columns and pd.notna(
            current_data["isovist_wkt"].iloc[0]
        ):
            try:
                isovist = wkt.loads(current_data["isovist_wkt"].iloc[0])
                if isovist and hasattr(isovist, "exterior"):
                    ax.fill(
                        *isovist.exterior.xy,
                        color="skyblue",
                        alpha=0.5,
                        label="Current Isovist",
                    )
                    ax.plot(*isovist.exterior.xy, color="blue", linewidth=1)
            except Exception as e:
                typer.echo(f"⚠ Error plotting isovist: {e}")
    except Exception as e:
        typer.echo(f"⚠ Error accessing isovist data: {e}")

    # Plot nearby points colored by score
    nearby_points = space_syntax_data.get_nearest_point_data(
        current_pos, k_neighbors=32
    )

    # nearby_points = space_syntax_data.get_points_within_radius(
    #     current_pos, search_radius
    # )
    if not nearby_points.empty and "score" in nearby_points.columns:
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

    # Plot key points
    ax.plot(current_pos.x, current_pos.y, "ro", markersize=10, label="Agent", zorder=5)

    # Color the target point based on adjustment type
    if adjustment_info and adjustment_info.was_adjusted:
        color_map = {
            "stepped_back": ("rx", "Target"),  # This is normal
            "fallback_to_current": ("mo", "Fallback Target"),
            "fallback_towards_goal": ("co", "Fallback to Goal"),
            "stuck_at_current": ("yo", "Stuck Target"),
        }
        color, label = color_map.get(adjustment_info.adjustment_type, ("go", "Target"))
        ax.plot(best_point.x, best_point.y, color, markersize=10, label=label, zorder=5)
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

    # Build informative title
    try:
        current_data = space_syntax_data.get_nearest_point_data(current_pos)
        score = best_score
    except Exception as e:
        print(e)
        score = 0

    dist = current_pos.distance(goal)
    title = f"Step {step} – Distance to goal: {dist:.2f}m – Space Score: {score:.3f}"

    if adjustment_info and adjustment_info.was_adjusted:
        if adjustment_info.adjustment_type == "stepped_back":
            title += " – Normal wall stepping"
        else:
            title += f" – {adjustment_info.adjustment_type}"

    ax.set_title(title)
    plt.tight_layout()

    # Ensure output directory exists
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(
        f"{output_dir}/isovist_step_{step:05d}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
