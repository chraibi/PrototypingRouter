import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
import io
import time

# Set page config
st.set_page_config(
    page_title="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_polygon_from_wkt(wkt_content):
    """Load polygon from WKT content"""
    try:
        import pedpy

        geometry = wkt.loads(wkt_content)
        walkable_area = pedpy.WalkableArea(geometry.geoms[0])
        polygon = walkable_area.polygon
        return polygon
    except Exception as e:
        st.error(f"Error loading WKT file: {e}")
        return None


def create_test_polygon():
    """Create a test polygon with obstacles for demo purposes"""
    exterior = [(0, 0), (20, 0), (20, 15), (0, 15)]
    obstacle1 = [(3, 3), (7, 3), (7, 7), (3, 7)]
    obstacle2 = [(12, 8), (16, 8), (16, 12), (12, 12)]
    obstacle3 = [(2, 10), (5, 10), (5, 13), (2, 13)]
    return Polygon(exterior, [obstacle1, obstacle2, obstacle3])


def calculate_weighted_score(df, weights):
    """Calculate weighted score based on normalized metrics from CSV"""
    # Normalize metrics to 0-1 range
    metrics_to_normalize = [
        "connectivity",
        "visual_connectivity",
        "integration",
        "through_vision",
        "mean_depth",
    ]

    normalized_data = df.copy()

    # Normalize each metric
    if "connectivity" in df.columns:
        max_connectivity = df["connectivity"].max()
        if max_connectivity > 0:
            normalized_data["norm_connectivity"] = df["connectivity"] / max_connectivity
        else:
            normalized_data["norm_connectivity"] = 0
    else:
        normalized_data["norm_connectivity"] = 0

    if "visual_connectivity" in df.columns:
        max_connectivity = df["visual_connectivity"].max()
        if max_connectivity > 0:
            normalized_data["norm_connectivity"] = (
                df["visual_connectivity"] / max_connectivity
            )
        else:
            normalized_data["norm_connectivity"] = 0
    else:
        normalized_data["norm_connectivity"] = 0

    if "integration" in df.columns:
        # Integration is already between 0-1 typically
        normalized_data["norm_integration"] = df["integration"].clip(0, 1)
    else:
        normalized_data["norm_integration"] = 0

    if "through_vision" in df.columns:
        # Normalize through vision (cap at reasonable maximum)
        max_vision = df["through_vision"].quantile(
            0.95
        )  # Use 95th percentile to avoid outliers
        if max_vision > 0:
            normalized_data["norm_through_vision"] = (
                df["through_vision"] / max_vision
            ).clip(0, 1)
        else:
            normalized_data["norm_through_vision"] = 0
    else:
        normalized_data["norm_through_vision"] = 0

    if "mean_depth" in df.columns:
        # Depth penalty - normalize and invert (lower depth = better)
        max_depth = df["mean_depth"].quantile(0.95)  # Use 95th percentile
        if max_depth > 0:
            normalized_data["norm_depth_penalty"] = (df["mean_depth"] / max_depth).clip(
                0, 1
            )
        else:
            normalized_data["norm_depth_penalty"] = 0
    else:
        normalized_data["norm_depth_penalty"] = 0

    # Calculate weighted score
    score = (
        weights["connectivity"] * normalized_data["norm_connectivity"]
        + weights["integration"] * normalized_data["norm_integration"]
        + weights["through_vision"] * normalized_data["norm_through_vision"]
        - weights["depth_penalty"] * normalized_data["norm_depth_penalty"]
    )

    normalized_data["score"] = score

    return normalized_data


def create_visibility_graph_from_csv(df, max_distance=5.0):
    """Create a simple visibility graph from CSV coordinates"""
    graph = nx.Graph()
    points = [Point(row["x"], row["y"]) for _, row in df.iterrows()]

    # Add nodes
    for i, point in enumerate(points):
        graph.add_node(i, pos=(point.x, point.y), point=point)

    # Add edges for points within max distance
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            distance = p1.distance(p2)

            if distance <= max_distance:
                graph.add_edge(i, j, weight=distance)

    return graph, points


def find_optimal_route_from_points(start_point, end_point, points, df, graph):
    """Find optimal route between two clicked points"""
    # Find nearest grid points to clicked locations
    start_distances = [
        Point(row["x"], row["y"]).distance(start_point) for _, row in df.iterrows()
    ]
    end_distances = [
        Point(row["x"], row["y"]).distance(end_point) for _, row in df.iterrows()
    ]

    start_idx = np.argmin(start_distances)
    end_idx = np.argmin(end_distances)

    try:
        # Find shortest path in graph
        path_indices = nx.shortest_path(graph, start_idx, end_idx)

        # Get route points and scores
        route_points = [points[i] for i in path_indices]
        route_scores = [df.iloc[i]["score"] for i in path_indices]
        route_quality = np.mean(route_scores)

        # Calculate total distance
        total_distance = sum(
            points[path_indices[i]].distance(points[path_indices[i + 1]])
            for i in range(len(path_indices) - 1)
        )

        return {
            "path_indices": path_indices,
            "route_points": route_points,
            "route_quality": route_quality,
            "total_distance": total_distance,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }
    except nx.NetworkXNoPath:
        return None


def create_interactive_plot(df, metric, title, colorscale="Viridis", polygon=None):
    """Create an interactive plotly scatter plot"""
    fig = go.Figure()

    # Add polygon if provided
    if polygon is not None:
        # Add exterior boundary
        x_ext, y_ext = polygon.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=list(x_ext),
                y=list(y_ext),
                mode="lines",
                line=dict(color="black", width=2),
                name="Boundary",
                showlegend=False,
            )
        )

        # Add interior obstacles
        for interior in polygon.interiors:
            x_int, y_int = interior.xy
            fig.add_trace(
                go.Scatter(
                    x=list(x_int),
                    y=list(y_int),
                    mode="lines",
                    fill="tonext",
                    fillcolor="rgba(128,128,128,0.7)",
                    line=dict(color="black", width=1),
                    name="Obstacle",
                    showlegend=False,
                )
            )

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=8,
                color=df[metric],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=metric.replace("_", " ").title()),
                symbol="square",
            ),
            text=[
                f"({x:.2f}, {y:.2f})<br>{metric}: {val:.3f}"
                for x, y, val in zip(df["x"], df["y"], df[metric])
            ],
            hovertemplate="%{text}<extra></extra>",
            name=title,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=600,
        height=500,
        showlegend=False,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def validate_csv_format(df):
    """Validate that the CSV has required columns"""
    required_columns = ["x", "y"]
    optional_columns = [
        "connectivity",
        "visual_connectivity",
        "integration",
        "through_vision",
        "mean_depth",
    ]

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        return False, f"Missing required columns: {missing_required}"

    available_metrics = [col for col in optional_columns if col in df.columns]
    if not available_metrics:
        return (
            False,
            f"No metric columns found. Expected at least one of: {optional_columns}",
        )

    return True, f"Found metrics: {available_metrics}"


def main():
    # Sidebar for settings and file uploads
    with st.sidebar:
        st.header("📁 Data Input")

        # CSV file upload
        uploaded_csv = st.file_uploader(
            "Upload CSV file with metrics",
            type=["csv"],
            help="CSV should contain: x, y (required) and any of: connectivity, integration, through_vision, mean_depth",
        )

        # Optional WKT file for visualization
        uploaded_wkt = st.file_uploader(
            "Upload WKT file (optional)",
            type=["wkt"],
            help="Upload WKT file for polygon visualization overlay",
        )

        polygon = None
        if uploaded_wkt is not None:
            wkt_content = uploaded_wkt.read().decode("utf-8")
            polygon = load_polygon_from_wkt(wkt_content)
            if polygon:
                st.success("✅ Loaded polygon from WKT file")
        else:
            # Option to use test polygon
            if st.checkbox("Show test polygon overlay", value=False):
                polygon = create_test_polygon()

    # Main content area
    if uploaded_csv is not None:
        try:
            # Load CSV
            original_df = pd.read_csv(uploaded_csv)

            # Validate format
            is_valid, message = validate_csv_format(original_df)
            if not is_valid:
                st.error(f"❌ Invalid CSV format: {message}")
                return
            else:
                st.sidebar.success(
                    f"✅ Loaded {len(original_df)} data points. {message}"
                )

            # Show available metrics
            available_metrics = []
            metric_info = {}

            if "connectivity" in original_df.columns:
                available_metrics.append("connectivity")
                metric_info["connectivity"] = (
                    f"Range: {original_df['connectivity'].min():.2f} - {original_df['connectivity'].max():.2f}"
                )
            if "visual_connectivity" in original_df.columns:
                available_metrics.append("visual_connectivity")
                metric_info["visual_connectivity"] = (
                    f"Range: {original_df['visual_connectivity'].min():.2f} - {original_df['visual_connectivity'].max():.2f}"
                )

            if "integration" in original_df.columns:
                available_metrics.append("integration")
                metric_info["integration"] = (
                    f"Range: {original_df['integration'].min():.3f} - {original_df['integration'].max():.3f}"
                )

            if "through_vision" in original_df.columns:
                available_metrics.append("through_vision")
                metric_info["through_vision"] = (
                    f"Range: {original_df['through_vision'].min():.2f} - {original_df['through_vision'].max():.2f}"
                )

            if "mean_depth" in original_df.columns:
                available_metrics.append("mean_depth")
                metric_info["mean_depth"] = (
                    f"Range: {original_df['mean_depth'].min():.2f} - {original_df['mean_depth'].max():.2f}"
                )

            # Weight settings in sidebar
            with st.sidebar:
                st.header("⚖️ Scoring Weights")
                st.markdown("Adjust weights for each available metric:")

                weights = {}
                if "visual_connectivity" in available_metrics:
                    weights["visual_connectivity"] = st.slider(
                        "Connectivity Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.25,
                        step=0.05,
                        help="Higher connectivity = more direct connections",
                    )
                    st.caption(metric_info["visual_connectivity"])
                if "connectivity" in available_metrics:
                    weights["connectivity"] = st.slider(
                        "Connectivity Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.25,
                        step=0.05,
                        help="Higher connectivity = more direct connections",
                    )
                    st.caption(metric_info["connectivity"])
                else:
                    weights["connectivity"] = 0.0

                if "integration" in available_metrics:
                    weights["integration"] = st.slider(
                        "Integration Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.25,
                        step=0.05,
                        help="Higher integration = more accessible from all points",
                    )
                    st.caption(metric_info["integration"])
                else:
                    weights["integration"] = 0.0

                if "through_vision" in available_metrics:
                    weights["through_vision"] = st.slider(
                        "Through Vision Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.25,
                        step=0.05,
                        help="Higher through vision = longer sight lines",
                    )
                    st.caption(metric_info["through_vision"])
                else:
                    weights["through_vision"] = 0.0

                if "mean_depth" in available_metrics:
                    weights["depth_penalty"] = st.slider(
                        "Depth Penalty Weight",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.25,
                        step=0.05,
                        help="Higher penalty = avoid deeper (less accessible) locations",
                    )
                    st.caption(metric_info["mean_depth"])
                else:
                    weights["depth_penalty"] = 0.0

                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
                    st.markdown(
                        f"**Normalized weights sum:** {sum(weights.values()):.3f}"
                    )
                else:
                    st.warning("⚠️ All weights are zero!")

                # Additional settings
                st.header("🔧 Analysis Settings")
                max_distance = st.slider(
                    "Max route distance",
                    min_value=1.0,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    help="Maximum distance for route connections",
                )

            # Calculate weighted scores
            if total_weight > 0:
                df = calculate_weighted_score(original_df, weights)

                # Create visibility graph for routing
                visibility_graph, points = create_visibility_graph_from_csv(
                    df, max_distance
                )

                # Statistics overview
                with st.expander("📊 Analysis Overview", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Total Points", len(df))
                    with col2:
                        st.metric("Avg Score", f"{df['score'].mean():.3f}")
                    with col3:
                        st.metric("Max Score", f"{df['score'].max():.3f}")
                    with col4:
                        st.metric("Min Score", f"{df['score'].min():.3f}")
                    with col5:
                        st.metric(
                            "Score Range",
                            f"{df['score'].max() - df['score'].min():.3f}",
                        )

                # Visualization tabs
                tab_names = ["🎯 Weighted Scores"]
                if "connectivity" in available_metrics:
                    tab_names.append("🔗 Connectivity")
                if "visual_connectivity" in available_metrics:
                    tab_names.append("🔗 Connectivity")

                if "integration" in available_metrics:
                    tab_names.append("🌐 Integration")
                if "through_vision" in available_metrics:
                    tab_names.append("👁️ Through Vision")

                tabs = st.tabs(tab_names)

                # Initialize session state for route finding
                if "start_point" not in st.session_state:
                    st.session_state.start_point = None
                if "end_point" not in st.session_state:
                    st.session_state.end_point = None

                # Weighted Scores tab
                with tabs[0]:
                    chart_container = st.empty()
                    fig_score = create_interactive_plot(
                        df, "score", "Weighted Route Scores", "RdYlGn", polygon
                    )
                    chart_container.plotly_chart(fig_score, use_container_width=True)

                    # Route finding interface
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Clear Route"):
                            st.session_state.start_point = None
                            st.session_state.end_point = None
                            st.rerun()

                    # Manual coordinate input
                    with st.expander("📍 Manual Coordinate Input"):
                        col1, col2 = st.columns(2)
                        with col1:
                            start_x = st.number_input(
                                "Start X", value=float(df["x"].min())
                            )
                            start_y = st.number_input(
                                "Start Y", value=float(df["y"].min())
                            )
                            if st.button("Set Start"):
                                st.session_state.start_point = Point(start_x, start_y)
                                st.success(
                                    f"Start point set to ({start_x:.2f}, {start_y:.2f})"
                                )

                        with col2:
                            end_x = st.number_input("End X", value=float(df["x"].max()))
                            end_y = st.number_input("End Y", value=float(df["y"].max()))
                            if st.button("Set End"):
                                st.session_state.end_point = Point(end_x, end_y)
                                st.success(
                                    f"End point set to ({end_x:.2f}, {end_y:.2f})"
                                )

                    # Route calculation and visualization
                    if st.session_state.start_point and st.session_state.end_point:
                        try:
                            route = find_optimal_route_from_points(
                                st.session_state.start_point,
                                st.session_state.end_point,
                                points,
                                df,
                                visibility_graph,
                            )

                            if route:
                                st.success("🛤️ Optimal Route Found!")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Route Quality", f"{route['route_quality']:.3f}"
                                    )
                                with col2:
                                    st.metric(
                                        "Total Distance",
                                        f"{route['total_distance']:.2f}",
                                    )
                                with col3:
                                    st.metric("Waypoints", len(route["route_points"]))

                                # Add route to plot
                                route_x = [p.x for p in route["route_points"]]
                                route_y = [p.y for p in route["route_points"]]

                                fig_score.add_trace(
                                    go.Scatter(
                                        x=route_x,
                                        y=route_y,
                                        mode="lines+markers",
                                        line=dict(color="red", width=3),
                                        marker=dict(size=10, color="red"),
                                        name="Optimal Route",
                                        showlegend=True,
                                    )
                                )
                                chart_container.plotly_chart(
                                    fig_score, use_container_width=True
                                )

                            else:
                                st.error(
                                    "❌ No route found between the selected points."
                                )

                        except Exception as e:
                            st.error(f"Error calculating route: {e}")

                # Individual metric tabs
                tab_idx = 1
                if "connectivity" in available_metrics:
                    with tabs[tab_idx]:
                        fig_conn = create_interactive_plot(
                            df, "connectivity", "Connectivity", "Blues", polygon
                        )
                        st.plotly_chart(fig_conn, use_container_width=True)
                        st.markdown(
                            "**Connectivity** measures how many other spaces are directly accessible from each point."
                        )
                    tab_idx += 1
                if "visual_connectivity" in available_metrics:
                    print("HHHHHH")
                    with tabs[tab_idx]:
                        fig_conn = create_interactive_plot(
                            df, "visual_connectivity", "Connectivity", "Blues", polygon
                        )
                        st.plotly_chart(fig_conn, use_container_width=True)
                        st.markdown(
                            "**Connectivity** measures how many other spaces are directly accessible from each point."
                        )
                    tab_idx += 1

                if "integration" in available_metrics:
                    with tabs[tab_idx]:
                        fig_int = create_interactive_plot(
                            df, "integration", "Integration", "Oranges", polygon
                        )
                        st.plotly_chart(fig_int, use_container_width=True)
                        st.markdown(
                            "**Integration** measures how easily accessible a space is from all other spaces."
                        )
                    tab_idx += 1

                if "through_vision" in available_metrics:
                    with tabs[tab_idx]:
                        fig_vision = create_interactive_plot(
                            df, "through_vision", "Through Vision", "Purples", polygon
                        )
                        st.plotly_chart(fig_vision, use_container_width=True)
                        st.markdown(
                            "**Through Vision** measures the longest unobstructed sight line from each point."
                        )

                # Data export and analysis
                with st.expander("📋 Data Analysis & Export"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Metric Correlations")
                        corr_columns = ["score"] + [
                            col for col in available_metrics if col != "mean_depth"
                        ]
                        if len(corr_columns) > 1:
                            correlation_matrix = df[corr_columns].corr()
                            st.dataframe(correlation_matrix.round(3))

                    with col2:
                        st.subheader("📈 Top Scoring Locations")
                        display_cols = ["x", "y", "score"] + available_metrics[
                            :2
                        ]  # Show top 2 metrics
                        top_locations = df.nlargest(5, "score")[display_cols]
                        st.dataframe(top_locations.round(3))

                    st.subheader("⚖️ Current Weight Settings")
                    weight_df = pd.DataFrame(
                        [
                            {
                                "Metric": k.replace("_", " ").title(),
                                "Weight": f"{v:.3f}",
                            }
                            for k, v in weights.items()
                            if v > 0
                        ]
                    )
                    st.dataframe(weight_df, use_container_width=True)

                    st.subheader("📋 Complete Data Table")
                    st.dataframe(df, use_container_width=True)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Results with Scores",
                            data=csv_buffer.getvalue(),
                            file_name=f"space_syntax_weighted_scores.csv",
                            mime="text/csv",
                        )

            else:
                st.warning(
                    "⚠️ Please set at least one weight above zero to calculate scores."
                )

        except Exception as e:
            st.error(f"❌ Error loading CSV file: {e}")
            st.info(
                "Make sure your CSV has 'x' and 'y' columns and at least one metric column."
            )

    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Getting Started

        ### 📊 CSV Upload Mode
        
        This streamlined version focuses on **fast, interactive scoring** of pre-calculated space syntax metrics.
        
        #### Required CSV Format:
        Your CSV file should contain:
        - **Required columns:** `x`, `y` (coordinates)
        - **Metric columns** (at least one): 
          - `connectivity` - number of direct connections
          - `integration` - accessibility from all points (0-1 scale)
          - `through_vision` - longest sight line distance  
          - `mean_depth` - average path length to all points

        #### Features:
        - ⚡ **Real-time scoring** - adjust weights and see results instantly
        - 🎯 **Route finding** - click points to find optimal paths
        - 📊 **Interactive visualizations** - explore all metrics
        - 📥 **Export results** - download weighted scores
        - 🗺️ **Polygon overlay** - optional WKT file for context

        #### Example CSV Structure:
        ```
        x,y,connectivity,integration,through_vision,mean_depth
        1.0,1.0,5,0.23,8.5,2.1
        2.0,1.0,3,0.18,6.2,2.8
        ...
        ```

        ### 📁 Upload Your Data
        Use the sidebar to upload your CSV file and start exploring!
        """)


if __name__ == "__main__":
    main()
