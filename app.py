import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = [
    "urban_traffic_flow_original.csv",
    "urban_traffic_flow_modified.csv",
    "urban_traffic_flow_with_target.csv",
]

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Urban Traffic Clustering",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_LAYOUT = {
    "margin": {"l": 24, "r": 24, "t": 40, "b": 24}
}


def render_page_header(title, subtitle, icon=""):
    heading = f"{icon} {title}".strip() if icon else title
    st.title(heading)
    st.caption(subtitle)


def resolve_data_file():
    for filename in DATA_FILE_CANDIDATES:
        candidate = BASE_DIR / filename
        if candidate.exists():
            return candidate
    return None

# ----------------------------
# Load and Prepare Dataset (CACHED)
# ----------------------------
@st.cache_data
def load_and_prepare_data():
    data_file = resolve_data_file()

    if data_file is None:
        available_csvs = sorted([p.name for p in BASE_DIR.glob("*.csv")])
        raise FileNotFoundError(
            "No expected dataset file found. "
            f"Expected one of: {', '.join(DATA_FILE_CANDIDATES)}. "
            f"Available CSV files in app directory: {available_csvs}"
        )

    df = pd.read_csv(data_file)

    required_columns = {'Timestamp', 'Vehicle_Count', 'Vehicle_Speed'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df = df[['Vehicle_Count', 'Vehicle_Speed', 'Hour']].dropna()

    if df.empty:
        raise ValueError("No valid rows found after preprocessing. Check timestamp values and missing data.")

    if len(df) < 3:
        raise ValueError("At least 3 valid rows are required to train a 3-cluster model.")

    return df

@st.cache_resource
def train_kmeans_model(df, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    
    return kmeans, scaler, scaled_data


def build_cluster_mapping(kmeans_model):
    centers_scaled = kmeans_model.cluster_centers_
    congestion_scores = centers_scaled[:, 0] - centers_scaled[:, 1]
    ordered_clusters = np.argsort(congestion_scores)
    canonical_labels = ["Free Flow", "Moderate Traffic", "Heavy Congestion"]

    mapping = {}
    for rank, cluster_id in enumerate(ordered_clusters):
        cluster_key = int(cluster_id)
        if rank < len(canonical_labels):
            mapping[cluster_key] = canonical_labels[rank]
        else:
            mapping[cluster_key] = f"Traffic Cluster {rank + 1}"

    return mapping


@st.cache_data(show_spinner=False)
def compute_model_quality_metrics(scaled_values, cluster_labels):
    silhouette = silhouette_score(scaled_values, cluster_labels)
    db_score = davies_bouldin_score(scaled_values, cluster_labels)
    return silhouette, db_score


@st.cache_data(show_spinner=False)
def compute_elbow_inertias(scaled_values, k_start=2, k_end=10):
    inertias = []
    for k in range(k_start, k_end + 1):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(scaled_values)
        inertias.append(kmeans_temp.inertia_)
    return inertias

# Load data
try:
    df = load_and_prepare_data()
except Exception as e:
    st.error("Unable to load dataset. Ensure at least one expected CSV file is present in the app directory.")
    st.info(
        "Expected one of: "
        + ", ".join(DATA_FILE_CANDIDATES)
        + ". If you recently changed files, redeploy and refresh the app."
    )
    st.exception(e)
    st.stop()

# Train model with optimal clusters
optimal_clusters = 3
try:
    kmeans, scaler, scaled_data = train_kmeans_model(df, optimal_clusters)
except Exception as e:
    st.error("Model initialization failed. Check whether the dataset has sufficient valid rows.")
    st.exception(e)
    st.stop()

# Add cluster labels to dataframe
df['Cluster'] = kmeans.predict(scaled_data)

# Map clusters to traffic conditions dynamically (cluster IDs are not semantically fixed)
cluster_mapping = build_cluster_mapping(kmeans)
df['Traffic_Condition'] = df['Cluster'].map(cluster_mapping).fillna('Unknown')
    
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🚦 Urban Traffic Studio")
st.sidebar.caption("Interactive clustering and traffic intelligence dashboard")
page = st.sidebar.radio("Select Page", ["Home", "Dashboard", "Predictions", "Data Analysis", "Model Info", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, Plotly, and Scikit-learn")

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "Home":
    render_page_header(
        "Urban Traffic Intelligence",
        "Explore congestion patterns, run instant predictions, and monitor cluster quality from one workspace.",
        icon="🏙️",
    )

    silhouette, db_score = compute_model_quality_metrics(scaled_data, df['Cluster'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    with col3:
        st.metric("Davies-Bouldin", f"{db_score:.3f}")

    st.markdown("---")
    st.subheader("Get Started")
    st.write("Use Dashboard for live cluster summaries, Predictions for scenario checks, and Data Analysis for deeper visual exploration.")
    st.info("Tip: For best predictions, enter values near the observed dataset ranges shown in the Predictions page.")

# ----------------------------
# DASHBOARD PAGE
# ----------------------------
elif page == "Dashboard":
    render_page_header(
        "Traffic Flow Pattern Clustering",
        "Monitor traffic behavior, cluster characteristics, and model quality at a glance.",
    )
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df), delta=None)
    with col2:
        st.metric("Number of Clusters", optimal_clusters, delta=None)
    with col3:
        silhouette, db_score = compute_model_quality_metrics(scaled_data, df['Cluster'])
        st.metric("Silhouette Score", f"{silhouette:.3f}", delta=None)
    with col4:
        st.metric("Davies-Bouldin Score", f"{db_score:.3f}", delta=None)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[cluster_mapping[i] for i in cluster_counts.index],
            hole=0.3
        )
        fig.update_layout(height=400, showlegend=True, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Traffic Conditions Count")
        condition_counts = df['Traffic_Condition'].value_counts()
        fig = px.bar(
            x=condition_counts.index,
            y=condition_counts.values,
            color=condition_counts.index
        )
        fig.update_layout(height=400, showlegend=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.markdown("---")
    st.subheader("📍 Cluster Characteristics")
    
    for cluster_id in sorted(df['Cluster'].unique()):
        with st.expander(f"**{cluster_mapping[cluster_id]}** (Cluster {cluster_id})"):
            col1, col2, col3 = st.columns(3)
            
            stats = df[df['Cluster'] == cluster_id]
            
            with col1:
                st.metric("Avg Vehicle Count", f"{stats['Vehicle_Count'].mean():.1f}")
                st.metric("Std Dev", f"{stats['Vehicle_Count'].std():.1f}")
            
            with col2:
                st.metric("Avg Vehicle Speed", f"{stats['Vehicle_Speed'].mean():.1f} km/h")
                st.metric("Std Dev", f"{stats['Vehicle_Speed'].std():.1f}")
            
            with col3:
                st.metric("Peak Hour", f"{int(stats['Hour'].mode()[0])}:00" if len(stats) > 0 else "N/A")
                st.metric("Avg Hour", f"{stats['Hour'].mean():.1f}")

# ----------------------------
# PREDICTIONS PAGE
# ----------------------------
elif page == "Predictions":
    render_page_header(
        "Traffic Pattern Prediction",
        "Enter real-time traffic signals to estimate congestion class and confidence.",
        icon="🔮",
    )
    
    st.write("Enter traffic details to predict the traffic pattern:")

    # Use observed data ranges so predictions stay within the model's learned domain.
    vc_min, vc_max = int(df['Vehicle_Count'].min()), int(df['Vehicle_Count'].max())
    speed_min = float(df['Vehicle_Speed'].min())
    speed_max = float(df['Vehicle_Speed'].max())
    hour_min, hour_max = int(df['Hour'].min()), int(df['Hour'].max())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default_vehicle_count = int(np.clip(df['Vehicle_Count'].median(), 0, 1000))
        vehicle_count = st.number_input(
            "Vehicle Count",
            min_value=0,
            max_value=1000,
            value=default_vehicle_count,
            step=1,
        )
    
    with col2:
        default_vehicle_speed = float(np.clip(df['Vehicle_Speed'].median(), 0.0, 150.0))
        vehicle_speed = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0,
            max_value=150.0,
            value=default_vehicle_speed,
            step=0.1,
        )
    
    with col3:
        hour = st.slider("Hour of Day", hour_min, hour_max, value=12)
    
    if st.button("🎯 Predict Traffic Pattern", use_container_width=True):
        try:
            validation_errors = []
            if vehicle_count <= 0:
                validation_errors.append("Vehicle Count should be greater than 0 for a meaningful prediction.")
            if vehicle_speed <= 0:
                validation_errors.append("Vehicle Speed should be greater than 0 km/h.")
            if vehicle_speed > 130:
                st.warning("Vehicle Speed is unusually high. Prediction reliability may be lower.")

            if validation_errors:
                for error_msg in validation_errors:
                    st.error(error_msg)
                st.stop()

            new_data = [[vehicle_count, vehicle_speed, hour]]
            new_scaled = scaler.transform(new_data)
            prediction = kmeans.predict(new_scaled)[0]
            
            # Distance to cluster center
            distance = np.linalg.norm(new_scaled - kmeans.cluster_centers_[prediction])
            confidence = (1 - min(distance / 10, 1))
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            condition = cluster_mapping.get(int(prediction), "Unknown")
            
            if condition == "Free Flow":
                st.success(f"### ✅ Traffic Condition: {condition}", icon="✅")
                st.write("Road is clear with smooth traffic flow. Optimal travel conditions.")
            elif condition == "Moderate Traffic":
                st.warning(f"### ⚠️ Traffic Condition: {condition}", icon="⚠️")
                st.write("Moderate congestion detected. Traffic may be slightly slower than normal.")
            else:
                st.error(f"### 🚫 Traffic Condition: {condition}", icon="🚫")
                st.write("Heavy congestion detected. Expect significant delays and slow traffic.")
            
            # Additional details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1%}",
                    help="Higher means your input is closer to the learned cluster pattern."
                )
            with col2:
                st.metric(
                    "Nearest Cluster",
                    prediction,
                    help="Internal KMeans cluster number used by the model (not a traffic label by itself)."
                )
            with col3:
                st.metric(
                    "Similarity Distance",
                    f"{distance:.3f}",
                    help="Distance from your input to the nearest cluster center in scaled feature space. Lower is better."
                )

            # Flag inputs that are outside the range seen during training.
            outside_training_range = (
                vehicle_count < vc_min
                or vehicle_count > vc_max
                or vehicle_speed < speed_min
                or vehicle_speed > speed_max
            )

            if outside_training_range:
                st.warning(
                    f"Input is outside training range (Vehicle Count: {vc_min}-{vc_max}, "
                    f"Vehicle Speed: {speed_min:.1f}-{speed_max:.1f}). "
                    "Prediction may be less reliable."
                )

            if confidence < 0.35:
                st.info(
                    "Low-confidence prediction: this input is far from common patterns in the training data."
                )
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# ----------------------------
# DATA ANALYSIS PAGE
# ----------------------------
elif page == "Data Analysis":
    render_page_header(
        "Data Analysis & Insights",
        "Explore distributions, temporal trends, and high-dimensional clustering patterns.",
        icon="📊",
    )
    
    # Feature distributions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Vehicle Count Distribution")
        fig = px.histogram(df, x='Vehicle_Count', nbins=30)
        fig.update_layout(height=350, showlegend=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vehicle Speed Distribution")
        fig = px.histogram(df, x='Vehicle_Speed', nbins=30)
        fig.update_layout(height=350, showlegend=False, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Traffic by Hour of Day")
        hourly_data = df.groupby('Hour')['Vehicle_Count'].mean()
        fig = px.line(x=hourly_data.index, y=hourly_data.values, markers=True)
        fig.update_layout(height=350, xaxis_title="Hour", yaxis_title="Avg Vehicle Count", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 3D Cluster Visualization
    st.subheader("3D Cluster Visualization")
    fig = px.scatter_3d(
        df,
        x='Vehicle_Count',
        y='Vehicle_Speed',
        z='Hour',
        color='Traffic_Condition',
        hover_data=['Vehicle_Count', 'Vehicle_Speed', 'Hour'],
        labels={'Vehicle_Count': 'Vehicle Count', 'Vehicle_Speed': 'Speed (km/h)', 'Hour': 'Hour of Day'},
        height=500
    )
    fig.update_layout(
        **CHART_LAYOUT,
        scene=dict(
            dragmode='turntable',
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.7, y=1.7, z=1.1),
            ),
        ),
        uirevision='cluster-3d',
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["resetCameraDefault3d", "hoverClosest3d"],
        },
    )
    
    st.markdown("---")
    
    # Detailed statistics table
    st.subheader("📋 Detailed Statistics")
    stats_table = df.describe().round(2)
    st.dataframe(stats_table, use_container_width=True)

# ----------------------------
# MODEL INFO PAGE
# ----------------------------
elif page == "Model Info":
    render_page_header(
        "Model Information",
        "Review clustering setup, quality metrics, and model behavior using elbow analysis.",
        icon="🔬",
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        st.write(f"**Algorithm:** KMeans Clustering")
        st.write(f"**Number of Clusters:** {optimal_clusters}")
        st.write(f"**Random State:** 42")
        st.write(f"**Initialization Method:** k-means++")
        st.write(f"**Features Used:** Vehicle Count, Vehicle Speed, Hour of Day")
    
    with col2:
        st.subheader("Model Performance")
        silhouette, db_score = compute_model_quality_metrics(scaled_data, df['Cluster'])
        
        st.write(f"**Silhouette Score:** {silhouette:.4f}")
        st.write("*(Higher is better, range: -1 to 1)*")
        st.write(f"**Davies-Bouldin Score:** {db_score:.4f}")
        st.write("*(Lower is better)*")
    
    st.markdown("---")
    st.subheader("📚 About Clustering")
    st.write("""
    **KMeans Clustering** is an unsupervised learning algorithm that partitions data into K clusters.
    
    **How it works:**
    1. Randomly initializes K cluster centers
    2. Assigns each point to the nearest center
    3. Recalculates centers based on assigned points
    4. Repeats until convergence
    
    **In this project:**
    - We use 3 clusters to represent different traffic conditions
    - Data is normalized using StandardScaler for fair distance calculations
    - Features: Vehicle Count, Vehicle Speed, and Hour of Day
    """)
    
    st.subheader("🔧 Elbow Method")
    K_range = list(range(2, 11))
    inertias = compute_elbow_inertias(scaled_data, k_start=2, k_end=10)
    
    fig = px.line(
        x=K_range,
        y=inertias,
        markers=True,
        title="Elbow Curve for Optimal K",
        labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
    )
    fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Current K=3")
    fig.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📂 Dataset Info")
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Features:** {', '.join(df.columns.tolist())}")
    st.write(f"**Data Types:**")
    st.dataframe(df.dtypes)

# ----------------------------
# ABOUT PAGE
# ----------------------------
elif page == "About":
    render_page_header(
        "About and Contact",
        "Project context, deployment details, and how to request enhancements.",
        icon="ℹ️",
    )

    st.subheader("Project Summary")
    st.write(
        "This application uses KMeans clustering on vehicle count, vehicle speed, and hour-of-day features "
        "to classify traffic conditions into interpretable congestion bands."
    )

    st.subheader("Deployment Notes")
    st.write("This Streamlit app auto-redeploys when new commits are pushed to the connected GitHub branch.")
    st.write("For reliable updates, keep app.py, requirements.txt, and at least one supported CSV dataset in the repository root.")

    st.subheader("Contact")
    st.write("For feature requests or issue reports, contact the project maintainer through your GitHub repository issues page.")
