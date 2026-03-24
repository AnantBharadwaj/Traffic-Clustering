import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import re
from urllib.parse import urlencode, quote
from urllib.request import urlopen
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


def trigger_auto_refresh(interval_ms, key):
    try:
        import importlib

        module = importlib.import_module("streamlit_autorefresh")
        module.st_autorefresh(interval=interval_ms, key=key)
        return True
    except Exception:
        return False


def get_tomtom_api_key():
    # Preferred source in deployment: Streamlit secrets.
    try:
        secret_key = st.secrets.get("TOMTOM_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    env_key = os.getenv("TOMTOM_API_KEY")
    if env_key:
        return env_key.strip()

    # Local fallback: read from test_api.py without importing executable code.
    api_file = BASE_DIR / "test_api.py"
    if api_file.exists():
        text = api_file.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1).strip()

    return None


def get_google_maps_api_key():
    try:
        secret_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    env_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if env_key:
        return env_key.strip()

    return None


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


@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_tomtom_flow(lat, lon, api_key):
    params = {
        "point": f"{lat},{lon}",
        "key": api_key,
    }
    url = (
        "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/22/json?"
        + urlencode(params)
    )

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "flowSegmentData" not in payload:
        raise ValueError(f"Unexpected API response: {payload}")

    return payload["flowSegmentData"]


def build_departure_timestamp(selected_time):
    now_dt = datetime.now()
    departure_dt = now_dt.replace(
        hour=selected_time.hour,
        minute=selected_time.minute,
        second=0,
        microsecond=0,
    )

    # If selected time already passed today, use next day for predictive traffic lookup.
    if departure_dt < now_dt:
        departure_dt = departure_dt + timedelta(days=1)

    return int(departure_dt.timestamp())


@st.cache_data(ttl=30, show_spinner=False)
def fetch_google_traffic_segment(lat, lon, api_key, departure_timestamp):
    # Use a short nearby destination to derive duration_in_traffic and travel time index.
    destination_lat = lat
    destination_lon = lon + 0.02

    url = (
        "https://maps.googleapis.com/maps/api/distancematrix/json?"
        + urlencode(
            {
                "origins": f"{lat},{lon}",
                "destinations": f"{destination_lat},{destination_lon}",
                "departure_time": departure_timestamp,
                "traffic_model": "best_guess",
                "key": api_key,
            }
        )
    )

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("rows", [])
    if not rows or not rows[0].get("elements"):
        raise ValueError(f"Unexpected Google traffic response: {payload}")

    element = rows[0]["elements"][0]
    if element.get("status") != "OK":
        raise ValueError(f"Google traffic element unavailable: {element}")

    distance_m = float(element["distance"]["value"])
    duration_s = float(element["duration"]["value"])
    duration_traffic_s = float(element.get("duration_in_traffic", {}).get("value", duration_s))

    safe_duration_s = max(duration_s, 1.0)
    safe_duration_traffic_s = max(duration_traffic_s, 1.0)

    free_flow_speed = (distance_m / safe_duration_s) * 3.6
    current_speed = (distance_m / safe_duration_traffic_s) * 3.6

    return {
        "currentSpeed": current_speed,
        "freeFlowSpeed": free_flow_speed,
        "currentTravelTime": duration_traffic_s,
        "freeFlowTravelTime": duration_s,
        "provider": "Google Maps",
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_location_name_tomtom(lat, lon, api_key):
    url = (
        f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json?"
        + urlencode({"key": api_key})
    )

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    addresses = payload.get("addresses", [])
    if not addresses:
        return "Unknown location"

    address_data = addresses[0].get("address", {})
    freeform = address_data.get("freeformAddress")
    if freeform:
        return freeform

    parts = [
        address_data.get("municipality"),
        address_data.get("countrySubdivision"),
        address_data.get("countryCodeISO3"),
    ]
    fallback = ", ".join([part for part in parts if part])
    return fallback if fallback else "Unknown location"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_location_name_google(lat, lon, api_key):
    url = (
        "https://maps.googleapis.com/maps/api/geocode/json?"
        + urlencode({"latlng": f"{lat},{lon}", "key": api_key})
    )

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = payload.get("results", [])
    if not results:
        return "Unknown location"

    return results[0].get("formatted_address", "Unknown location")


@st.cache_data(ttl=180, show_spinner=False)
def search_locations_google(query, api_key, limit=5, country_code="in"):
    q = query.strip()
    if not q:
        return []

    auto_url = (
        "https://maps.googleapis.com/maps/api/place/autocomplete/json?"
        + urlencode(
            {
                "input": q,
                "types": "geocode",
                "components": f"country:{country_code.lower()}",
                "region": country_code.lower(),
                "key": api_key,
            }
        )
    )

    with urlopen(auto_url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    predictions = payload.get("predictions", [])[:limit]
    results = []

    for item in predictions:
        place_id = item.get("place_id")
        description = item.get("description", "Unknown location")
        if not place_id:
            continue

        details_url = (
            "https://maps.googleapis.com/maps/api/place/details/json?"
            + urlencode({"place_id": place_id, "fields": "geometry,formatted_address", "key": api_key})
        )
        with urlopen(details_url, timeout=15) as details_response:
            details_payload = json.loads(details_response.read().decode("utf-8"))

        result = details_payload.get("result", {})
        geometry = result.get("geometry", {}).get("location", {})
        lat = geometry.get("lat")
        lon = geometry.get("lng")
        if lat is None or lon is None:
            continue

        label = result.get("formatted_address") or description
        results.append({"label": label, "lat": float(lat), "lon": float(lon)})

    return results


@st.cache_data(ttl=300, show_spinner=False)
def search_locations_by_text(query, api_key, limit=5, country_set="IN"):
    q = query.strip()
    if not q:
        return []

    url = (
        "https://api.tomtom.com/search/2/geocode/"
        + quote(q, safe="")
        + ".json?"
        + urlencode({"key": api_key, "limit": limit, "countrySet": country_set})
    )

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = []
    for item in payload.get("results", []):
        position = item.get("position", {})
        address = item.get("address", {})
        lat = position.get("lat")
        lon = position.get("lon")
        if lat is None or lon is None:
            continue
        label = address.get("freeformAddress", "Unknown location")
        results.append({"label": label, "lat": float(lat), "lon": float(lon)})

    return results


def resolve_location_name(lat, lon, tomtom_api_key, google_maps_api_key):
    if google_maps_api_key:
        try:
            return fetch_location_name_google(lat, lon, google_maps_api_key)
        except Exception:
            pass

    if tomtom_api_key:
        return fetch_location_name_tomtom(lat, lon, tomtom_api_key)

    return "Unknown location"


def resolve_location_suggestions(query, tomtom_api_key, google_maps_api_key, limit=5, country_code="in"):
    if google_maps_api_key:
        try:
            return search_locations_google(query, google_maps_api_key, limit=limit, country_code=country_code)
        except Exception:
            pass

    if tomtom_api_key:
        return search_locations_by_text(query, tomtom_api_key, limit=limit, country_set=country_code.upper())

    return []


def estimate_features_from_live_flow(flow_data, historical_df, selected_hour):
    current_speed = float(flow_data.get("currentSpeed", 0.0))
    free_flow_speed = float(flow_data.get("freeFlowSpeed", current_speed if current_speed > 0 else 1.0))
    safe_free_flow_speed = max(free_flow_speed, 1.0)

    congestion_ratio = float(np.clip(1 - (current_speed / safe_free_flow_speed), 0, 1))
    current_travel_time = float(flow_data.get("currentTravelTime", 0.0))
    free_flow_travel_time = float(flow_data.get("freeFlowTravelTime", 0.0))
    if free_flow_travel_time > 0:
        travel_time_index = current_travel_time / free_flow_travel_time
    else:
        travel_time_index = 1.0 + congestion_ratio

    # Map relative speed ratio into historical speed domain.
    speed_min = float(historical_df["Vehicle_Speed"].min())
    speed_max = float(historical_df["Vehicle_Speed"].max())
    relative_speed_ratio = float(np.clip(current_speed / safe_free_flow_speed, 0.0, 1.0))
    modeled_speed = speed_min + (relative_speed_ratio * (speed_max - speed_min))
    clipped_speed = float(np.clip(modeled_speed, speed_min, speed_max))

    current_hour = int(selected_hour)

    hour_profiles = historical_df[historical_df["Hour"] == current_hour]
    if hour_profiles.empty:
        hour_profiles = historical_df

    base_count = float(hour_profiles["Vehicle_Count"].median())
    lower_count = float(hour_profiles["Vehicle_Count"].quantile(0.2))
    upper_count = float(hour_profiles["Vehicle_Count"].quantile(0.8))
    spread = max(upper_count - lower_count, 1.0)

    estimated_vehicle_count = int(round(base_count + ((congestion_ratio - 0.5) * spread)))
    estimated_vehicle_count = int(np.clip(estimated_vehicle_count, historical_df["Vehicle_Count"].min(), historical_df["Vehicle_Count"].max()))

    return {
        "vehicle_count": estimated_vehicle_count,
        "vehicle_speed": clipped_speed,
        "raw_vehicle_speed": current_speed,
        "hour": current_hour,
        "congestion_ratio": congestion_ratio,
        "free_flow_speed": free_flow_speed,
        "speed_was_clipped": bool(clipped_speed != current_speed),
        "speed_clip_range": (speed_min, speed_max),
        "travel_time_index": float(travel_time_index),
        "relative_speed_ratio": relative_speed_ratio,
        "provider": flow_data.get("provider", "Live API"),
    }


def derive_live_condition_from_index(travel_time_index, congestion_ratio, fallback_condition):
    # Backward-compatible default: Balanced profile.
    return derive_live_condition_with_profile(
        travel_time_index=travel_time_index,
        congestion_ratio=congestion_ratio,
        fallback_condition=fallback_condition,
        profile_name="Balanced",
    )


def get_live_threshold_profile(profile_name):
    profiles = {
        "Conservative": {
            "moderate_tti": 1.12,
            "moderate_ratio": 0.18,
            "heavy_tti": 1.42,
            "heavy_ratio": 0.48,
        },
        "Balanced": {
            "moderate_tti": 1.18,
            "moderate_ratio": 0.22,
            "heavy_tti": 1.55,
            "heavy_ratio": 0.55,
        },
        "Aggressive": {
            "moderate_tti": 1.25,
            "moderate_ratio": 0.30,
            "heavy_tti": 1.70,
            "heavy_ratio": 0.65,
        },
    }
    return profiles.get(profile_name, profiles["Balanced"])


def is_free_flow_override_hour(hour_value):
    # User-requested fixed windows for forcing Free Flow.
    return (
        hour_value in {22, 23, 0, 1, 2, 3, 4, 5, 6, 7}
        or hour_value in {12, 13, 14}
    )


def derive_live_condition_with_profile(travel_time_index, congestion_ratio, fallback_condition, profile_name, selected_hour=None):
    # Direct real-time condition from live API indices with configurable sensitivity.
    if selected_hour is not None and is_free_flow_override_hour(int(selected_hour)):
        return "Free Flow", "Time-based override applied for configured free-flow hours."

    if travel_time_index <= 0 or congestion_ratio < 0:
        return fallback_condition, "Live indices unavailable, using model fallback."

    profile = get_live_threshold_profile(profile_name)

    if travel_time_index >= profile["heavy_tti"] or congestion_ratio >= profile["heavy_ratio"]:
        return "Heavy Congestion", f"High delay/congestion in {profile_name} profile."
    if travel_time_index >= profile["moderate_tti"] or congestion_ratio >= profile["moderate_ratio"]:
        return "Moderate Traffic", f"Moderate delay/congestion in {profile_name} profile."
    return "Free Flow", f"Low delay/congestion in {profile_name} profile."


def run_prediction(vehicle_count, vehicle_speed, hour, model, model_scaler, mapping):
    new_data = [[vehicle_count, vehicle_speed, hour]]
    new_scaled = model_scaler.transform(new_data)
    prediction = int(model.predict(new_scaled)[0])

    distance = float(np.linalg.norm(new_scaled - model.cluster_centers_[prediction]))
    confidence = float(1 - min(distance / 10, 1))
    condition = mapping.get(prediction, "Unknown")

    return {
        "prediction": prediction,
        "distance": distance,
        "confidence": confidence,
        "condition": condition,
    }


def update_live_prediction_history(history, entry, min_gap_minutes=30):
    if not history:
        return [entry], True

    last_entry = history[-1]
    try:
        last_time = datetime.fromisoformat(last_entry["captured_at"])
        current_time = datetime.fromisoformat(entry["captured_at"])
    except Exception:
        return history + [entry], True

    gap_minutes = (current_time - last_time).total_seconds() / 60.0
    if gap_minutes >= min_gap_minutes:
        return (history + [entry])[-10:], True

    return history, False


def render_prediction_result(result):
    st.markdown("---")
    st.subheader("Prediction Result")

    condition = result["condition"]
    if condition == "Free Flow":
        st.success(f"### ✅ Traffic Condition: {condition}", icon="✅")
        st.write("Road is clear with smooth traffic flow. Optimal travel conditions.")
    elif condition == "Moderate Traffic":
        st.warning(f"### ⚠️ Traffic Condition: {condition}", icon="⚠️")
        st.write("Moderate congestion detected. Traffic may be slightly slower than normal.")
    else:
        st.error(f"### 🚫 Traffic Condition: {condition}", icon="🚫")
        st.write("Heavy congestion detected. Expect significant delays and slow traffic.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Prediction Confidence",
            f"{result['confidence']:.1%}",
            help="Higher means your input is closer to the learned cluster pattern.",
        )
    with col2:
        st.metric(
            "Nearest Cluster",
            result["prediction"],
            help="Internal KMeans cluster number used by the model (not a traffic label by itself).",
        )
    with col3:
        st.metric(
            "Similarity Distance",
            f"{result['distance']:.3f}",
            help="Distance from input to the nearest cluster center in scaled feature space. Lower is better.",
        )

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
tomtom_api_key = get_tomtom_api_key()
google_maps_api_key = get_google_maps_api_key()

if "location_candidates" not in st.session_state:
    st.session_state["location_candidates"] = []
if "live_prediction_history" not in st.session_state:
    st.session_state["live_prediction_history"] = []
if "live_required_time" not in st.session_state:
    st.session_state["live_required_time"] = datetime.now().time().replace(second=0, microsecond=0)
    
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
    
    st.write("Choose manual values or live API flow data to predict the traffic pattern.")

    # Use observed data ranges so predictions stay within the model's learned domain.
    vc_min, vc_max = int(df['Vehicle_Count'].min()), int(df['Vehicle_Count'].max())
    speed_min = float(df['Vehicle_Speed'].min())
    speed_max = float(df['Vehicle_Speed'].max())
    hour_min, hour_max = int(df['Hour'].min()), int(df['Hour'].max())
    
    source_mode = st.radio(
        "Prediction Source",
        ["Live API", "Manual Input"],
        horizontal=True,
        index=0 if tomtom_api_key else 1,
        help="Live API uses TomTom flow speed and estimates vehicle count from learned data range.",
    )

    if source_mode == "Live API":
        coord_mode = st.radio(
            "Location Input Mode",
            ["Manual Coordinates", "Search by Location Name"],
            horizontal=True,
        )

        latitude = 17.3850
        longitude = 78.4867
        has_valid_location = True

        if coord_mode == "Manual Coordinates":
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", value=17.3850, format="%.6f")
            with col2:
                longitude = st.number_input("Longitude", value=78.4867, format="%.6f")
        else:
            location_query = st.text_input(
                "Type location or area",
                value="Hyderabad",
                help="Suggestions appear while typing. Google Maps is used when GOOGLE_MAPS_API_KEY is configured.",
            )

            if len(location_query.strip()) >= 3:
                try:
                    st.session_state["location_candidates"] = resolve_location_suggestions(
                        location_query,
                        tomtom_api_key=tomtom_api_key,
                        google_maps_api_key=google_maps_api_key,
                        limit=8,
                        country_code="in",
                    )
                except Exception as e:
                    st.error(f"Error searching location: {str(e)}")

            candidates = st.session_state.get("location_candidates", [])
            if candidates:
                option_labels = [c["label"] for c in candidates]
                selected_label = st.selectbox("Select area from results", option_labels)
                selected_candidate = next((c for c in candidates if c["label"] == selected_label), candidates[0])
                latitude = selected_candidate["lat"]
                longitude = selected_candidate["lon"]
                st.caption(f"Selected coordinates: {latitude:.6f}, {longitude:.6f}")
            else:
                st.info("Type at least 3 characters to get location suggestions.")
                has_valid_location = False

        selected_time = st.time_input(
            "Required Time of Day",
            value=st.session_state["live_required_time"],
            step=1800,
            key="live_required_time",
            help="Prediction hour will use this selected time instead of current system time.",
        )
        selected_hour = int(selected_time.hour)

        live_profile = st.selectbox(
            "Live Sensitivity Profile",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            help="Controls how easily live condition changes between Free, Moderate, and Heavy.",
        )

        auto_refresh = st.checkbox("Auto refresh every 30 seconds", value=True)
        refresh_available = False
        if auto_refresh:
            refresh_available = trigger_auto_refresh(interval_ms=30_000, key="live_api_refresh")
        if auto_refresh and not refresh_available:
            st.warning("Install streamlit-autorefresh to enable timed refresh. Use manual refresh button for now.")

        if not tomtom_api_key:
            st.error("TomTom API key not found.")
            st.info(
                "Set TOMTOM_API_KEY in Streamlit secrets for deployment, or define API_KEY in test_api.py for local testing."
            )
            st.info("For better area names and suggestions, also set GOOGLE_MAPS_API_KEY in Streamlit secrets.")
        elif has_valid_location:
            try:
                departure_timestamp = build_departure_timestamp(selected_time)
                source_note = ""

                if google_maps_api_key:
                    try:
                        flow_data = fetch_google_traffic_segment(
                            latitude,
                            longitude,
                            google_maps_api_key,
                            departure_timestamp,
                        )
                        source_note = "Traffic source: Google Maps (selected-time traffic estimate)."
                    except Exception:
                        flow_data = fetch_live_tomtom_flow(latitude, longitude, tomtom_api_key)
                        flow_data["provider"] = "TomTom"
                        source_note = "Traffic source fallback: TomTom live flow."
                else:
                    flow_data = fetch_live_tomtom_flow(latitude, longitude, tomtom_api_key)
                    flow_data["provider"] = "TomTom"
                    source_note = "Traffic source: TomTom live flow (Google Maps key not configured)."

                location_name = resolve_location_name(
                    latitude,
                    longitude,
                    tomtom_api_key=tomtom_api_key,
                    google_maps_api_key=google_maps_api_key,
                )
                features = estimate_features_from_live_flow(flow_data, df, selected_hour)
                features["hour"] = selected_hour

                st.markdown("---")
                st.subheader("Location")
                st.write(f"{location_name}")
                st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), use_container_width=True)

                st.subheader("Live Data Snapshot")
                l1, l2, l3, l4 = st.columns(4)
                with l1:
                    st.metric("Current Speed", f"{features['raw_vehicle_speed']:.1f} km/h")
                with l2:
                    st.metric("Free Flow Speed", f"{features['free_flow_speed']:.1f} km/h")
                with l3:
                    st.metric("Congestion Ratio", f"{features['congestion_ratio']:.1%}")
                with l4:
                    st.metric("Estimated Vehicle Count", features['vehicle_count'])
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Travel Time Index", f"{features['travel_time_index']:.2f}")
                with m2:
                    st.metric("Relative Speed Ratio", f"{features['relative_speed_ratio']:.2f}")
                st.caption(f"Last updated at {datetime.now().strftime('%H:%M:%S')}")
                st.caption(
                    f"Live feature hour uses selected time: {selected_time.strftime('%H:%M')} (hour={features['hour']})."
                )
                st.caption(
                    "Note: provider traffic timestamp is not exposed by this endpoint; shown time is fetch time."
                )
                st.caption(source_note)

                if features["speed_was_clipped"]:
                    min_spd, max_spd = features["speed_clip_range"]
                    st.info(
                        f"Current speed was clipped to model range ({min_spd:.1f}-{max_spd:.1f} km/h) "
                        "for stable prediction."
                    )

                result = run_prediction(
                    vehicle_count=features["vehicle_count"],
                    vehicle_speed=features["vehicle_speed"],
                    hour=features["hour"],
                    model=kmeans,
                    model_scaler=scaler,
                    mapping=cluster_mapping,
                )
                model_condition = result["condition"]
                live_condition, live_reason = derive_live_condition_with_profile(
                    travel_time_index=features["travel_time_index"],
                    congestion_ratio=features["congestion_ratio"],
                    fallback_condition=model_condition,
                    profile_name=live_profile,
                    selected_hour=selected_hour,
                )
                result["condition"] = live_condition
                render_prediction_result(result)
                st.info(f"Live condition basis: {live_reason}")
                st.caption(f"Model cluster condition (secondary): {model_condition}")

                timeline_now = datetime.now()
                entry = {
                    "timestamp": timeline_now.strftime("%H:%M"),
                    "captured_at": timeline_now.isoformat(),
                    "location": location_name,
                    "lat": latitude,
                    "lon": longitude,
                    "condition": result["condition"],
                    "model_condition": model_condition,
                    "confidence": float(result["confidence"]),
                    "speed": float(features["raw_vehicle_speed"]),
                    "estimated_count": int(features["vehicle_count"]),
                    "travel_time_index": float(features["travel_time_index"]),
                    "congestion_ratio": float(features["congestion_ratio"]),
                    "source": features.get("provider", "Live API"),
                }

                history = st.session_state.get("live_prediction_history", [])
                updated_history, was_added = update_live_prediction_history(history, entry, min_gap_minutes=30)
                st.session_state["live_prediction_history"] = updated_history

                if not was_added:
                    st.info("Timeline is recorded at 30-minute intervals. Next point will be added after 30 minutes.")

                history_df = pd.DataFrame(st.session_state["live_prediction_history"])
                if not history_df.empty:
                    if "captured_at" in history_df.columns:
                        history_df["captured_at"] = pd.to_datetime(history_df["captured_at"], errors="coerce")
                    else:
                        history_df["captured_at"] = pd.NaT

                    history_df = history_df.dropna(subset=["captured_at"]).sort_values("captured_at")
                    history_df["Time"] = history_df["captured_at"].dt.strftime("%d %b %Y, %H:%M")

                    st.subheader("Recent Live Predictions")
                    st.caption("Last 10 predictions sampled at least 30 minutes apart.")

                    summary_view = history_df[["Time", "condition", "location", "speed", "estimated_count", "confidence"]].copy()
                    summary_view = summary_view.rename(
                        columns={
                            "condition": "Condition",
                            "location": "Location",
                            "speed": "Speed (km/h)",
                            "estimated_count": "Estimated Vehicle Count",
                            "confidence": "Confidence",
                        }
                    )
                    if "source" in history_df.columns:
                        summary_view["Traffic Source"] = history_df["source"]
                    if "travel_time_index" in history_df.columns:
                        summary_view["Travel Time Index"] = history_df["travel_time_index"].round(2)
                    if "congestion_ratio" in history_df.columns:
                        summary_view["Congestion Ratio"] = (history_df["congestion_ratio"] * 100).round(1).astype(str) + "%"
                    summary_view = summary_view.tail(10)
                    st.dataframe(summary_view, use_container_width=True, hide_index=True)

                if result["confidence"] < 0.35:
                    st.info("Low-confidence prediction: this live input is far from common patterns in the training data.")

            except Exception as e:
                st.error(f"Error fetching live API data or predicting traffic condition: {str(e)}")
        else:
            st.info("Select a valid India location from suggestions to fetch live traffic condition.")

    else:
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
                else:
                    result = run_prediction(
                        vehicle_count=vehicle_count,
                        vehicle_speed=vehicle_speed,
                        hour=hour,
                        model=kmeans,
                        model_scaler=scaler,
                        mapping=cluster_mapping,
                    )
                    render_prediction_result(result)

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

                    if result["confidence"] < 0.35:
                        st.info("Low-confidence prediction: this input is far from common patterns in the training data.")

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
    st.write("For live traffic mode, configure TOMTOM_API_KEY in Streamlit secrets.")

    st.subheader("Contact")
    st.write("For feature requests or issue reports, contact the project maintainer through your GitHub repository issues page.")
