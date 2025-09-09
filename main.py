import streamlit as st
import folium
from streamlit_folium import st_folium
from pyrosm import OSM
import networkx as nx
import numpy as np
import pandas as pd
import math

PBF_PATH = "planet_75.74,22.649_75.986,22.795.osm.pbf"   # <-- path to your PBF file

# ----------------- Helpers -----------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def nearest_node(nodes_df, lat, lon):
    lat_arr = nodes_df["lat"].to_numpy()
    lon_arr = nodes_df["lon"].to_numpy()
    d2 = (lat_arr-lat)**2 + (lon_arr-lon)**2
    idx = int(np.argmin(d2))
    return int(nodes_df.iloc[idx]["id"])

def color_cycle(n):
    base = ["blue", "red", "purple", "orange", "darkred",
            "cadetblue", "black", "pink", "brown"]
    return [base[i % len(base)] for i in range(n)]

# ----------------- Streamlit -----------------
st.set_page_config(page_title="Traffic Routing", layout="wide")
st.title("🚦 Traffic-Aware Routing (Indore)")

# Sidebar input
start_lat = st.sidebar.number_input("Start Latitude", value=22.7196)
start_lon = st.sidebar.number_input("Start Longitude", value=75.8577)
end_lat = st.sidebar.number_input("End Latitude", value=22.7500)
end_lon = st.sidebar.number_input("End Longitude", value=75.9000)
K = st.sidebar.slider("Number of Alternative Routes", 1, 5, 3)

if st.sidebar.button("Find Route"):
    # Load OSM
    osm = OSM(PBF_PATH)
    nodes_all, edges_all = osm.get_network(nodes=True, network_type="driving")

    # Crop bounding box
    lat_min, lat_max = min(start_lat, end_lat)-0.02, max(start_lat, end_lat)+0.02
    lon_min, lon_max = min(start_lon, end_lon)-0.02, max(start_lon, end_lon)+0.02
    nodes = nodes_all[(nodes_all["lat"] >= lat_min) & (nodes_all["lat"] <= lat_max) &
                      (nodes_all["lon"] >= lon_min) & (nodes_all["lon"] <= lon_max)]
    node_ids = set(nodes["id"].astype(int).tolist())
    edges = edges_all[edges_all["u"].isin(node_ids) & edges_all["v"].isin(node_ids)].copy()

    # Build graph with random vehicle counts
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        vid = int(r["id"])
        veh_count = np.random.randint(20, 200)
        G.add_node(vid, x=float(r["lon"]), y=float(r["lat"]), vehicles=veh_count)

    if "length" not in edges.columns:
        def geodesic_len(row):
            geom = row.geometry
            coords = list(geom.coords) if geom else []
            total = 0.0
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                total += haversine_m(y1, x1, y2, x2)
            return total
        edges["length"] = edges.apply(geodesic_len, axis=1)

    for _, r in edges.iterrows():
        u, v = int(r["u"]), int(r["v"])
        if u not in G.nodes or v not in G.nodes: 
            continue
        length = float(r["length"]) if pd.notna(r["length"]) else 1.0
        u_veh, v_veh = G.nodes[u]["vehicles"], G.nodes[v]["vehicles"]
        weight = length + ((u_veh+v_veh)/2)*5
        G.add_edge(u, v, length=length, weight=weight)

    # Nearest nodes
    s_node = nearest_node(nodes, start_lat, start_lon)
    e_node = nearest_node(nodes, end_lat, end_lon)

    # Compute routes
    paths_iter = nx.shortest_simple_paths(G, s_node, e_node, weight="weight")
    routes = []
    for idx, path in enumerate(paths_iter):
        L, w, vehicles = 0.0, 0.0, 0
        for n in path:
            vehicles += G.nodes[n]["vehicles"]
        for i in range(len(path)-1):
            d = G[path[i]][path[i+1]]
            d = d if isinstance(d, dict) else list(d.values())[0]
            L += d.get("length", 0.0)
            w += d.get("weight", d.get("length", 0.0))
        routes.append({"path": path, "length": L, "weight": w, "vehicles": vehicles})
        if idx >= K-1:
            break

    best_route = min(routes, key=lambda r: r["weight"])

    # Build folium map
    m = folium.Map(location=[(start_lat+end_lat)/2, (start_lon+end_lon)/2], zoom_start=13)
    folium.Marker([start_lat, start_lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([end_lat, end_lon], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    cols = color_cycle(len(routes))
    for i, r in enumerate(routes, start=1):
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in r["path"]]
        label = f"Route {i}: {r['length']/1000:.2f} km, {r['vehicles']} vehicles"
        if r is best_route:
            folium.PolyLine(coords, color="green", weight=8, opacity=1, tooltip=label).add_to(m)
        else:
            folium.PolyLine(coords, color=cols[i-1], weight=5, opacity=0.6, tooltip=label).add_to(m)

    # Store map persistently
    st.session_state["map_obj"] = m
    st.session_state["best_route"] = best_route

# Only render saved map
if "map_obj" in st.session_state:
    st_folium(st.session_state["map_obj"], width=1000, height=500)
