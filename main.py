import streamlit as st
import folium
from streamlit_folium import st_folium
from pyrosm import OSM
import networkx as nx
import numpy as np
import pandas as pd
import math
import os

PBF_PATH = "planet_75.74,22.649_75.986,22.795.osm.pbf"   # <-- your Indore PBF file

# ---------- Helpers ----------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def lonlat_bbox_with_margin_km(lat1, lon1, lat2, lon2, margin_km=2.0):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians((lat1+lat2)/2.0))
    dlat = margin_km / km_per_deg_lat
    dlon = margin_km / max(km_per_deg_lon, 1e-6)
    return (min(lat1, lat2)-dlat, max(lat1, lat2)+dlat,
            min(lon1, lon2)-dlon, max(lon1, lon2)+dlon)

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

# ---------- Offline Geocoder (from OSM POIs) ----------
@st.cache_resource
def load_places():
    osm = OSM(PBF_PATH)
    return osm.get_pois(custom_filter={"name": True})

places = load_places()

def offline_geocode(query):
    results = places[places["name"].str.contains(query, case=False, na=False)]
    if len(results) == 0:
        return None
    row = results.iloc[0]
    return {"latitude": row["lat"], "longitude": row["lon"], "name": row["name"]}

# ---------- Streamlit Layout ----------
st.set_page_config(page_title="Traffic Routing - Indore", layout="wide")
st.title("🚦 Traffic-Aware Routing - Indore")

st.sidebar.header("Route Planner")
start_text = st.sidebar.text_input("Start Location", "Rajwada Palace")
end_text = st.sidebar.text_input("End Location", "Vijay Nagar")
K = st.sidebar.slider("Number of Alternative Routes", 1, 5, 3)

if st.sidebar.button("Find Route"):
    s_loc = offline_geocode(start_text)
    e_loc = offline_geocode(end_text)

    if not s_loc or not e_loc:
        st.error("❌ Could not find one of the addresses in local OSM data")
    else:
        s_lat, s_lon = s_loc["latitude"], s_loc["longitude"]
        e_lat, e_lon = e_loc["latitude"], e_loc["longitude"]

        # Load OSM network
        osm = OSM(PBF_PATH)
        nodes_all, edges_all = osm.get_network(nodes=True, network_type="driving")

        # Crop bounding box
        lat_min, lat_max, lon_min, lon_max = lonlat_bbox_with_margin_km(s_lat, s_lon, e_lat, e_lon, margin_km=2.0)
        nodes = nodes_all[(nodes_all["lat"] >= lat_min) & (nodes_all["lat"] <= lat_max) &
                          (nodes_all["lon"] >= lon_min) & (nodes_all["lon"] <= lon_max)]
        node_ids = set(nodes["id"].astype(int).tolist())
        edges = edges_all[edges_all["u"].isin(node_ids) & edges_all["v"].isin(node_ids)].copy()

        # Build graph with random node traffic
        G = nx.DiGraph()
        np.random.seed()
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
            u_veh = G.nodes[u]["vehicles"]
            v_veh = G.nodes[v]["vehicles"]
            avg_veh = (u_veh + v_veh) / 2
            weight = length + avg_veh * 5
            G.add_edge(u, v, length=length, weight=weight)

        if not nx.is_weakly_connected(G):
            G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
            nodes = nodes[nodes["id"].isin(G.nodes)]

        s_node = nearest_node(nodes, s_lat, s_lon)
        e_node = nearest_node(nodes, e_lat, e_lon)

        # Compute routes (your method)
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
        m = folium.Map(location=[(s_lat+e_lat)/2, (s_lon+e_lon)/2], zoom_start=13, tiles="cartodbpositron")
        folium.Marker([s_lat, s_lon], tooltip="Start: "+start_text, icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([e_lat, e_lon], tooltip="End: "+end_text, icon=folium.Icon(color="red")).add_to(m)

        cols = color_cycle(len(routes))
        for i, r in enumerate(routes, start=1):
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in r["path"]]
            km = r["length"]/1000.0
            eta = (km/25.0)*60
            label = f"Route {i}: {km:.2f} km, ETA {eta:.1f} min, {r['vehicles']} vehicles"

            if r is best_route:
                folium.PolyLine(coords, color="green", weight=8, opacity=1, tooltip=label).add_to(m)
            else:
                folium.PolyLine(coords, color=cols[i-1], weight=5, opacity=0.6, tooltip=label).add_to(m)

            # Show vehicle count only at intersections
            for n in r["path"]:
                if G.degree(n) >= 3:
                    folium.CircleMarker(
                        [G.nodes[n]["y"], G.nodes[n]["x"]],
                        radius=5, color="black", fill=True, fill_color="yellow",
                        tooltip=f"Node {n}: {G.nodes[n]['vehicles']} vehicles"
                    ).add_to(m)

        # Save to session_state so it persists
        st.session_state["map_obj"] = m
        st.session_state["best_route"] = best_route

# ---------- Display Map Persistently ----------
if "map_obj" in st.session_state:
    st_folium(st.session_state["map_obj"], width=1000, height=500)

    best_route = st.session_state["best_route"]
    st.subheader("📌 Summary")
    st.markdown(f"""
    - Best route: **{start_text} → {end_text}**  
    - Distance: **{best_route['length']/1000:.2f} km**  
    - ETA: **{(best_route['length']/1000/25)*60:.1f} min**  
    - Vehicles: **{best_route['vehicles']}**  
    - Alternatives: **{K-1}**  
    """)
