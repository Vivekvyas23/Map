import streamlit as st
import folium
from streamlit_folium import st_folium
import networkx as nx
from pyrosm import OSM
import numpy as np
import pandas as pd
import math

# Path to your OSM PBF file
PBF_PATH = "planet_75.74,22.649_75.986,22.795.osm.pbf"

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

# ----------------- Offline Geocoder -----------------
@st.cache_resource
def load_places():
    osm = OSM(PBF_PATH)
    return osm.get_pois(custom_filter={"name": True})

places = load_places()

def offline_geocode(query):
    """Return lat/lon for the best matching place name from OSM file"""
    results = places[places["name"].str.contains(query, case=False, na=False)]
    if len(results) == 0:
        return None
    row = results.iloc[0]
    return {"latitude": row["lat"], "longitude": row["lon"], "name": row["name"]}

# ----------------- Streamlit Layout -----------------
st.set_page_config(page_title="Smart Routing - Indore", layout="wide")
st.title("🚦 Smart Traffic Routing - Indore")

# Sidebar inputs
st.sidebar.header("Route Planner")
start_text = st.sidebar.text_input("Start Location", "Rajwada Palace")
end_text = st.sidebar.text_input("End Location", "Vijay Nagar")
K = st.sidebar.slider("Number of Alternative Routes", 1, 5, 3)

run_button = st.sidebar.button("Find Routes")

# ----------------- Route Computation -----------------
if run_button:
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
        lat_min, lat_max = min(s_lat, e_lat)-0.02, max(s_lat, e_lat)+0.02
        lon_min, lon_max = min(s_lon, e_lon)-0.02, max(s_lon, e_lon)+0.02
        nodes = nodes_all[(nodes_all["lat"] >= lat_min) & (nodes_all["lat"] <= lat_max) &
                          (nodes_all["lon"] >= lon_min) & (nodes_all["lon"] <= lon_max)]
        node_ids = set(nodes["id"].astype(int).tolist())
        edges = edges_all[edges_all["u"].isin(node_ids) & edges_all["v"].isin(node_ids)].copy()

        # Build graph with random node traffic
        G = nx.DiGraph()
        np.random.seed()
        for _, r in nodes.iterrows():
            vid = int(r["id"])
            veh_count = np.random.randint(20, 200)  # random traffic
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
            weight = length + ((u_veh+v_veh)/2) * 5
            G.add_edge(u, v, length=length, weight=weight)

        s_node = nearest_node(nodes, s_lat, s_lon)
        e_node = nearest_node(nodes, e_lat, e_lon)

        # Compute K alternative routes (Yen-style uniqueness check)
        paths_iter = nx.shortest_simple_paths(G, s_node, e_node, weight="weight")
        routes = []
        seen = []
        for path in paths_iter:
            # skip if path is very similar to an existing one
            if any(set(path) == set(p["path"]) for p in seen):
                continue
            L, w, vehicles = 0.0, 0.0, 0
            for n in path:
                vehicles += G.nodes[n]["vehicles"]
            for i in range(len(path)-1):
                d = G[path[i]][path[i+1]]
                d = d if isinstance(d, dict) else list(d.values())[0]
                L += d.get("length", 0.0)
                w += d.get("weight", d.get("length", 0.0))
            route = {"path": path, "length": L, "weight": w, "vehicles": vehicles}
            routes.append(route)
            seen.append(route)
            if len(routes) >= K:
                break

        if len(routes) == 0:
            st.error("❌ No routes found")
        else:
            best_route = min(routes, key=lambda r: r["weight"])

            # Map rendering
            m = folium.Map(location=[(s_lat+e_lat)/2, (s_lon+e_lon)/2],
                           zoom_start=13, tiles="cartodbpositron")
            folium.Marker([s_lat, s_lon], tooltip="Start: "+start_text,
                          icon=folium.Icon(color="green")).add_to(m)
            folium.Marker([e_lat, e_lon], tooltip="End: "+end_text,
                          icon=folium.Icon(color="red")).add_to(m)

            cols = color_cycle(len(routes))
            for i, r in enumerate(routes, start=1):
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in r["path"]]
                km = r["length"]/1000.0
                eta = (km/25.0)*60
                label = f"Route {i}: {km:.2f} km, ETA {eta:.1f} min, {r['vehicles']} vehicles"

                if r is best_route:
                    folium.PolyLine(coords, color="green", weight=8, opacity=1,
                                    tooltip=label).add_to(m)
                else:
                    folium.PolyLine(coords, color=cols[i-1], weight=5, opacity=0.6,
                                    tooltip=label).add_to(m)

            # Save to session_state (so map persists)
            st.session_state["map_obj"] = m
            st.session_state["routes"] = routes
            st.session_state["best_route"] = best_route

# ----------------- Display Map Persistently -----------------
if "map_obj" in st.session_state:
    st_folium(st.session_state["map_obj"], width=1000, height=500)

    best_route = st.session_state["best_route"]
    st.subheader("📌 5 Point Summary")
    st.markdown(f"""
    - Best route selected: **{start_text} → {end_text}**  
    - Total distance: **{best_route['length']/1000:.2f} km**  
    - Estimated time: **{(best_route['length']/1000/25)*60:.1f} min**  
    - Vehicles on route: **{best_route['vehicles']}**  
    - Alternate routes: **{len(st.session_state['routes'])-1}** explored  
    """)
