#!/usr/bin/env python3
# STAR-CS Command Center with GRM-style Risk Field, Predictive Overlay, and Rich Event Context

import math
import os
import json
import threading
from typing import Dict, Any, Tuple, Set, List

import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Point, PoseStamped, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion

# ---------------- Configuration ----------------

AGENT_IDS = ["agent1", "agent2", "agent3"]
AGENT_TYPES = {"agent1": "worker", "agent2": "worker", "agent3": "ugv"}

AGENT_TRAJECTORY_COLORS = [
    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=0.6, b=0.0, a=1.0),
    ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
]

# Site bounds (meters). Grid cell ~1.2192 m (4 ft)
SITE_X_MIN, SITE_X_MAX = -11.43, 11.43
SITE_Y_MIN, SITE_Y_MAX = -6.86, 6.86
GRID_RESOLUTION = 1.2192

# Static hazards are predetermined (no detection node in STAR-CS)
PREDEFINED_HAZARDS = [
    {"class": "skid steer",             "x": -8.5, "y":  4.2, "z": 0.1, "id": "sh_1"},
    {"class": "electric service panel", "x":  9.1, "y": -3.8, "z": 0.1, "id": "esp_1"},
    {"class": "cables",                 "x": -2.3, "y":  5.7, "z": 0.1, "id": "cab_1"},
    {"class": "table saw",              "x":  6.8, "y":  2.1, "z": 0.1, "id": "ts_1"},
    {"class": "rebar w cap",            "x": -5.2, "y": -4.9, "z": 0.1, "id": "rwc_1"},
    {"class": "rebar w-o cap",          "x":  3.7, "y": -1.5, "z": 0.1, "id": "rwoc_1"},
]

# Risk composition weights
W_STATIC, W_DYNAMIC, W_PREDICTIVE = 0.10, 0.20, 0.70

# Distance kernel powers
P_STATIC = 2.0
P_DYNAMIC = 2.0
P_PRED = 2.0
DIST_EPS = 0.5


def kernel_dist_power(d: float, p: float, eps: float) -> float:
    """Distance-based kernel: larger when distance is small."""
    if d < 0.0:
        d = 0.0
    return 1.0 / ((d + eps) ** p)


# TTC kernel
TTC_EPS = 0.5
TTC_MAX = 8.0


def kernel_ttc(ttc: float, ttc_eps: float, ttc_max: float) -> float:
    """Time-to-collision kernel: larger when collision is sooner."""
    if ttc is None or ttc <= 0.0:
        return 0.0
    t = min(ttc, ttc_max)
    return 1.0 / (t + ttc_eps)


def time_to_collision(px, py, vx, vy, qx, qy, ux, uy):
    """
    TTC between moving points P(p,v) and Q(q,u) in 2D (None if they aren't moving towards each other).
    """
    rx, ry = (px - qx), (py - qy)
    vxr, vyr = (vx - ux), (vy - uy)
    v2 = vxr * vxr + vyr * vyr
    if v2 < 1e-6:
        return None
    dp = rx * vxr + ry * vyr
    if dp >= 0.0:
        return None
    return -dp / v2


# Hazard severity from simple risk matrix (C,E)
HAZARD_RISK_MATRIX = {
    "skid steer":             (5, 3),
    "electric service panel": (5, 3),
    "cables":                 (3, 2),
    "table saw":              (4, 3),
    "rebar w cap":            (2, 2),
    "rebar w-o cap":          (5, 3),
}
DEFAULT_CE = (3, 2)


def hazard_severity_from_matrix(hclass: str) -> float:
    C, E = HAZARD_RISK_MATRIX.get(hclass, DEFAULT_CE)
    S = float(C) * float(E)
    return max(1.0, min(25.0, S))


# Agent hazardousness proxy from kinetic energy
AGENT_MASS = {"worker": 80.0, "ugv": 120.0}
M_REF = 80.0
V_REF = 1.40
KAPPA = 6.0


def agent_hazardousness(agent_type: str, speed_m_s: float) -> float:
    """Approximate risk from kinetic energy: heavier + faster = more hazardous."""
    m = AGENT_MASS.get(agent_type, M_REF)
    H = KAPPA * (m * (speed_m_s ** 2)) / (M_REF * (V_REF ** 2) + 1e-9)
    return max(1.0, min(25.0, H))


# Interaction gain
GAMMA_K = 1.0

# Predictive tightening thresholds
MIN_SPEED_PRED = 0.20
MIN_REL_SPEED_PRED = 0.20
COS_HEADING_CONE_AH = 0.5  # ~60 deg cone in front of agent

# Predictive peak selection
THRESH_MODE = "p90"  # percentile mode
EXCEED_PERCENTILE = 90.0
ABS_THRESH = 3.5
TOPK_EXCEED_CELLS = 20

# Risk snapshots → reasoning engine
RISK_SNAPSHOT_INTERVAL_S = 5.0  # seconds


class CommandCenterNode(Node):
    """
    ROS2 node that:
    - Listens to agent poses
    - Computes GRM-like risk field (static + dynamic + predictive)
    - Detects Zone Entry events (with zone hazards and zone agents)
    - Periodically sends Risk Snapshot events (risk zones + predicted interactions)
    - Publishes RViz markers.
    """

    def __init__(self):
        super().__init__("command_center_node")

        self.data_lock = threading.Lock()

        # Agent state
        self.agents: Dict[str, Dict[str, Any]] = {}
        for agent_id in AGENT_IDS:
            self.agents[agent_id] = {
                "pose": None,
                "velocity": Vector3(),
                "type": AGENT_TYPES.get(agent_id, "worker"),
                "last_update": self.get_clock().now(),
                "trajectory_points": [],
            }
            self.create_subscription(
                PoseStamped,
                f"/{agent_id}/pose",
                lambda msg, aid=agent_id: self.agent_pose_callback(msg, aid),
                10,
            )

        # Static hazards
        self.hazards: Dict[str, Dict[str, Any]] = {}
        for hz in PREDEFINED_HAZARDS:
            self.hazards[hz["id"]] = {
                "class": hz["class"],
                "position": Point(x=hz["x"], y=hz["y"], z=hz["z"]),
            }

        # Site plan and zones
        self.site_plan = self.load_site_plan()
        self.agent_zones: Dict[str, Any] = {aid: None for aid in AGENT_IDS}

        # Risk grid
        self.x_coords = np.arange(SITE_X_MIN, SITE_X_MAX + 1e-9, GRID_RESOLUTION)
        self.y_coords = np.arange(SITE_Y_MIN, SITE_Y_MAX + 1e-9, GRID_RESOLUTION)
        self.risk_grid = np.zeros((len(self.y_coords), len(self.x_coords)), dtype=float)
        self._viz_pred_cells: Set[Tuple[int, int]] = set()
        self.predicted_interactions: List[Dict[str, Any]] = []

        # Publishers / timers
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.create_timer(0.2, self.visualization_and_risk_loop)

        # Risk snapshot throttling
        self._last_risk_snapshot_time = 0.0

        self.get_logger().info("STAR-CS Command Center with Risk Field initialized.")

    # ---------- Utility: site plan & zones ----------

    def load_site_plan(self) -> Dict[str, Any]:
        """Load site_plan.json from the ROS2 package."""
        try:
            pkg_share = get_package_share_directory("star_cs_agent")
            plan_path = os.path.join(pkg_share, "data", "site_plan.json")
            self.get_logger().info(f"Loading site plan from: {plan_path}")
            with open(plan_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"FATAL: failed to load site plan: {e}")
            return {}

    def is_point_in_zone(self, point: Point, polygon: List[List[float]]) -> bool:
        """Return True if a 2D point lies inside a polygon (ray-casting algorithm)."""
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # ---------- Callbacks ----------

    def agent_pose_callback(self, msg: PoseStamped, agent_id: str):
        """Update pose, velocity, and trajectory for a given agent."""
        with self.data_lock:
            st = self.agents[agent_id]
            now = self.get_clock().now()
            if st["pose"] is not None:
                dt = (now - st["last_update"]).nanoseconds / 1e9
                if dt > 0.01:
                    dx = msg.pose.position.x - st["pose"].position.x
                    dy = msg.pose.position.y - st["pose"].position.y
                    vx = dx / dt
                    vy = dy / dt
                    st["velocity"] = Vector3(x=vx, y=vy, z=0.0)
            st["pose"] = msg.pose
            st["last_update"] = now

            pos = msg.pose.position
            st["trajectory_points"].append(Point(x=pos.x, y=pos.y, z=pos.z))
            if len(st["trajectory_points"]) > 150:
                st["trajectory_points"].pop(0)

    # ---------- Risk peak helper ----------

    def _compute_pred_peaks(self) -> Tuple[float, Set[Tuple[int, int]]]:
        """Find high-risk cells (predictive peaks) based on percentile or absolute threshold."""
        if THRESH_MODE.lower().startswith("p"):
            p = float(EXCEED_PERCENTILE)
            thr = float(np.percentile(self.risk_grid, p))
        else:
            thr = float(ABS_THRESH)

        idxs = np.argwhere(self.risk_grid >= thr)
        cells: Set[Tuple[int, int]] = set()
        if idxs.size > 0:
            def is_local_peak(i, j):
                v = self.risk_grid[i, j]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ii = i + di
                        jj = j + dj
                        if 0 <= ii < self.risk_grid.shape[0] and 0 <= jj < self.risk_grid.shape[1]:
                            if self.risk_grid[ii, jj] > v:
                                return False
                return True

            peaks = []
            for (i, j) in idxs:
                if is_local_peak(i, j):
                    peaks.append((i, j, float(self.risk_grid[i, j])))
            peaks.sort(key=lambda x: x[2], reverse=True)
            peaks = peaks[:TOPK_EXCEED_CELLS]
            cells = {(i, j) for (i, j, _) in peaks}
        return thr, cells

    # ---------- Main loop: risk, events, visualization ----------

    def visualization_and_risk_loop(self):
        # Snapshot of current state
        with self.data_lock:
            hazards_copy = dict(self.hazards)
            active_agents = {aid: st for aid, st in self.agents.items() if st["pose"] is not None}
            now_ros = self.get_clock().now()
            now_msg = now_ros.to_msg()
            now_sec = now_ros.nanoseconds / 1e9

        # --------- Compute risk field (static + dynamic + predictive) ---------
        ny, nx = self.risk_grid.shape
        self.risk_grid[:, :] = 0.0

        interactions: List[Dict[str, Any]] = []

        # Agent–agent predictive interactions
        items = list(active_agents.items())
        for a in range(len(items)):
            idA, A = items[a]
            if A["pose"] is None:
                continue
            pAx, pAy = A["pose"].position.x, A["pose"].position.y
            vAx, vAy = A["velocity"].x, A["velocity"].y
            spdA = math.hypot(vAx, vAy)
            phiA = agent_hazardousness(A["type"], spdA)
            for b in range(a + 1, len(items)):
                idB, B = items[b]
                if B["pose"] is None:
                    continue
                pBx, pBy = B["pose"].position.x, B["pose"].position.y
                vBx, vBy = B["velocity"].x, B["velocity"].y
                spdB = math.hypot(vBx, vBy)
                phiB = agent_hazardousness(B["type"], spdB)

                rel_spd = math.hypot(vAx - vBx, vAy - vBy)
                if rel_spd < MIN_REL_SPEED_PRED:
                    continue

                ttc = time_to_collision(pAx, pAy, vAx, vAy, pBx, pBy, vBx, vBy)
                if ttc is None:
                    continue

                kpair = GAMMA_K * math.sqrt(phiA * phiB)
                ttc_w = kernel_ttc(ttc, TTC_EPS, TTC_MAX)
                midx, midy = 0.5 * (pAx + pBx), 0.5 * (pAy + pBy)

                # Map focus point to nearest grid cell
                ix = int((midx - SITE_X_MIN) / GRID_RESOLUTION)
                iy = int((midy - SITE_Y_MIN) / GRID_RESOLUTION)
                ix = max(0, min(nx - 1, ix))
                iy = max(0, min(ny - 1, iy))
                cell_id = iy * nx + ix
                cell_x = self.x_coords[ix] + GRID_RESOLUTION / 2.0
                cell_y = self.y_coords[iy] + GRID_RESOLUTION / 2.0

                interactions.append(
                    {
                        "type": "agent-agent",
                        "agents": [idA, idB],
                        "focus": (midx, midy),
                        "ttc": float(ttc),
                        "kpair": float(kpair),
                        "ttc_weight": float(ttc_w),
                        "cell_id": int(cell_id),
                        "cell_i": int(iy),
                        "cell_j": int(ix),
                        "cell_x": float(cell_x),
                        "cell_y": float(cell_y),
                    }
                )

        # Agent–hazard predictive interactions
        for idA, A in active_agents.items():
            if A["pose"] is None:
                continue
            pAx, pAy = A["pose"].position.x, A["pose"].position.y
            vAx, vAy = A["velocity"].x, A["velocity"].y
            vmag = math.hypot(vAx, vAy)
            if vmag < MIN_SPEED_PRED:
                continue
            q = A["pose"].orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            hx_hat, hy_hat = math.cos(yaw), math.sin(yaw)
            phiA = agent_hazardousness(A["type"], vmag)

            for hz_id, H in hazards_copy.items():
                pHx, pHy = H["position"].x, H["position"].y
                toH_x, toH_y = (pHx - pAx), (pHy - pAy)
                toH_norm = math.hypot(toH_x, toH_y)
                if toH_norm > 1e-6:
                    cos_ang = (hx_hat * toH_x + hy_hat * toH_y) / toH_norm
                    if cos_ang < COS_HEADING_CONE_AH:
                        continue

                ttc = time_to_collision(pAx, pAy, vAx, vAy, pHx, pHy, 0.0, 0.0)
                if ttc is None:
                    continue

                phiH = hazard_severity_from_matrix(H["class"])
                kpair = GAMMA_K * math.sqrt(phiA * phiH)
                ttc_w = kernel_ttc(ttc, TTC_EPS, TTC_MAX)

                # Map agent position (focus) to nearest grid cell
                ix = int((pAx - SITE_X_MIN) / GRID_RESOLUTION)
                iy = int((pAy - SITE_Y_MIN) / GRID_RESOLUTION)
                ix = max(0, min(nx - 1, ix))
                iy = max(0, min(ny - 1, iy))
                cell_id = iy * nx + ix
                cell_x = self.x_coords[ix] + GRID_RESOLUTION / 2.0
                cell_y = self.y_coords[iy] + GRID_RESOLUTION / 2.0

                interactions.append(
                    {
                        "type": "agent-hazard",
                        "agents": [idA, hz_id],
                        "hazard_class": H["class"],
                        "focus": (pAx, pAy),
                        "ttc": float(ttc),
                        "kpair": float(kpair),
                        "ttc_weight": float(ttc_w),
                        "cell_id": int(cell_id),
                        "cell_i": int(iy),
                        "cell_j": int(ix),
                        "cell_x": float(cell_x),
                        "cell_y": float(cell_y),
                    }
                )

        # Store top interactions (for LLM narrative)
        interactions_sorted = sorted(
            interactions, key=lambda d: d["kpair"] * d["ttc_weight"], reverse=True
        )
        self.predicted_interactions = interactions_sorted[:5]

        # Risk field combination for each cell
        for iy, y in enumerate(self.y_coords):
            cy = y + GRID_RESOLUTION / 2.0
            for ix, x in enumerate(self.x_coords):
                cx = x + GRID_RESOLUTION / 2.0

                # Static hazard contribution
                r_static = 0.0
                for _, hz in hazards_copy.items():
                    dist = math.hypot(cx - hz["position"].x, cy - hz["position"].y)
                    sev = hazard_severity_from_matrix(hz["class"])
                    r_static += sev * kernel_dist_power(dist, P_STATIC, DIST_EPS)

                # Dynamic agent contribution
                r_dynamic = 0.0
                for _, ag in active_agents.items():
                    dist = math.hypot(cx - ag["pose"].position.x, cy - ag["pose"].position.y)
                    spd = math.hypot(ag["velocity"].x, ag["velocity"].y)
                    hzn = agent_hazardousness(ag["type"], spd)
                    r_dynamic += hzn * kernel_dist_power(dist, P_DYNAMIC, DIST_EPS)

                # Predictive contribution from interactions
                r_predictive = 0.0
                for inter in interactions:
                    fx, fy = inter["focus"]
                    d_focus = math.hypot(cx - fx, cy - fy)
                    spatial = kernel_dist_power(d_focus, P_PRED, DIST_EPS)
                    r_predictive += inter["kpair"] * inter["ttc_weight"] * spatial

                self.risk_grid[iy, ix] = (
                    W_STATIC * r_static + W_DYNAMIC * r_dynamic + W_PREDICTIVE * r_predictive
                )

        # Predictive peaks (yellow overlay)
        _, cells = self._compute_pred_peaks()
        self._viz_pred_cells = cells

        # --------- Zone entry detection (after risk field so we have full context) ---------
        for agent_id, data in active_agents.items():
            current_pos = data["pose"].position
            previous_zone = self.agent_zones[agent_id]
            current_zone = None
            for zone_id, zone_data in self.site_plan.items():
                if self.is_point_in_zone(current_pos, zone_data["polygon_coords"]):
                    current_zone = zone_id
                    break
            if current_zone != previous_zone:
                self.agent_zones[agent_id] = current_zone
                if current_zone is not None:
                    self._trigger_zone_entry_event(
                        timestamp=int(now_sec),
                        agent_id=agent_id,
                        zone_id=current_zone,
                        agent_pos=current_pos,
                        active_agents=active_agents,
                        hazards=hazards_copy,
                    )

        # --------- Risk snapshot trigger ---------
        if cells and (now_sec - self._last_risk_snapshot_time) >= RISK_SNAPSHOT_INTERVAL_S:
            self._last_risk_snapshot_time = now_sec
            self._trigger_risk_snapshot_event(
                timestamp=int(now_sec),
                active_agents=active_agents,
                hazards=hazards_copy,
                peak_cells=cells,
                interactions=self.predicted_interactions,
            )

        # --------- Visualization ---------
        self._publish_markers(now_msg, active_agents, hazards_copy)

    # ---------- Event helpers ----------

    def _trigger_zone_entry_event(
        self,
        timestamp: int,
        agent_id: str,
        zone_id: str,
        agent_pos: Point,
        active_agents: Dict[str, Any],
        hazards: Dict[str, Any],
    ):
        """Send a rich Zone Entry event to the reasoning server."""
        import requests

        zone_poly = self.site_plan[zone_id]["polygon_coords"]

        # Hazards inside this zone
        zone_hazards = []
        for hid, hz in hazards.items():
            if self.is_point_in_zone(hz["position"], zone_poly):
                zone_hazards.append(
                    {
                        "hazard_id": hid,
                        "class": hz["class"],
                        "x": hz["position"].x,
                        "y": hz["position"].y,
                        "z": hz["position"].z,
                    }
                )

        # Agents inside this zone (with type and speed)
        zone_agents = []
        for aid, st in active_agents.items():
            if self.is_point_in_zone(st["pose"].position, zone_poly):
                vx, vy = st["velocity"].x, st["velocity"].y
                speed = math.hypot(vx, vy)
                zone_agents.append(
                    {
                        "agent_id": aid,
                        "agent_type": st["type"],
                        "speed_m_s": float(speed),
                        "x": st["pose"].position.x,
                        "y": st["pose"].position.y,
                        "z": st["pose"].position.z,
                    }
                )

        payload = {
            "timestamp": timestamp,
            "event_type": "Zone Entry",
            "agent_id": agent_id,
            "zone_id": zone_id,
            "details": (
                f"Agent {agent_id} entered zone '{zone_id}' "
                f"({self.site_plan[zone_id].get('description', '')})."
            ),
            "agent_location": {"x": agent_pos.x, "y": agent_pos.y, "z": agent_pos.z},
            "all_agents_locations": {
                aid: {
                    "x": st["pose"].position.x,
                    "y": st["pose"].position.y,
                    "z": st["pose"].position.z,
                }
                for aid, st in active_agents.items()
            },
            "zone_hazards": zone_hazards,
            "zone_agents": zone_agents,
        }

        try:
            requests.post("http://127.0.0.1:5001/analyze", json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Could not connect to Reasoning API Server (zone entry): {e}")

    def _trigger_risk_snapshot_event(
        self,
        timestamp: int,
        active_agents: Dict[str, Any],
        hazards: Dict[str, Any],
        peak_cells: Set[Tuple[int, int]],
        interactions: List[Dict[str, Any]],
    ):
        """Send periodic risk snapshot with high-risk zones and predicted interactions."""
        import requests

        # Aggregate zone-level risk using only peak cells
        zone_risk: Dict[str, Dict[str, Any]] = {}
        ny, nx = self.risk_grid.shape

        for (i, j) in peak_cells:
            cx = self.x_coords[j] + GRID_RESOLUTION / 2.0
            cy = self.y_coords[i] + GRID_RESOLUTION / 2.0
            cell_point = Point(x=cx, y=cy, z=0.0)

            zone_id = None
            for zid, zdata in self.site_plan.items():
                if self.is_point_in_zone(cell_point, zdata["polygon_coords"]):
                    zone_id = zid
                    break
            if zone_id is None:
                continue

            zr = zone_risk.setdefault(
                zone_id,
                {
                    "zone_id": zone_id,
                    "total_risk": 0.0,
                    "max_risk": 0.0,
                    "num_cells": 0,
                    "cells": [],
                },
            )
            val = float(self.risk_grid[i, j])
            zr["total_risk"] += val
            zr["max_risk"] = max(zr["max_risk"], val)
            zr["num_cells"] += 1
            zr["cells"].append({"i": int(i), "j": int(j), "x": cx, "y": cy})

        # Attach agents/hazards membership and hazard details per zone
        for zid, zr in zone_risk.items():
            zone_poly = self.site_plan[zid]["polygon_coords"]

            zr["agents"] = [
                aid for aid, st in active_agents.items()
                if self.is_point_in_zone(st["pose"].position, zone_poly)
            ]

            hazard_ids = [
                hid for hid, hz in hazards.items()
                if self.is_point_in_zone(hz["position"], zone_poly)
            ]
            zr["hazards"] = hazard_ids
            zr["hazard_details"] = [
                {
                    "hazard_id": hid,
                    "class": hazards[hid]["class"],
                    "x": hazards[hid]["position"].x,
                    "y": hazards[hid]["position"].y,
                    "z": hazards[hid]["position"].z,
                }
                for hid in hazard_ids
            ]

            if zr["num_cells"] > 0:
                zr["average_risk"] = zr["total_risk"] / zr["num_cells"]
            else:
                zr["average_risk"] = 0.0

        risk_zones = sorted(zone_risk.values(), key=lambda z: z["total_risk"], reverse=True)

        payload = {
            "timestamp": timestamp,
            "event_type": "Risk Snapshot",
            "details": "Periodic risk snapshot with high-risk zones and predicted interactions.",
            "risk_zones": risk_zones,
            "predictive_interactions": interactions,
            "all_agents_locations": {
                aid: {
                    "x": st["pose"].position.x,
                    "y": st["pose"].position.y,
                    "z": st["pose"].position.z,
                }
                for aid, st in active_agents.items()
            },
        }

        try:
            requests.post("http://127.0.0.1:5001/analyze", json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Could not connect to Reasoning API Server (risk snapshot): {e}")

    # ---------- Visualization: risk, peaks, hazards, agents, zones ----------

    def _publish_markers(self, current_time_msg, active_agents, hazards_copy):
        marker_array = MarkerArray()
        mid = 0

        ny, nx = self.risk_grid.shape

        # 1) Risk field base heatmap
        risk_m = Marker()
        risk_m.header.frame_id = "map"
        risk_m.header.stamp = current_time_msg
        risk_m.ns = "risk_field"
        risk_m.id = mid
        mid += 1
        risk_m.type = Marker.CUBE_LIST
        risk_m.action = Marker.ADD
        risk_m.pose.orientation.w = 1.0
        risk_m.scale.x = GRID_RESOLUTION * 0.95
        risk_m.scale.y = GRID_RESOLUTION * 0.95
        risk_m.scale.z = 0.05

        vmax = float(np.percentile(self.risk_grid, 98)) if np.any(self.risk_grid > 0) else 1.0
        if vmax <= 0:
            vmax = 1.0
        lo = (0.6, 0.8, 1.0)
        hi = (0.7, 0.0, 0.2)

        for iy, y in enumerate(self.y_coords):
            for ix, x in enumerate(self.x_coords):
                v = min(1.0, float(self.risk_grid[iy, ix]) / vmax)
                risk_m.points.append(
                    Point(x=x + GRID_RESOLUTION / 2.0, y=y + GRID_RESOLUTION / 2.0, z=0.0)
                )
                r = lo[0] + (hi[0] - lo[0]) * v
                g = lo[1] + (hi[1] - lo[1]) * v
                b = lo[2] + (hi[2] - lo[2]) * v
                a = 0.15 + 0.75 * v
                risk_m.colors.append(ColorRGBA(r=r, g=g, b=b, a=a))
        marker_array.markers.append(risk_m)

        # 1b) Yellow predictive overlay for peak cells
        if self._viz_pred_cells:
            pred_m = Marker()
            pred_m.header.frame_id = "map"
            pred_m.header.stamp = current_time_msg
            pred_m.ns = "predictive_overlay"
            pred_m.id = mid
            mid += 1
            pred_m.type = Marker.CUBE_LIST
            pred_m.action = Marker.ADD
            pred_m.pose.orientation.w = 1.0
            pred_m.scale.x = GRID_RESOLUTION * 0.95
            pred_m.scale.y = GRID_RESOLUTION * 0.95
            pred_m.scale.z = 0.06
            for (i, j) in self._viz_pred_cells:
                x = self.x_coords[j] + GRID_RESOLUTION / 2.0
                y = self.y_coords[i] + GRID_RESOLUTION / 2.0
                pred_m.points.append(Point(x=x, y=y, z=0.03))
                pred_m.colors.append(ColorRGBA(r=1.0, g=0.9, b=0.0, a=0.9))
            marker_array.markers.append(pred_m)

        # 2) Ground plane
        ground = Marker()
        ground.header.frame_id = "map"
        ground.header.stamp = current_time_msg
        ground.ns = "site_layout"
        ground.id = mid
        mid += 1
        ground.type = Marker.CUBE
        ground.action = Marker.ADD
        center_x = (SITE_X_MIN + SITE_X_MAX) / 2.0
        center_y = (SITE_Y_MIN + SITE_Y_MAX) / 2.0
        ground.pose.position = Point(x=center_x, y=center_y, z=-0.05)
        ground.pose.orientation.w = 1.0
        ground.scale.x = (SITE_X_MAX - SITE_X_MIN)
        ground.scale.y = (SITE_Y_MAX - SITE_Y_MIN)
        ground.scale.z = 0.02
        ground.color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.3)
        marker_array.markers.append(ground)

        # 3) Hazard zones from site plan
        for zone_id, zone_data in self.site_plan.items():
            poly = zone_data["polygon_coords"]
            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = current_time_msg
            line.ns = "zones"
            line.id = mid
            mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color = ColorRGBA(r=0.2, g=0.2, b=0.8, a=0.8)
            for (x, y) in poly + [poly[0]]:
                line.points.append(Point(x=x, y=y, z=0.02))
            marker_array.markers.append(line)

        # 4) Static hazards
        for hz in PREDEFINED_HAZARDS:
            cyl = Marker()
            cyl.header.frame_id = "map"
            cyl.header.stamp = current_time_msg
            cyl.ns = "hazards"
            cyl.id = mid
            mid += 1
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position = Point(x=hz["x"], y=hz["y"], z=hz["z"])
            cyl.pose.orientation.w = 1.0
            cyl.scale = Vector3(x=0.5, y=0.5, z=0.4)
            cyl.color = ColorRGBA(r=0.6, g=0.1, b=0.6, a=0.9)
            marker_array.markers.append(cyl)

            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = current_time_msg
            txt.ns = "hazard_labels"
            txt.id = mid
            mid += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position = Point(x=hz["x"], y=hz["y"], z=hz["z"] + 0.5)
            txt.scale.z = 0.4
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = hz["class"]
            marker_array.markers.append(txt)

        # 5) Agents (spheres + arrows + trajectories)
        for idx, (agent_id, data) in enumerate(active_agents.items()):
            color = AGENT_TRAJECTORY_COLORS[idx % len(AGENT_TRAJECTORY_COLORS)]

            # Position sphere
            pos_marker = Marker()
            pos_marker.header.frame_id = "map"
            pos_marker.header.stamp = current_time_msg
            pos_marker.ns = f"{agent_id}_viz"
            pos_marker.id = mid
            mid += 1
            pos_marker.type = Marker.SPHERE
            pos_marker.action = Marker.ADD
            pos_marker.pose = data["pose"]
            pos_marker.scale = Vector3(x=0.4, y=0.4, z=0.4)
            pos_marker.color = color
            marker_array.markers.append(pos_marker)

            # Orientation arrow
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = current_time_msg
            arrow.ns = f"{agent_id}_viz"
            arrow.id = mid
            mid += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = data["pose"]
            arrow.scale = Vector3(x=0.8, y=0.15, z=0.15)
            arrow.color = color
            arrow.color.a = 0.8
            marker_array.markers.append(arrow)

            # Trajectory line
            traj = Marker()
            traj.header.frame_id = "map"
            traj.header.stamp = current_time_msg
            traj.ns = f"{agent_id}_viz"
            traj.id = mid
            mid += 1
            traj.type = Marker.LINE_STRIP
            traj.action = Marker.ADD
            traj.pose.orientation.w = 1.0
            traj.scale.x = 0.05
            traj.color = color
            traj.color.a = 0.4
            traj.points = data["trajectory_points"]
            marker_array.markers.append(traj)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = CommandCenterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
