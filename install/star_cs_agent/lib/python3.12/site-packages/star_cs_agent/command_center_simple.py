#!/usr/bin/env python3

# Filename: command_center_simple.py (Final Enhanced Version)
import rclpy
from rclpy.node import Node
import threading
import json
import os
import requests
import math

# A ROS 2 utility to find the path to package files
from ament_index_python.packages import get_package_share_directory

# ROS message types
from geometry_msgs.msg import Point, PoseStamped, Vector3
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# --- Configuration Constants ---
AGENT_IDS = ["agent1", "agent2", "agent3"]
AGENT_TRAJECTORY_COLORS = [
    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
    ColorRGBA(r=0.0, g=0.6, b=0.0, a=1.0),  # Green
    ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)   # Blue
]
SITE_X_MIN, SITE_X_MAX = -11.43, 11.43
SITE_Y_MIN, SITE_Y_MAX = -6.86, 6.86
PREDEFINED_HAZARDS = [
    {"class": "skid steer", "x": -8.5, "y": 4.2, "z": 0.1, "id": "sh_1"},
    {"class": "electric service panel", "x": 9.1, "y": -3.8, "z": 0.1, "id": "esp_1"},
    {"class": "cables", "x": -2.3, "y": 5.7, "z": 0.1, "id": "cab_1"},
    {"class": "table saw", "x": 6.8, "y": 2.1, "z": 0.1, "id": "ts_1"},
    {"class": "rebar w cap", "x": -5.2, "y": -4.9, "z": 0.1, "id": "rwc_1"},
    {"class": "rebar w-o cap", "x": 3.7, "y": -1.5, "z": 0.1, "id": "rwoc_1"},
]

class CommandCenterNode(Node):
    def __init__(self):
        super().__init__('command_center_node')
        self.agents = {}
        self.data_lock = threading.Lock()
        self.site_plan = self.load_site_plan()
        self.agent_zones = {agent_id: None for agent_id in AGENT_IDS}

        for agent_id in AGENT_IDS:
            self.agents[agent_id] = {'pose': None, 'trajectory_points': []}
            self.create_subscription(
                PoseStamped, f'/{agent_id}/pose',
                lambda msg, aid=agent_id: self.agent_pose_callback(msg, aid), 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.create_timer(0.2, self.visualization_loop)
        self.get_logger().info("Enhanced Command Center Initialized and running.")

    def load_site_plan(self):
        package_share_dir = get_package_share_directory('star_cs_agent')
        plan_path = os.path.join(package_share_dir, 'data', 'site_plan.json')
        try:
            with open(plan_path, 'r') as f:
                self.get_logger().info(f"Loading site plan from: {plan_path}")
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load site plan: {e}")
            return {}

    def is_point_in_zone(self, point, polygon):
        x, y = point.x, point.y; n = len(polygon); inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def agent_pose_callback(self, msg, agent_id):
        with self.data_lock:
            self.agents[agent_id]['pose'] = msg.pose
            self.agents[agent_id]['trajectory_points'].append(msg.pose.position)
            if len(self.agents[agent_id]['trajectory_points']) > 150:
                self.agents[agent_id]['trajectory_points'].pop(0)

    def visualization_loop(self):
        with self.data_lock:
            active_agents = {aid: data.copy() for aid, data in self.agents.items() if data['pose'] is not None}
        
        current_time_msg = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        marker_id_counter = 0

        # Check for Zone Entry Events and Call AI Agent
        for agent_id, data in active_agents.items():
            current_pos = data['pose'].position
            previous_zone = self.agent_zones[agent_id]
            current_zone = None
            for zone_id, zone_data in self.site_plan.items():
                if self.is_point_in_zone(current_pos, zone_data['polygon_coords']):
                    current_zone = zone_id; break
            if current_zone != previous_zone:
                self.agent_zones[agent_id] = current_zone
                if current_zone is not None:
                    self.get_logger().warn(f"SAFETY EVENT: Agent '{agent_id}' entered high-risk zone '{current_zone}'! Triggering AI Agent.")
                    event_payload = {
                        "timestamp": current_time_msg.sec, "event_type": "Zone Entry", "agent_id": agent_id, "zone_id": current_zone,
                        "details": f"Agent {agent_id} entered the '{self.site_plan[current_zone]['description']}'",
                        "agent_location": {"x": current_pos.x, "y": current_pos.y, "z": current_pos.z},
                        "all_agents_locations": {aid: {"location": {"x": a_data['pose'].position.x, "y": a_data['pose'].position.y, "z": a_data['pose'].position.z}} for aid, a_data in active_agents.items()}
                    }
                    try:
                        requests.post("http://127.0.0.1:5001/analyze", json=event_payload, timeout=10)
                    except requests.exceptions.RequestException as e:
                        self.get_logger().error(f"Could not connect to Reasoning API Server: {e}")

        # --- COMPLETE VISUALIZATION CODE ---
        # Site Grid Marker
        grid_marker = Marker(); grid_marker.header.frame_id = "map"; grid_marker.header.stamp = current_time_msg
        grid_marker.ns = "site_layout"; grid_marker.id = marker_id_counter; marker_id_counter += 1
        grid_marker.type = Marker.LINE_LIST; grid_marker.action = Marker.ADD
        grid_marker.pose.orientation.w = 1.0
        grid_marker.scale.x = 0.05; grid_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4)
        grid_cell_size = 1.0
        x = SITE_X_MIN
        while x <= SITE_X_MAX:
            grid_marker.points.append(Point(x=x, y=SITE_Y_MIN, z=0.01))
            grid_marker.points.append(Point(x=x, y=SITE_Y_MAX, z=0.01))
            x += grid_cell_size
        y = SITE_Y_MIN
        while y <= SITE_Y_MAX:
            grid_marker.points.append(Point(x=SITE_X_MIN, y=y, z=0.01))
            grid_marker.points.append(Point(x=SITE_X_MAX, y=y, z=0.01))
            y += grid_cell_size
        marker_array.markers.append(grid_marker)

        # Pre-defined Hazard Markers
        for hazard in PREDEFINED_HAZARDS:
            haz_marker = Marker(); haz_marker.header.frame_id = "map"; haz_marker.header.stamp = current_time_msg
            haz_marker.ns = "static_hazards"; haz_marker.id = marker_id_counter; marker_id_counter += 1
            haz_marker.type = Marker.CYLINDER; haz_marker.action = Marker.ADD
            haz_marker.pose.position = Point(x=hazard['x'], y=hazard['y'], z=hazard['z'])
            haz_marker.pose.orientation.w = 1.0
            haz_marker.scale = Vector3(x=0.8, y=0.8, z=0.2); 
            haz_marker.color = ColorRGBA(r=0.8, g=0.1, b=0.1, a=0.8)
            marker_array.markers.append(haz_marker)
            text_marker = Marker(); text_marker.header.frame_id = "map"; text_marker.header.stamp = current_time_msg
            text_marker.ns = "hazard_labels"; text_marker.id = marker_id_counter; marker_id_counter += 1
            text_marker.type = Marker.TEXT_VIEW_FACING; text_marker.action = Marker.ADD
            text_marker.pose.position = Point(x=hazard['x'], y=hazard['y'], z=hazard['z'] + 0.5)
            text_marker.scale.z = 0.4; text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text = hazard['class']
            marker_array.markers.append(text_marker)

        # Agent Markers with Orientation
        for agent_id, data in active_agents.items():
            agent_color = AGENT_TRAJECTORY_COLORS[AGENT_IDS.index(agent_id) % len(AGENT_TRAJECTORY_COLORS)]
            pos_marker = Marker(); pos_marker.header.frame_id = "map"; pos_marker.header.stamp = current_time_msg
            pos_marker.ns = f"{agent_id}_viz"; pos_marker.id = marker_id_counter; marker_id_counter += 1
            pos_marker.type = Marker.SPHERE; pos_marker.action = Marker.ADD; pos_marker.pose = data['pose']
            pos_marker.scale = Vector3(x=0.4, y=0.4, z=0.4); pos_marker.color = agent_color
            marker_array.markers.append(pos_marker)
            
            arrow_marker = Marker(); arrow_marker.header.frame_id = "map"; arrow_marker.header.stamp = current_time_msg
            arrow_marker.ns = f"{agent_id}_viz"; arrow_marker.id = marker_id_counter; marker_id_counter += 1
            arrow_marker.type = Marker.ARROW; arrow_marker.action = Marker.ADD; arrow_marker.pose = data['pose']
            arrow_marker.scale = Vector3(x=0.8, y=0.15, z=0.15); arrow_marker.color = agent_color
            marker_array.markers.append(arrow_marker)
            
            traj_marker = Marker(); traj_marker.header.frame_id = "map"; traj_marker.header.stamp = current_time_msg
            traj_marker.ns = f"{agent_id}_viz"; traj_marker.id = marker_id_counter; marker_id_counter += 1
            traj_marker.type = Marker.LINE_STRIP; traj_marker.action = Marker.ADD; traj_marker.pose.orientation.w = 1.0
            traj_marker.scale.x = 0.05; traj_marker.color = agent_color; traj_marker.color.a = 0.4
            traj_marker.points = data['trajectory_points']
            marker_array.markers.append(traj_marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    command_center_node = CommandCenterNode()
    rclpy.spin(command_center_node)
    command_center_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
