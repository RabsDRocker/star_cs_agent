#!/usr/bin/env python3

# Standard ROS and system libraries
import rclpy
from rclpy.node import Node
import threading
import json
import os
import requests # Used to make HTTP requests to our AI server

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

class CommandCenterNode(Node):
    def __init__(self):
        """Initializes the command center node."""
        super().__init__('command_center_node')
        
        # --- Data Storage ---
        self.agents = {}
        self.data_lock = threading.Lock()
        
        # --- Digital Twin Data ---
        self.site_plan = self.load_site_plan()
        self.agent_zones = {agent_id: None for agent_id in AGENT_IDS}

        # --- ROS Subscribers ---
        for agent_id in AGENT_IDS:
            self.agents[agent_id] = {'pose': None, 'trajectory_points': []}
            # Use a lambda function to pass the agent_id to the callback
            self.create_subscription(
                PoseStamped,
                f'/{agent_id}/pose',
                lambda msg, aid=agent_id: self.agent_pose_callback(msg, aid),
                10)

        # --- ROS Publisher ---
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        
        # --- Main Loop Timer ---
        self.create_timer(0.2, self.visualization_loop)
        self.get_logger().info("Command Center Initialized and running.")

    def load_site_plan(self):
        """Loads the site plan JSON using the robust ROS 2 package path finder."""
        try:
            package_share_dir = get_package_share_directory('star_cs_agent')
            plan_path = os.path.join(package_share_dir, 'data', 'site_plan.json')
            self.get_logger().info(f"Loading site plan from: {plan_path}")
            with open(plan_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to load site plan: {e}")
            return {}

    def is_point_in_zone(self, point, polygon):
        """Checks if a 2D point is inside a 2D polygon."""
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
        """Updates an agent's position when a new pose is received."""
        with self.data_lock:
            self.agents[agent_id]['pose'] = msg.pose
            self.agents[agent_id]['trajectory_points'].append(msg.pose.position)
            if len(self.agents[agent_id]['trajectory_points']) > 150:
                self.agents[agent_id]['trajectory_points'].pop(0)

    def visualization_loop(self):
        """The main loop for checking events and publishing visualizations."""
        with self.data_lock:
            active_agents = {aid: data.copy() for aid, data in self.agents.items() if data['pose'] is not None}
        
        current_time_msg = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        marker_id_counter = 0

        # --- Check for Zone Entry Events and Call AI Agent ---
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
                        "timestamp": current_time_msg.sec,
                        "event_type": "Zone Entry", "agent_id": agent_id, "zone_id": current_zone,
                        "details": f"Agent {agent_id} entered the '{self.site_plan[current_zone]['description']}'",
                        "agent_location": {"x": current_pos.x, "y": current_pos.y, "z": current_pos.z},
                        "all_agents_locations": {aid: {"location": {"x": a_data['pose'].position.x, "y": a_data['pose'].position.y, "z": a_data['pose'].position.z}} for aid, a_data in active_agents.items()}
                    }
                    try:
                        requests.post("http://127.0.0.1:5001/analyze", json=event_payload, timeout=10)
                    except requests.exceptions.RequestException as e:
                        self.get_logger().error(f"Could not connect to Reasoning API Server: {e}")

        # --- COMPLETE VISUALIZATION CODE ---
        # Visualize Hazard Zones
        for zone_id, zone_data in self.site_plan.items():
            zone_marker = Marker(); zone_marker.header.frame_id = "map"; zone_marker.header.stamp = current_time_msg
            zone_marker.ns = "hazard_zones"; zone_marker.id = marker_id_counter; marker_id_counter += 1
            zone_marker.type = Marker.LINE_STRIP; zone_marker.action = Marker.ADD; zone_marker.pose.orientation.w = 1.0
            zone_marker.scale.x = 0.1; zone_marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.6) # Yellow
            for p in zone_data['polygon_coords']: zone_marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
            zone_marker.points.append(Point(x=float(zone_data['polygon_coords'][0][0]), y=float(zone_data['polygon_coords'][0][1]), z=0.0))
            marker_array.markers.append(zone_marker)
        
        # Visualize Agents and Trajectories
        for agent_id, data in active_agents.items():
            agent_color = AGENT_TRAJECTORY_COLORS[AGENT_IDS.index(agent_id) % len(AGENT_TRAJECTORY_COLORS)]
            pos_marker = Marker(); pos_marker.header.frame_id = "map"; pos_marker.header.stamp = current_time_msg
            pos_marker.ns = f"{agent_id}_viz"; pos_marker.id = marker_id_counter; marker_id_counter += 1
            pos_marker.type = Marker.SPHERE; pos_marker.action = Marker.ADD; pos_marker.pose = data['pose']
            pos_marker.scale = Vector3(x=0.4, y=0.4, z=0.4); pos_marker.color = agent_color
            marker_array.markers.append(pos_marker)
            
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