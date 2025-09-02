# Filename: agent_simulator.py (ROS 2 Version)
import rclpy
from rclpy.node import Node
import random
import math
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler

# Constants remain the same
PUBLISH_RATE = 2.5
SITE_X_MIN, SITE_X_MAX = -11.43, 11.43
SITE_Y_MIN, SITE_Y_MAX = -6.86, 6.86
Z_HEIGHT_WORKER = 1.2192
Z_HEIGHT_UGV = 0.4572
PREDEFINED_HAZARDS = [
    {"class": "skid steer", "x": -8.5, "y": 4.2, "z": 0.1, "id": "sh_1"},
    {"class": "electric service panel", "x": 9.1, "y": -3.8, "z": 0.1, "id": "esp_1"},
    {"class": "cables", "x": -2.3, "y": 5.7, "z": 0.1, "id": "cab_1"},
    {"class": "table saw", "x": 6.8, "y": 2.1, "z": 0.1, "id": "ts_1"},
]
HAZARD_DETECTION_RADIUS = 1.5

class AgentSimulatorNode(Node):
    def __init__(self):
        # Initialize the Node with a unique name
        super().__init__(f'agent_simulator_{random.randint(100,999)}')
        
        # Declare and get parameters in the ROS 2 way
        self.declare_parameter('agent_id', 'agent_default')
        self.declare_parameter('movement_type', 'human')
        self.agent_id = self.get_parameter('agent_id').get_parameter_value().string_value
        self.agent_type = self.get_parameter('movement_type').get_parameter_value().string_value

        # --- Initialize agent state variables ---
        initial_z = Z_HEIGHT_WORKER if self.agent_type == "human" else Z_HEIGHT_UGV
        self.current_pos = Point(x=random.uniform(SITE_X_MIN + 1, SITE_X_MAX - 1),
                                 y=random.uniform(SITE_Y_MIN + 1, SITE_Y_MAX - 1), z=initial_z)
        self.current_yaw = random.uniform(-math.pi, math.pi)
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.last_decision_time = self.get_clock().now()
        self.decision_interval = rclpy.duration.Duration(seconds=random.uniform(3.0, 7.0))
        self.recently_reported = {}

        # Create publishers in the ROS 2 way
        self.pose_pub = self.create_publisher(PoseStamped, f'/{self.agent_id}/pose', 10)
        self.hazard_pub = self.create_publisher(String, f'/{self.agent_id}/hazard_detected', 10)
        
        # Create a timer to call the main loop
        self.create_timer(1.0 / PUBLISH_RATE, self.update_loop)
        self.get_logger().info(f"Agent '{self.agent_id}' ({self.agent_type}) starting.")

    def update_loop(self):
        dt = (1.0 / PUBLISH_RATE)
        self.update_movement_targets()
        self.move_agent(dt)
        self.publish_pose()
        self.check_and_report_hazards()

    def update_movement_targets(self):
        if (self.get_clock().now() - self.last_decision_time) > self.decision_interval:
            if self.agent_type == "ugv":
                self.linear_speed = random.uniform(0.3, 0.5)
                self.angular_speed = random.uniform(-0.3, 0.3)
            else: # human
                self.linear_speed = random.uniform(0.1, 0.4)
                self.angular_speed = random.uniform(-0.6, 0.6)
                if random.random() < 0.25: self.linear_speed = 0.0
            
            self.last_decision_time = self.get_clock().now()
            self.decision_interval = rclpy.duration.Duration(seconds=random.uniform(3.0, 7.0))

    def move_agent(self, dt):
        # Movement logic is the same, no ROS calls here
        self.current_yaw += self.angular_speed * dt
        self.current_pos.x += self.linear_speed * math.cos(self.current_yaw) * dt
        self.current_pos.y += self.linear_speed * math.sin(self.current_yaw) * dt
        
        if not (SITE_X_MIN < self.current_pos.x < SITE_X_MAX) or not (SITE_Y_MIN < self.current_pos.y < SITE_Y_MAX):
            self.current_yaw += math.pi # Turn around at boundary

        self.current_pos.x = max(SITE_X_MIN, min(SITE_X_MAX, self.current_pos.x))
        self.current_pos.y = max(SITE_Y_MIN, min(SITE_Y_MAX, self.current_pos.y))

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        q = quaternion_from_euler(0, 0, self.current_yaw)
        pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        pose_msg.pose.position = self.current_pos
        self.pose_pub.publish(pose_msg)

    def check_and_report_hazards(self):
        # This logic also remains largely the same
        for hazard in PREDEFINED_HAZARDS:
            distance = math.sqrt((self.current_pos.x - hazard["x"])**2 + (self.current_pos.y - hazard["y"])**2)
            if distance <= HAZARD_DETECTION_RADIUS:
                current_time = self.get_clock().now()
                if hazard['id'] not in self.recently_reported or \
                   (current_time - self.recently_reported[hazard['id']]) > rclpy.duration.Duration(seconds=10.0):
                    self.recently_reported[hazard['id']] = current_time
                    
                    hazard_msg_str = f"{hazard['id']};{hazard['class']};{hazard['x']:.2f};{hazard['y']:.2f};{hazard['z']:.2f}"
                    self.hazard_pub.publish(String(data=hazard_msg_str))

def main(args=None):
    rclpy.init(args=args)
    agent_simulator_node = AgentSimulatorNode()
    rclpy.spin(agent_simulator_node)
    agent_simulator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()