import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer

from keras.models import load_model
import argparse
import numpy as np

class PolicyPublisher(Node):

    def __init__(self, env, goal, task, goal_pose, decoy_pose, actions, model):
        super().__init__('policy_publisher')
        self.env = env
        self.goal = goal
        self.task = task
        self.goal_pose = goal_pose
        self.decoy_pose = decoy_pose
        self.actions = actions
        self.model = model

        self.policy = policy
        self.ee_pose = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # We publish end-effector commands to this topic
        self.publisher_ = self.create_publisher(TwistStamped, '/servo_server/delta_twist_cmds', 10)

        # This will get the next action from the policy every 0.1 seconds
        self.policy_timer = self.create_timer(0.5, self.policy_callback)

        # This will check the robot's current end-effector pose every 0.1 seconds
        self.tf_timer = self.create_timer(0.1, self.eef_callback)


    def eef_callback(self):
        # Look up the end-effector pose using the transform tree
        try:
            t = self.tf_buffer.lookup_transform(
                "link_base", #to_frame_rel,
                "link_eef", #from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not get transform: {ex}')
            rclpy.time.sleep(1)
            return

        self.ee_pose = t.transform.translation


    def policy_callback(self):
        if self.ee_pose is None:
            print("Waiting for ee pose...")
            return

        current_pose = np.array([self.ee_pose.x, self.ee_pose.y, self.ee_pose.z])
        reward = []
        for action in self.actions:
            future_pose = current_pose + action
            input_state = np.append(future_pose, self.goal_pose)
            input_state = np.append(input_state, self.decoy_pose)
            r = self.model.predict(input_state)[self.task]
            reward.append(r)
        index = np.argmax(reward)
        result = self.actions[index]
        
        # Convert the action vector into a Twist message
        twist = TwistStamped()
        twist.twist.linear.x = result[0]
        twist.twist.linear.y = result[1]
        twist.twist.linear.z = result[2]
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0
        twist.header.frame_id = "link_base"
        twist.header.stamp = self.get_clock().now().to_msg()

        self.publisher_.publish(twist)

def main(args=None):
    env1_goal = np.array([[-0.035, 0.142, 1.245], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.142, 1.245]])
    env2_goal = np.array([[-0.035, 0.042, 1.345], [-0.15, 0.042, 1.245], [-0.266, 0.142, 1.345], [-0.383, 0.042, 1.245]])
    env3_goal = np.array([[-0.035, 0.042, 1.145], [-0.15, 0.142, 1.245], [-0.266, 0.142, 1.245], [-0.383, 0.042, 1.145]])
    goals = np.array([env1_goal, env2_goal, env3_goal])
    env1_decoy = np.array([[-0.383379, 0.165331, 1.14576], [-0.183635, 0.239076, 1.31077], [-0.153111, 0.110957, 1.29395], [0.022497, 0.180345, 1.1708]])
    env2_decoy = np.array([[-0.416498, 0.078665, 1.32583], [-0.196313, 0.310306, 1.34149], [-0.133594, 0.184351, 1.34316], [-0.074792, 0.148929, 1.43017]])
    env3_decoy = np.array([[-0.344683, 0.024459, 1.20518], [-0.16924, 0.226221, 1.2826], [-0.099357, 0.12819, 1.32999], [-0.024455, 0.073603, 1.2148]])
    decoys = np.array([env1_decoy, env2_decoy, env3_decoy])

    parser = argparse.ArgumentParser()

    parser.add_argument("-env", "--environment", dest = "env", default = 1, help="Environment number", type=int)
    parser.add_argument("-g", "--goal", dest = "g", default = 1, help="Goal number", type=int)
    parser.add_argument("-task", "--task",dest ="task", default = 1, help="Task number", type=int)

    args = parser.parse_args()

    env = args.env - 1 # 1~3
    g = args.g - 1 # 1~8
    task = args.task - 1 # 1~3

    # Sample policy that will move the end-effector in a box-like shape
    model = load_model('my_model.h5')

    action_x = [-0.02, 0, 0.02]
    action_y = [-0.02, 0, 0.02]
    action_z = [-0.02, 0, 0.02]
    actions = []
    for a_x in action_x:
        for a_y in action_y:
            for a_z in action_z:
                actions.append([a_x, a_y, a_z])
    actions = np.array(actions)



    rclpy.init(args=args)

    policy_publisher = PolicyPublisher(env, g, task, goals[env][g], decoys[env], actions, model)

    rclpy.spin(policy_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    policy_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
