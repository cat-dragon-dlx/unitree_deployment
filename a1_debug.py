import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import os
import os.path as osp
from collections import OrderedDict
from typing import Tuple

import rospy
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import LegsCmd
from unitree_legged_msgs.msg import Float32MultiArrayStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import ros_numpy
import json
import os

class UnitreeDebug:
    """ This is the handler that works for ROS 1 on unitree. """
    def __init__(self,
            robot_namespace= "a112138",
            low_state_topic="/low_state",
            legs_cmd_topic="/legs_cmd",
        ):
        self.robot_namespace = robot_namespace
        self.legs_cmd_topic = legs_cmd_topic
        self.last_saved_data = None  # 用于判断是否重复
        self.save_path = "contactData.json"  # 保存的文件路径
        self.threshold=10 
        self.low_state_topic=low_state_topic
        default_joint_pos = [ # in simulation joint order
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5
        ]
        self.default_joint_pos = torch.tensor(default_joint_pos, dtype=torch.float32)

        self.joint_map = [
            3, 0, 9, 6,
            4, 1, 10, 7,
            5, 2, 11, 8
        ]

        # get ROS params for hardware configs
        self.joint_limits_high = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_max".format(s)) \
            for s in ["hip", "hip", "hip", "hip", 
                      "thigh", "thigh", "thigh", "thigh", 
                      "calf", "calf", "calf", "calf"]
        ])
        self.joint_limits_low = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_min".format(s)) \
            for s in ["hip", "hip", "hip", "hip", 
                      "thigh", "thigh", "thigh", "thigh", 
                      "calf", "calf", "calf", "calf"]
        ])

        self.init_foot_force = None

    def set_init_foot_force(self, init_foot_force):
        self.init_foot_force = np.array(init_foot_force)
        print("set init_foot_force:", init_foot_force)

    def start_ros(self):
       
        self.base_position_buffer = torch.zeros((1, 3), requires_grad= False)
        self.legs_cmd_publisher = rospy.Publisher(
            self.robot_namespace + self.legs_cmd_topic,
            LegsCmd,
            queue_size= 1,
        )
        # self.debug_publisher = rospy.Publisher(
        #     "/DNNmodel_debug",
        #     Float32MultiArray,
        #     queue_size= 1,
        # )
        # NOTE: this launches the subscriber callback function
        self.low_state_subscriber = rospy.Subscriber(
            self.robot_namespace + self.low_state_topic,
            LowState,
            self.update_low_state,
            queue_size= 1,
        )
        
    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg

    def get_foot_force(self):
        default_foot_force = 10.
        low_state = self.low_state_buffer
        foot_force = np.array(low_state.footForce) - self.init_foot_force + default_foot_force
        foot_force = foot_force.tolist()
        contact = torch.tensor(foot_force) > 1.0
        contact = contact.numpy().tolist()

        # 生成数据结构
        current_data = {
            "footForce": foot_force,
            "contact": contact
        }

        print(f"footForce: FR:{foot_force[0]}, FL:{foot_force[1]}, RR:{foot_force[2]}, RL:{foot_force[3]}")
        print(f"contact: FR:{contact[0]}, FL:{contact[1]}, RR:{contact[2]}, RL:{contact[3]}")
        # 如果是第一次保存数据，直接保存
        if self.last_saved_data is None:
            self.last_saved_data = current_data
            with open(self.save_path, "a") as f:
                f.write(json.dumps(current_data) + "\n")
            return current_data

        # 判断 footForce 和 contact 是否超过阈值
        foot_force_diff = np.max(np.abs(np.array(current_data["footForce"]) - np.array(self.last_saved_data["footForce"])))
        contact_diff = np.sum(np.array(current_data["contact"]) != np.array(self.last_saved_data["contact"]))

        # 只要 footForce 或 contact 中的任意一个差异超过阈值，就记录数据
        if foot_force_diff > self.threshold or contact_diff > 0:
            self.last_saved_data = current_data
            with open(self.save_path, "a") as f:
                f.write(json.dumps(current_data) + "\n")

        return current_data

    def wait_untill_ros_working(self):
        rate = rospy.Rate(100)
        while not hasattr(self, "low_state_buffer"):
            rate.sleep()
        rospy.loginfo("UnitreeA1Real.low_state_buffer acquired, stop waiting.")

    def publish_legs_cmd(self, robot_coordinates_action, kp=None, kd=None):
        """ publish the joint position directly to the robot. NOTE: The joint order from input should
        be in simulation order. The value should be absolute value rather than related to dof_pos.
        """
        robot_coordinates_action = torch.clip(
            robot_coordinates_action.cpu(),
            self.joint_limits_low,
            self.joint_limits_high,
        )
        legs_cmd = LegsCmd()
        for sim_joint_idx in range(12):
            real_joint_idx = self.joint_map[sim_joint_idx]
            legs_cmd.cmd[real_joint_idx].mode = 10
            legs_cmd.cmd[real_joint_idx].q = robot_coordinates_action[0, sim_joint_idx] 
            legs_cmd.cmd[real_joint_idx].dq = 0.
            legs_cmd.cmd[real_joint_idx].tau = 0.
            legs_cmd.cmd[real_joint_idx].Kp = self.p_gains if kp is None else kp
            legs_cmd.cmd[real_joint_idx].Kd = self.d_gains if kd is None else kd
        self.legs_cmd_publisher.publish(legs_cmd)

def standup_procedure(handler, ros_rate, angle_tolerance=0.1, 
                      kp=None, kd=None, warmup_timesteps=25, device=torch.device("cpu")):
    """
    Args:
        warmup_timesteps: the number of timesteps to linearly increase the target position
    """
    rospy.loginfo("Robot standing up, please wait ...")

    target_pos = torch.zeros((1, 12), device=device, dtype=torch.float32)
    standup_timestep_i = 0
    while not rospy.is_shutdown():
        joint_pos = [handler.low_state_buffer.motorState[handler.joint_map[i]].q for i in range(12)]
        diff = [handler.default_joint_pos[i].item() - joint_pos[i] for i in range(12)]
        direction = [1 if i > 0 else -1 for i in diff]
        if standup_timestep_i < warmup_timesteps:
            direction = [standup_timestep_i / warmup_timesteps * i for i in direction]
        if all([abs(i) < angle_tolerance for i in diff]):
            break
        print("max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
        for i in range(12):
            target_pos[0, i] = joint_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else handler.default_joint_pos[i]
        handler.publish_legs_cmd(target_pos, kp=kp, kd=kd)
        ros_rate.sleep()
        standup_timestep_i += 1

    rospy.loginfo("Robot stood up! press R1 on the remote control to continue ...")
    # record init foot force
    handler.set_init_foot_force(handler.low_state_buffer.footForce)
    while not rospy.is_shutdown():
        if handler.low_state_buffer.wirelessRemote.btn.components.R1:
            break
        if handler.low_state_buffer.wirelessRemote.btn.components.L2 or handler.low_state_buffer.wirelessRemote.btn.components.R2:
            handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=0, kd=0.5)
            rospy.signal_shutdown("Controller send stop signal, exiting")
            exit(0)
        handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=kp, kd=kd)
        ros_rate.sleep()
    rospy.loginfo("Robot standing up procedure finished!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
            type= str,
            default= "/a112138",                    
        )
    args = parser.parse_args()
    log_level = rospy.DEBUG 
    rospy.init_node("a1_debug" , log_level=log_level)

    handler = UnitreeDebug(robot_namespace= args.namespace)
    handler.start_ros()
    handler.wait_untill_ros_working()
    duration = 0.02 # in seconds
    rate = rospy.Rate(1 / duration)
    standup_procedure(handler, rate, 
                      angle_tolerance=0.2,
                      kp=40,
                      kd=0.5,
                      warmup_timesteps=50,
                     )
    
    standup_timestep_i = 0
    target_pos = torch.zeros((1, 12), dtype=torch.float32)
    angle_tolerance = 0.1
    kp=40; kd=0.5
    while not rospy.is_shutdown():
        joint_pos = [handler.low_state_buffer.motorState[handler.joint_map[i]].q for i in range(12)]
        diff = [handler.default_joint_pos[i].item() - joint_pos[i] for i in range(12)]
        direction = [1 if i > 0 else -1 for i in diff]
        print("max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
        for i in range(12):
            target_pos[0, i] = joint_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else handler.default_joint_pos[i]
        handler.publish_legs_cmd(target_pos, kp=kp, kd=kd)
        rate.sleep()
        standup_timestep_i += 1
        handler.get_foot_force()
        if handler.low_state_buffer.wirelessRemote.btn.components.down:
            rospy.loginfo_throttle(0.1, "model reset")
        if handler.low_state_buffer.wirelessRemote.btn.components.L2 or handler.low_state_buffer.wirelessRemote.btn.components.R2:
            handler.publish_legs_cmd(handler.default_dof_pos.unsqueeze(0), kp= 2, kd= 0.5)