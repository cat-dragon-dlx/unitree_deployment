import numpy as np
import rospy

import torch
import torch.nn as nn
from torch.distributions import Normal
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import LegsCmd
from std_msgs.msg import Bool  # 导入 std_msgs/Bool 消息类型

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class UnitreeRosHandler:
    """ This is the handler that works for ROS on unitree, from issac lab to real robot. """
    def __init__(self, 
                 robot_namespace="a112138",
                 low_state_topic="/low_state",
                 legs_cmd_topic="/legs_cmd",
                 device=torch.device('cpu'),
        ):
        """
        NOTE:
            * Must call start_ros() before using this class's get_obs() and send_action()
            * Joint order of simulation and of real A1 protocol are different, see dof_names
            * We store all joints values in the order of simulation in this class
        """

        self.device = device
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        self.legs_cmd_topic = legs_cmd_topic

        # cfg
        self.action_scale = 0.25
        self.d_gains = 1.0
        self.p_gains = 40.0
        self.history_len = 10
        self.obs_scales = {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
        }
        self.commands_scale = torch.tensor([self.obs_scales['lin_vel'], self.obs_scales['lin_vel'], self.obs_scales['ang_vel']], device=self.device)

        # init buffers
        self.actions = torch.zeros((1, 12), dtype=torch.float32, device=device)
        self.joint_pos = torch.zeros_like(self.actions)
        self.joint_vel = torch.zeros_like(self.actions)
        self.command_buf = torch.zeros((1, 3,), dtype=torch.float32, device=device)
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.obs_history_buf = torch.zeros(1, self.history_len, 45, device=self.device, dtype=torch.float)

        """
        Joint order (applied to both policy trained in sim and the real robot):
        [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        joint_map[i] = real_joint_order.index(sim_joint_order[i])
        e.g. joint_map[0] = real_joint_order.index("FL_hip_joint") = 3
        """

        default_joint_pos = [ 
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 1.0, -1.5,
            0.1, 1.0, -1.5
        ]
        self.default_joint_pos = torch.tensor(default_joint_pos, dtype=torch.float32, device=device)

        # limits
        self.action_limits_low, self.action_limits_high = self._get_action_limits()
        self.torque_limits= torch.tensor([
            20.0, 55.0, 55.0,
            20.0, 55.0, 55.0,
            20.0, 55.0, 55.0,
            20.0, 55.0, 55.0,
        ], dtype=torch.float32, device=device)
        # get ROS params for hardware configs
        self.joint_limits_high = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_max".format(s)) \
            for s in ["hip", "thigh", "calf",
                      "hip", "thigh", "calf",
                      "hip", "thigh", "calf",
                      "hip", "thigh", "calf",]
        ])
        self.joint_limits_low = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_min".format(s)) \
            for s in ["hip", "thigh", "calf",
                      "hip", "thigh", "calf",
                      "hip", "thigh", "calf",
                      "hip", "thigh", "calf",]
        ])

        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), dtype=torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1

    def _get_action_limits(self):
        """ 
        sdk_dof_range = dict(  # copied from a1_const.h from unitree_legged_sdk
            Hip_max=0.802,
            Hip_min=-0.802,
            Thigh_max=4.19,
            Thigh_min=-1.05,
            Calf_max=-0.916,
            Calf_min=-2.7,
        )
        """
        joint_pos_range = [
            (-0.802, 0.802), (-1.05, 4.19), (-2.7, -0.916),
            (-0.802, 0.802), (-1.05, 4.19), (-2.7, -0.916),
            (-0.802, 0.802), (-1.05, 4.19), (-2.7, -0.916),
            (-0.802, 0.802), (-1.05, 4.19), (-2.7, -0.916),
        ]
        redundancy_factor = 0.95
        joint_pos_redundancy = [redundancy_factor*(joint_pos_range[i][1] - joint_pos_range[i][0]) for i in range(12)]
        action_limits_low = [(joint_pos_range[i][0] - joint_pos_redundancy[i] - self.default_joint_pos[i])/self.action_scale for i in range(12)]
        action_limits_high = [(joint_pos_range[i][1] + joint_pos_redundancy[i] - self.default_joint_pos[i])/self.action_scale for i in range(12)]
        action_limits_low = torch.tensor(action_limits_low, dtype=torch.float32, device=self.device)
        action_limits_high = torch.tensor(action_limits_high, dtype=torch.float32, device=self.device)

        return action_limits_low, action_limits_high

    def start_ros(self):
        # initialze several buffers so that the system works even without message update.
        # self.low_state_buffer = LowState() # not initialized, let input message update it.
        self.legs_cmd_publisher = rospy.Publisher(
            self.robot_namespace + self.legs_cmd_topic,
            LegsCmd,
            queue_size=1,
        )
        self.low_state_subscriber = rospy.Subscriber(
            self.robot_namespace + self.low_state_topic,
            LowState,
            self.update_low_state,
            queue_size=1,
        )
        self.stand_up_publisher = rospy.Publisher(
            self.robot_namespace + "/stop_standup",
            Bool,
            queue_size=1,
        )
    
    def publish_stop_stand_up(self, stop_standup):
        """
        After we modify unitree_ros_real, we can use this function to stop stand up.
        Args:
            stop_standup: ,if it's true, ros node will reinitialize position protection.
        """
        msg = Bool() 
        msg.data = stop_standup 
        self.stand_up_publisher.publish(msg)  
    
    def wait_untill_ros_working(self):
        """used after start ros"""
        rate = rospy.Rate(100)
        while not hasattr(self, "low_state_buffer"):
            rate.sleep()
        rospy.loginfo("UnitreeA1Real.low_state_buffer acquired, stop waiting.")

    def stand_up(self, ros_rate, angle_tolerance, kp, kd, warmup_timesteps, device=torch.device("cpu")):
        """
        Args:
            warmup_timesteps: the number of timesteps to linearly increase the target position
        """
        rospy.loginfo("Robot standing up, please wait...")

        target_pos = torch.zeros((1, 12), device=device, dtype=torch.float32)
        standup_timestep_i = 0
        while not rospy.is_shutdown():
            joint_pos = [self.low_state_buffer.motorState[i].q for i in range(12)]
            diff = [self.default_joint_pos[i].item() - joint_pos[i] for i in range(12)]
            if all([abs(i) < angle_tolerance for i in diff]):
                break
            direction = [1 if i > 0 else -1 for i in diff]
            if standup_timestep_i < warmup_timesteps:
                direction = [standup_timestep_i / warmup_timesteps * i for i in direction]

            print(f"{standup_timestep_i}: max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
            for i in range(12):
                target_pos[0, i] = joint_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else self.default_joint_pos[i]
            self.publish_legs_cmd(target_pos, kp=kp, kd=kd)
            ros_rate.sleep()
            standup_timestep_i += 1
        
        rospy.loginfo("Robot stood up! Press R1 to continue...")
        self.wait_for_R1(ros_rate, kp, kd)
    
    def wait_for_R1(self, ros_rate, kp, kd):
        while not rospy.is_shutdown():
            if self.low_state_buffer.wirelessRemote.btn.components.R1:
                break
            if self.low_state_buffer.wirelessRemote.btn.components.L2 or self.low_state_buffer.wirelessRemote.btn.components.R2:
                self.publish_legs_cmd(self.default_joint_pos.unsqueeze(0), kp=0, kd=0.5)
                rospy.signal_shutdown("Controller send stop signal, exiting")
                exit(0)
            self.publish_legs_cmd(self.default_joint_pos.unsqueeze(0), kp=kp, kd=kd)
            ros_rate.sleep()

    def get_obs(self):
        low_state = self.low_state_buffer

        base_ang_vel = torch.tensor(low_state.imu.gyroscope, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(
            torch.tensor(low_state.imu.quaternion).unsqueeze(0),
            self.gravity_vec,
        ).to(self.device)
        self.joint_pos = joint_pos = torch.tensor(
            [low_state.motorState[i].q for i in range(12)], 
            dtype=torch.float32, device=self.device).unsqueeze(0)
        self.joint_vel = joint_vel = torch.tensor(
            [low_state.motorState[i].dq for i in range(12)], 
            dtype=torch.float32, device=self.device).unsqueeze(0)

        obs_buf = torch.cat((
            base_ang_vel * self.obs_scales['ang_vel'],
            projected_gravity,
            self.command_buf * self.commands_scale,
            (joint_pos - self.default_joint_pos) * self.obs_scales['dof_pos'],
            joint_vel * self.obs_scales['dof_vel'],
            self.actions
        ), dim=-1)

        self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(1, -1)], dim=-1)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return self.obs_buf
    
    def send_action(self, actions):
        """ The function that send commands to the real robot.
        """
        self.actions = self.clip_action_before_scale(actions)
        actions_scaled = actions * self.action_scale
        actions_scaled_clipped = self.clip_by_torque_limit(actions_scaled)
        robot_coordinates_action = actions_scaled_clipped + self.default_joint_pos.unsqueeze(0)
        motor_state, legs_cmd = self.publish_legs_cmd(robot_coordinates_action)
        return self.actions, actions_scaled, actions_scaled_clipped, motor_state, legs_cmd

    def clip_action_before_scale(self, actions):
        actions = torch.clip(actions, self.action_limits_low, self.action_limits_high)
        return actions

    def clip_by_torque_limit(self, actions_scaled):
        p_limits_low = (-self.torque_limits) + self.d_gains*self.joint_vel
        p_limits_high = (self.torque_limits) + self.d_gains*self.joint_vel
        actions_low = (p_limits_low/self.p_gains) - self.default_joint_pos + self.joint_pos
        actions_high = (p_limits_high/self.p_gains) - self.default_joint_pos + self.joint_pos
        return torch.clip(actions_scaled, actions_low, actions_high)
    
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
        low_state = self.low_state_buffer

        for i in range(12):
            legs_cmd.cmd[i].mode = 10
            legs_cmd.cmd[i].q = robot_coordinates_action[0, i] 
            legs_cmd.cmd[i].dq = 0.
            legs_cmd.cmd[i].tau = 0.
            legs_cmd.cmd[i].Kp = self.p_gains if kp is None else kp
            legs_cmd.cmd[i].Kd = self.d_gains if kd is None else kd
            # if i == 2:
            #     print(legs_cmd.cmd[i])
            
        self.legs_cmd_publisher.publish(legs_cmd)
        return low_state.motorState, legs_cmd.cmd

    """ ROS callbacks and handlers that update the buffer """
    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg
        self.command_buf[0, 0] = self.low_state_buffer.wirelessRemote.ly
        self.command_buf[0, 1] = -self.low_state_buffer.wirelessRemote.lx # right-moving stick is positive
        self.command_buf[0, 2] = -self.low_state_buffer.wirelessRemote.rx # right-moving stick is positive
        # set the command to zero if it is too small
        if torch.norm(self.command_buf[0, :2]) < 0.1:
            self.command_buf[0, :2] = 0.
        if torch.abs(self.command_buf[0, 2]) < 0.1:
            self.command_buf[0, 2] = 0.
