import torch
import numpy as np
import rospy
import ros_numpy
from collections import OrderedDict

# from unitree
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import LegsCmd
from unitree_legged_msgs.msg import Float32MultiArrayStamped
# from ros
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import torch.nn.functional as F
from torch.autograd import Variable


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data

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

class UnitreeA1Real:
    """ This is the handler that works for ROS 1 on unitree. """
    def __init__(self, 
                 env_cfg,
                 agent_cfg,
                 lin_vel_estimator,
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

        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.device = device
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        self.legs_cmd_topic = legs_cmd_topic

        # from cfg
        self.action_scale = env_cfg['action_scale']
        self.d_gains = env_cfg['scene']['robot']['actuators']['base_legs']['damping']
        self.p_gains = env_cfg['scene']['robot']['actuators']['base_legs']['stiffness']

        self.lin_vel_estimator = lin_vel_estimator

        self.last_cmd = torch.tensor([1, 0.4, 1, -0.4, 2, 0.4, 2, -0.4], device=device).unsqueeze(0)
        self.n_dim_cmd_total = self.last_cmd.shape[1]

        # init buffers
        # Joint position command (deviation from default joint positions)
        self.actions = torch.zeros((1, 12), dtype=torch.float32, device=device)
        self.joint_pos = torch.zeros_like(self.actions)
        self.joint_vel = torch.zeros_like(self.actions)
        # feet contact filter
        self._contact_filt = torch.zeros((1, 4), dtype=torch.bool, device=device)
        self._last_contact = torch.zeros_like(self._contact_filt)

        """
        Simulation joint order:
        [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        ]
        Real joint order:
        [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        joint_map[i] = real_joint_order.index(sim_joint_order[i])
        e.g. joint_map[0] = real_joint_order.index("FL_hip_joint") = 3
        """
        self.joint_map = [
            3, 0, 9, 6,
            4, 1, 10, 7,
            5, 2, 11, 8
        ]
        default_joint_pos = [ # in simulation joint order
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5
        ]
        self.default_joint_pos = torch.tensor(default_joint_pos, dtype=torch.float32, device=device)

        # limits
        self.action_limits_low, self.action_limits_high = self._get_action_limits()
        self.torque_limits= torch.tensor([
            20.0, 20.0, 20.0, 20.0,
            20.0, 20.0, 20.0, 20.0,
            25.0, 25.0, 25.0, 25.0
        ], dtype=torch.float32, device=device)
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
            (-0.802, 0.802), (-0.802, 0.802), (-0.802, 0.802), (-0.802, 0.802), 
            (-1.05, 4.19), (-1.05, 4.19), (-1.05, 4.19), (-1.05, 4.19), 
            (-2.7, -0.916), (-2.7, -0.916), (-2.7, -0.916), (-2.7, -0.916)
        ]
        redundancy_factor = 0.95
        joint_pos_redundancy = [redundancy_factor*(joint_pos_range[i][1] - joint_pos_range[i][0]) for i in range(12)]
        action_limits_low = [(joint_pos_range[i][0] + joint_pos_redundancy[i] - self.default_joint_pos[i])/self.action_scale for i in range(12)]
        action_limits_high = [(joint_pos_range[i][1] - joint_pos_redundancy[i] - self.default_joint_pos[i])/self.action_scale for i in range(12)]
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
    
    def wait_untill_ros_working(self):
        rate = rospy.Rate(100)
        while not hasattr(self, "low_state_buffer"):
            rate.sleep()
        rospy.loginfo("UnitreeA1Real.low_state_buffer acquired, stop waiting.")

    def get_obs(self):
        if not hasattr(self, "last_step_time"):
            self.last_step_time = rospy.Time.now()
        dt = (rospy.Time.now() - self.last_step_time).to_sec()
        print(f"dt={dt:.6f}")  # TODO: debug
        self.last_step_time = rospy.Time.now()

        proprio_obs = self._get_proprio_obs()
        est_lin_vel = self.lin_vel_estimator(proprio_obs)
        # use est_lin_vel to compute command
        cmd = self.last_cmd.clone()
        for ep_i in range(0, self.n_dim_cmd_total, 2):
            ep = self.last_cmd[:, ep_i:ep_i+2]
            ep[:, 0] -= est_lin_vel[:, 0] * dt
            ep[:, 1] -= est_lin_vel[:, 1] * dt
            cmd[:, ep_i:ep_i+2] = ep
        self.last_cmd = cmd
        obs = torch.cat((cmd, proprio_obs, est_lin_vel), dim=-1)
        return obs
    
    def _get_proprio_obs(self):
        low_state = self.low_state_buffer
        # get contact_filt from contact sensor
        print("low_state.footForce:", low_state.footForce)
        contact = torch.tensor(low_state.footForce, device=self.device).unsqueeze(0) > 1.0
        print(low_state.footForce, contact)  # TODO: debug
        self._contact_filt = torch.logical_or(self._last_contact, contact)
        self._last_contact = contact

        base_ang_vel = torch.tensor(low_state.imu.gyroscope, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(
            torch.tensor(low_state.imu.quaternion).unsqueeze(0),
            self.gravity_vec,
        ).to(self.device)
        self.joint_pos = torch.tensor(
            [low_state.motorState[self.joint_map[i]].q for i in range(12)], 
            dtype=torch.float32, device=self.device).unsqueeze(0)
        self.joint_vel = torch.tensor(
            [low_state.motorState[self.joint_map[i]].dq for i in range(12)], 
            dtype=torch.float32, device=self.device).unsqueeze(0)

        proprio = torch.cat((self._contact_filt.float() * 2 - 1.0,
                             base_ang_vel,
                             projected_gravity,
                             self.joint_pos - self.default_joint_pos,
                             self.joint_vel,
                             self.actions), dim=-1)
        return proprio
    
    def send_action(self, actions):
        """ The function that send commands to the real robot.
        """
        self.actions = self.clip_action_before_scale(actions)
        robot_coordinates_action = self.clip_by_torque_limit(actions * self.action_scale) + self.default_joint_pos.unsqueeze(0)
        self.publish_legs_cmd(robot_coordinates_action)

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
        for sim_joint_idx in range(12):
            real_joint_idx = self.joint_map[sim_joint_idx]
            legs_cmd.cmd[real_joint_idx].mode = 10
            legs_cmd.cmd[real_joint_idx].q = robot_coordinates_action[0, sim_joint_idx] 
            legs_cmd.cmd[real_joint_idx].dq = 0.
            legs_cmd.cmd[real_joint_idx].tau = 0.
            legs_cmd.cmd[real_joint_idx].Kp = self.p_gains if kp is None else kp
            legs_cmd.cmd[real_joint_idx].Kd = self.d_gains if kd is None else kd
        self.legs_cmd_publisher.publish(legs_cmd)

    """ ROS callbacks and handlers that update the buffer """
    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg

