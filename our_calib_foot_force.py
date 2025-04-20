import os
import json
import numpy as np
import torch
import yaml

import rospy
from unitree_legged_msgs.msg import LegsCmd, LowState


def load_cfg_from_yaml(cfg_yaml) -> dict:
    print(f"[INFO]: Parsing configuration from: {cfg_yaml}")
    with open(cfg_yaml, encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    return cfg


def wait_for_R1(handler, ros_rate, kp, kd):
    while not rospy.is_shutdown():
        if handler.low_state_buffer.wirelessRemote.btn.components.R1:
            break
        if handler.low_state_buffer.wirelessRemote.btn.components.L2 or handler.low_state_buffer.wirelessRemote.btn.components.R2:
            handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=0, kd=0.5)
            rospy.signal_shutdown("Controller send stop signal, exiting")
            exit(0)
        handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=kp, kd=kd)
        ros_rate.sleep()


def calibrate_foot_force(handler, ros_rate, angle_tolerance, kp, kd, warmup_timesteps, device=torch.device("cpu")):
    """
    Args:
        warmup_timesteps: the number of timesteps to linearly increase the target position
    """
    rospy.loginfo("Robot standing up, please wait...")

    target_pos = torch.zeros((1, 12), device=device, dtype=torch.float32)
    standup_timestep_i = 0
    while not rospy.is_shutdown():
        joint_pos = [handler.low_state_buffer.motorState[handler.joint_map[i]].q for i in range(12)]
        diff = [handler.default_joint_pos[i].item() - joint_pos[i] for i in range(12)]
        if all([abs(i) < angle_tolerance for i in diff]):
            break
        direction = [1 if i > 0 else -1 for i in diff]
        if standup_timestep_i < warmup_timesteps:
            direction = [standup_timestep_i / warmup_timesteps * i for i in direction]

        print(f"{standup_timestep_i}: max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
        for i in range(12):
            target_pos[0, i] = joint_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else handler.default_joint_pos[i]
        handler.publish_legs_cmd(target_pos, kp=kp, kd=kd)
        ros_rate.sleep()
        standup_timestep_i += 1
    rospy.loginfo("Robot stood up! Recording standup readings...")
    standup_readings = []
    for _ in range(100):  # Collect multiple samples
        foot_force = handler.low_state_buffer.footForce
        print(f"footForce: FR:{foot_force[0]}, FL:{foot_force[1]}, RR:{foot_force[2]}, RL:{foot_force[3]}")
        standup_readings.append(foot_force)
        ros_rate.sleep()
    standup_baseline = torch.mean(torch.tensor(standup_readings, type=torch.float), dim=0)
    rospy.loginfo("Recorded! Now lift the robot off the ground and press R1 to record zero readings...")
    wait_for_R1(handler, ros_rate, kp, kd)
    rospy.loginfo("Robot lifted! Recording zero readings...")
    zero_readings = []
    for _ in range(100):  # Collect multiple samples
        foot_force = handler.low_state_buffer.footForce
        print(f"footForce: FR:{foot_force[0]}, FL:{foot_force[1]}, RR:{foot_force[2]}, RL:{foot_force[3]}")
        zero_readings.append(foot_force)
        ros_rate.sleep()
    zero_baseline = torch.mean(torch.tensor(zero_readings), dim=0)
    rospy.loginfo("Recorded! Now put the robot back on the ground and press R1 to continue...")
    wait_for_R1(handler, ros_rate, kp, kd)
    return standup_baseline, zero_baseline


def print_calibrated_force(handler, force_offsets, force_scaling):
    raw_forces = torch.tensor(handler.low_state_buffer.footForce)
    calibrated_forces = (raw_forces - force_offsets) / force_scaling
    contact = calibrated_forces > 0.1
    print(f"Calibrated forces: FR:{calibrated_forces[0]}, FL:{calibrated_forces[1]}, RR:{calibrated_forces[2]}, RL:{calibrated_forces[3]}")
    print(f"Contact: FR:{contact[0]}, FL:{contact[1]}, RR:{contact[2]}, RL:{contact[3]}")


def test_calibration(handler, ros_rate, standup_baseline, zero_baseline, kp, kd):
    robot_weight = 12.   # kg
    force_offsets = zero_baseline
    force_scaling = (standup_baseline - zero_baseline) / robot_weight
    rospy.loginfo("Testing calibration...")
    while not rospy.is_shutdown():
        print_calibrated_force(handler, force_offsets, force_scaling)
        handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=kp, kd=kd)
        ros_rate.sleep()
        press_A = handler.low_state_buffer.wirelessRemote.btn.components.A
        press_B = handler.low_state_buffer.wirelessRemote.btn.components.B
        press_X = handler.low_state_buffer.wirelessRemote.btn.components.X
        press_Y = handler.low_state_buffer.wirelessRemote.btn.components.Y
        if press_A or press_B or press_X or press_Y:
            CALF_TARGET = -1.7
            tolerance = 0.02
            if press_A:
                calf_name = "FR"
                calf_idx = 9    # in simulation joint order
                button_name = "A"
            elif press_B:
                calf_name = "FL"
                calf_idx = 8    # in simulation joint order
                button_name = "B"
            elif press_X:
                calf_name = "RR"
                calf_idx = 11    # in simulation joint order
                button_name = "X"
            elif press_Y:
                calf_name = "RL"
                calf_idx = 10    # in simulation joint order
                button_name = "Y"
            rospy.loginfo(f"Lifting {calf_name} calf...")
            while not rospy.is_shutdown():
                joint_pos = [handler.low_state_buffer.motorState[handler.joint_map[i]].q for i in range(12)]
                if abs(joint_pos[calf_idx] - CALF_TARGET) > tolerance:
                    applied_calf_target_pos = joint_pos[calf_idx] - 0.005
                else:
                    applied_calf_target_pos = CALF_TARGET
                    rospy.loginfo(f"{calf_name} calf target position reached! Press {button_name} to move back to default position.")
                    if handler.low_state_buffer.wirelessRemote.btn.components.A:
                        break
                target_joint_pos = handler.default_joint_pos.clone()
                target_joint_pos[calf_idx] = applied_calf_target_pos
                handler.publish_legs_cmd(target_joint_pos.unsqueeze(0), kp=kp, kd=kd)
                ros_rate.sleep()
                print_calibrated_force(handler, force_offsets, force_scaling)
            rospy.loginfo("Moving back to default position...")
            while not rospy.is_shutdown():
                joint_pos = [handler.low_state_buffer.motorState[handler.joint_map[i]].q for i in range(12)]
                diff = [handler.default_joint_pos[i].item() - joint_pos[i] for i in range(12)]
                direction = [1 if i > 0 else -1 for i in diff]
                if all([abs(i) < tolerance for i in diff]):
                    break
                print("max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
                for i in range(12):
                    target_joint_pos[i] = joint_pos[i] + direction[i] * 0.005 if abs(diff[i]) > tolerance else handler.default_joint_pos[i]
                handler.publish_legs_cmd(target_joint_pos.unsqueeze(0), kp=kp, kd=kd)
                ros_rate.sleep()
                print_calibrated_force(handler, force_offsets, force_scaling)
            rospy.loginfo("Back to default position!")


class UnitreeA1RealCalib:
    """ This is the handler that works for ROS 1 on unitree. """
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

        # from cfg
        self.d_gains = 0.5
        self.p_gains = 25.

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


def main(args):
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("a1_turning", log_level=log_level)

    # instantiate handler
    handler = UnitreeA1RealCalib()
    handler.start_ros()
    handler.wait_untill_ros_working()
    duration = 0.02 # in seconds
    rate = rospy.Rate(1 / duration)
    with torch.no_grad():
        standup_baseline, zero_baseline = calibrate_foot_force(handler, rate, angle_tolerance=0.1, kp=25, kd=0.5, warmup_timesteps=50)
        rospy.loginfo("Calibration finished! Now testing calibration...")
        test_calibration(handler, rate, standup_baseline, zero_baseline, kp=25, kd=0.5)

    
if __name__ == "__main__":
    """ The script to run the A1 script in ROS.
    It's designed as a main function and not designed to be a scalable code.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
        type= str,
        default= "/a112138",                    
    )
    parser.add_argument("--debug",
        action= "store_true",
    )
    args = parser.parse_args()
    main(args)