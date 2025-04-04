import os
import os.path as osp
import json
import numpy as np
import torch
from collections import OrderedDict
from functools import partial
from typing import Tuple

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import ros_numpy

from a1_real import UnitreeA1Real, resize2d
from policy_runner import PolicyRunner
import yaml


def load_cfg_from_yaml(cfg_yaml) -> dict:
    print(f"[INFO]: Parsing configuration from: {cfg_yaml}")
    with open(cfg_yaml, encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    return cfg


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


def main(args):
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("a1_turning", log_level=log_level)

    # prepare for cfg and models
    env_cfg_path="logs/params/env.yaml"
    agent_cfg_path="logs/params/agent.yaml"
    checkpoint_path = "logs/model_1999.pt"
    env_cfg = load_cfg_from_yaml(env_cfg_path)
    agent_cfg = load_cfg_from_yaml(agent_cfg_path)
    device = torch.device("cuda")
    runner = PolicyRunner(env_cfg=env_cfg,agent_cfg=agent_cfg, device=device)
    runner.load(checkpoint_path)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    policy = runner.get_inference_policy(device=device)
    lin_vel_estimator = runner.get_lin_vel_estimator()

    # instantiate handler
    handler = UnitreeA1Real(
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        lin_vel_estimator=lin_vel_estimator,
        device=device
    )
    handler.start_ros()
    handler.wait_untill_ros_working()
    duration = env_cfg["sim"]["dt"] * env_cfg["decimation"] # in seconds
    rate = rospy.Rate(1 / duration)
    with torch.no_grad():
        standup_procedure(handler, rate, 
                          angle_tolerance=0.2,
                          kp=40,
                          kd=0.5,
                          warmup_timesteps=50,
                          device=device,
                          )
        while not rospy.is_shutdown():
            obs = handler.get_obs()
            actions = policy(obs)
            handler.send_action(actions)
            rate.sleep()

            if handler.low_state_buffer.wirelessRemote.btn.components.L2 or handler.low_state_buffer.wirelessRemote.btn.components.R2:
                handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=2, kd=0.5)
                rospy.signal_shutdown("Controller send stop signal, exiting")

    
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