import os
from datetime import datetime
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

from a1_handler import UnitreeRosHandler
from our_policy_runner import PolicyRunner
import yaml


def load_cfg_from_yaml(cfg_yaml) -> dict:
    print(f"[INFO]: Parsing configuration from: {cfg_yaml}")
    with open(cfg_yaml, encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    return cfg


class StandOnlyModel(torch.nn.Module):
    def __init__(self, action_scale, joint_pos_scale, with_contact=False, tolerance=0.2, delta=0.1):
        rospy.loginfo("Initializing stand only model")
        super().__init__()
        if isinstance(action_scale, (tuple, list)):
            self.register_buffer("action_scale", torch.tensor(action_scale))
        else:
            self.action_scale = action_scale
        if isinstance(joint_pos_scale, (tuple, list)):
            self.register_buffer("joint_pos_scale", torch.tensor(joint_pos_scale))
        else:
            self.joint_pos_scale = joint_pos_scale
        self.tolerance = tolerance
        self.delta = delta
        self.with_contact = with_contact

    def forward(self, obs):
        # Determine the index for joint positions based on observation structure
        # Command is 4 values, then ang_vel (3), proj_g (3), and then joint positions (12)
        joint_pos_start = 10
        if self.with_contact:
            joint_pos_start = 14  # Offset by 4 more if contact info is included
        
        joint_positions = obs[..., joint_pos_start:joint_pos_start+12] / self.joint_pos_scale
        diff_large_mask = torch.abs(joint_positions) > self.tolerance
        target_positions = torch.zeros_like(joint_positions)
        target_positions[diff_large_mask] = joint_positions[diff_large_mask] - self.delta * torch.sign(joint_positions[diff_large_mask])
        return torch.clip(
            target_positions / self.action_scale,
            -1.0, 1.0,
        )
    
    def reset(self, *args, **kwargs):
        pass


def main(args):
    save_obs = True
    save_tau = True
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("a1_turning", log_level=log_level)
    # prepare for cfg and models
    env_cfg_path="logs/params/env.yaml"
    agent_cfg_path="logs/params/agent.yaml"
    checkpoint_path = "logs/model_5999.pt"
    env_cfg = load_cfg_from_yaml(env_cfg_path)
    agent_cfg = load_cfg_from_yaml(agent_cfg_path)
    device = torch.device("cpu")
    runner = PolicyRunner(env_cfg=env_cfg, agent_cfg=agent_cfg, device=device)  # for initializing and loading models
    runner.load(checkpoint_path)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    policy = runner.get_inference_policy(device=device)
    if agent_cfg["lin_vel_estimator"]["train_lin_vel_estimator"]:
        lin_vel_estimator = runner.get_lin_vel_estimator()
    else:
        lin_vel_estimator = None

    # Create a logs directory if it doesn't exist
    if save_obs:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        # Create a log file with timestamp
        obs_filename = os.path.join(log_dir, f'obs-{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
    if save_tau:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        # Create a log file with timestamp
        tau_filename = os.path.join(log_dir, f'tau-{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')

    # Initialize the stand-only model (copied from parkour deployment code)
    stand_only_model = StandOnlyModel(
        action_scale=env_cfg['action_scale'],
        joint_pos_scale=env_cfg['observation']['joint_pos']['scale'],
        with_contact=env_cfg['with_contact'],
        tolerance=0.2,
        delta=0.1
    )
    stand_only_policy = torch.jit.script(stand_only_model)

    # instantiate handler
    handler = UnitreeRosHandler(
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        lin_vel_estimator=lin_vel_estimator,
        device=device
    )

    # Warmup the model with a dummy inference pass
    print("[INFO]: Running comprehensive model warmup...")
    with torch.no_grad():
        # Get exact input dimensions
        dummy_proprio = torch.zeros((1, handler.n_dim_proprio), device=device)
        dummy_cmd = torch.zeros((1, handler.n_dim_cmd_total), device=device)
        
        # Multiple passes to ensure JIT compilation is complete
        for _ in range(10):  # Run multiple iterations to ensure full warmup
            if lin_vel_estimator:
                dummy_vel_est = lin_vel_estimator(dummy_proprio)
                dummy_obs = torch.cat((dummy_cmd, dummy_proprio, dummy_vel_est), dim=-1)
            else:
                dummy_obs = torch.cat((dummy_cmd, dummy_proprio), dim=-1)
            _ = policy(dummy_obs)
    print("[INFO]: Model warmup complete")
    # Add after the warmup loop
    torch.cuda.synchronize()  # Force synchronization of CUDA operations
    print("[INFO]: CUDA synchronized")
    
    using_stand_only_policy = True # switch between forward policy and stand only policy
    handler.start_ros()
    handler.wait_untill_ros_working()
    duration = env_cfg["sim"]["dt"] * env_cfg["decimation"] # in seconds
    print(f"[INFO]: duration: {duration}")
    rate = rospy.Rate(1 / duration)
    with torch.no_grad():
        rospy.sleep(1.0)
        handler.calibrate_foot_force(rate, angle_tolerance=0.2, kp=40, kd=0.5, warmup_timesteps=50, device=device)
        handler.publish_stop_stand_up(True)
        print("send stop stand up")
        timestep = 0
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            obs = handler.get_obs()
            if handler.low_state_buffer.wirelessRemote.ry > 0.1:    # move forward
                if using_stand_only_policy:
                    rospy.loginfo_throttle(0.1, "switch to skill policy")
                    using_stand_only_policy = False
                    runner.actor_critic.reset()
            else:
                if not using_stand_only_policy:
                    rospy.loginfo_throttle(0.1, "switch to walk policy")
                    using_stand_only_policy = True
                    stand_only_model.reset()
            if not using_stand_only_policy:
                actions = policy(obs)
            else:
                actions = stand_only_policy(obs)
            actions_clipped_action_limits, actions_scaled, actions_scaled_clipped_torque_limits, motor_state, legs_cmd = handler.send_action(actions)

            # save observations
            if save_obs:
                with open(obs_filename, 'a') as f:
                    # Define indices based on whether contact info is included
                    contact_offset = 4 if env_cfg['with_contact'] else 0
                    
                    cmd_idx = (0, 4)
                    ang_vel_idx = (4 + contact_offset, 7 + contact_offset)
                    proj_g_idx = (7 + contact_offset, 10 + contact_offset)
                    joint_pos_idx = (10 + contact_offset, 22 + contact_offset)
                    joint_vel_idx = (22 + contact_offset, 34 + contact_offset)
                    last_actions_idx = (34 + contact_offset, 46 + contact_offset)
                    est_lin_vel_idx = 46 + contact_offset
                    
                    # Create unified data dictionary
                    data = {
                        'timestep': timestep,
                        'time': f"{(rospy.Time.now() - start_time).to_sec():.3f}",
                        'cmd': [f"{v:.3f}" for v in obs[0, cmd_idx[0]:cmd_idx[1]].cpu().numpy().tolist()],
                        'ang_vel': [f"{v:.3f}" for v in obs[0, ang_vel_idx[0]:ang_vel_idx[1]].cpu().numpy().tolist()],
                        'proj_g': [f"{v:.3f}" for v in obs[0, proj_g_idx[0]:proj_g_idx[1]].cpu().numpy().tolist()],
                        'joint_pos': [f"{v:.3f}" for v in obs[0, joint_pos_idx[0]:joint_pos_idx[1]].cpu().numpy().tolist()],
                        'joint_vel': [f"{v:.3f}" for v in obs[0, joint_vel_idx[0]:joint_vel_idx[1]].cpu().numpy().tolist()],
                        'last_actions': [f"{v:.3f}" for v in obs[0, last_actions_idx[0]:last_actions_idx[1]].cpu().numpy().tolist()],
                        'raw_actions': [f"{v:.3f}" for v in actions[0, :].cpu().numpy().tolist()],
                        'actions_clipped_action_limits': [f"{v:.3f}" for v in actions_clipped_action_limits[0, :].cpu().numpy().tolist()],
                        'actions_scaled': [f"{v:.3f}" for v in actions_scaled[0, :].cpu().numpy().tolist()],
                        'actions_scaled_clipped_torque_limits': [f"{v:.3f}" for v in actions_scaled_clipped_torque_limits[0, :].cpu().numpy().tolist()],
                    }
                    
                    # Add contact_filt data only if with_contact is True
                    if env_cfg['with_contact']:
                        contact_filt_idx = (4, 8)
                        data['contact_filt'] = [f"{v:.0f}" for v in obs[0, contact_filt_idx[0]:contact_filt_idx[1]].cpu().numpy().tolist()]
                    
                    # Add estimated linear velocity if estimator exists
                    if lin_vel_estimator:
                        data['est_lin_vel'] = [f"{v:.3f}" for v in obs[0, est_lin_vel_idx:est_lin_vel_idx+3].cpu().numpy().tolist()]
                    
                    f.write(json.dumps(data) + '\n')
            if save_tau:
                data = {
                    'timestep': timestep,
                    'time': f"{(rospy.Time.now() - start_time).to_sec():.3f}",
                    'joints': {}
                }
                # Collect data for all 12 joints
                for sim_joint_idx in range(12):
                    real_joint_idx = handler.joint_map[sim_joint_idx]
                    cmd = legs_cmd[real_joint_idx].q
                    q = motor_state[real_joint_idx].q
                    dq = motor_state[real_joint_idx].dq
                    Kp = legs_cmd[real_joint_idx].Kp
                    Kd = legs_cmd[real_joint_idx].Kd
                    computed_torque = (cmd-q)*Kp-dq*Kd
                    tauEst = motor_state[real_joint_idx].tauEst
                    
                    # Store joint data using the format from the print statement
                    data['joints'][f'joint_{real_joint_idx}'] = {
                        'cmd': f"{cmd:.3f}",
                        'q': f"{q:.3f}",
                        'dq': f"{dq:.3f}",
                        'computed_torque': f"{computed_torque:.3f}",
                        'tauEst': f"{tauEst:.3f}"
                    }
                
                # Write data to file
                with open(tau_filename, 'a') as f:
                    f.write(json.dumps(data) + '\n')

            rate.sleep()
            if handler.low_state_buffer.wirelessRemote.btn.components.L2 or handler.low_state_buffer.wirelessRemote.btn.components.R2:
                handler.publish_legs_cmd(handler.default_joint_pos.unsqueeze(0), kp=2, kd=0.5)
                rospy.signal_shutdown("Controller send stop signal, exiting")
            timestep += 1
            # if timestep > 400:
            #     break

    
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