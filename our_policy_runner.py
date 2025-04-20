import torch
import numpy as np
import torch.nn as nn

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = resolve_nn_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)


class PolicyRunner:
    def __init__(self, env_cfg, agent_cfg, device=torch.device("cpu")):

        self.agent_cfg = agent_cfg
        self.policy_cfg = self.agent_cfg["policy"]
        self.env_cfg = env_cfg
        self.device = device
        self.estimator_cfg = agent_cfg["lin_vel_estimator"]

        #Parameter for our policy
        self.n_dim_cmd = 4  # =env._command.n_dim_cmd
        self.n_dim_proprio = env_cfg["n_dim_proprio"]
        self.num_obs = self.n_dim_cmd * env_cfg["command"]["num_goals_in_cmd"] + self.n_dim_proprio
        if self.estimator_cfg["train_lin_vel_estimator"]:
            self.num_obs += agent_cfg["lin_vel_estimator"]["n_dim_output"]
        num_critic_obs = self.num_obs

        # Initialize networks
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        self.actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            self.num_obs, num_critic_obs, self.env_cfg["action_space"], **self.policy_cfg
        ).to(self.device)

        # linear velocity estimator
        if self.estimator_cfg["train_lin_vel_estimator"]:
            self.estimator = Estimator(input_dim=self.estimator_cfg["n_dim_input"], output_dim=self.estimator_cfg["n_dim_output"], hidden_dims=self.estimator_cfg["mlp_hidden_dims"]).to(self.device)

        # Initialize algorithm no need to initialize the PPO
        self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        policy = self.actor_critic.act_inference
        return policy
    
    def get_lin_vel_estimator(self, device=None):
        self.estimator.eval()
        if device is not None:
            self.estimator.to(device)
        return self.estimator.inference
    
    def load(self, path):
        loaded_dict = torch.load(path)
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.estimator_cfg["train_lin_vel_estimator"]:
            self.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        return loaded_dict["infos"]
    
    def eval_mode(self):
        self.actor_critic.eval()
