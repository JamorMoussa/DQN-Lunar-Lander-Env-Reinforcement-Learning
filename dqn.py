import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np 

from dataclasses import dataclass


@dataclass
class DQNConfigs:

    lr: float = 0.01

    input_dims: int 
    fc1_dims: int = 256
    fc2_dims: int = 256

    n_ations: int 

    @staticmethod
    def get_defaults():
        return DQNConfigs()


class DeepQNetwork(nn.Module):

    def __init__(
        self,
        configs: DQNConfigs
    ):
        super(DeepQNetwork, self).__init__()

        self.configs: DQNConfigs = configs

        self.dqn = nn.Sequential(
            nn.Linear(*self.configs.input_dims, self.configs.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.configs.fc1_dims, self.configs.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.configs.fc2_dims, self.configs.n_ations)
        )

        self.opt = optim.Adam(self.parameters(), lr=self.configs.lr)

        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    
    def forward(self, state) -> torch.Tensor:

        return self.dqn(state)


@dataclass
class AgentConfigs:

    gamma: float 
    eps: float 
    lr: float 
    input_dims: int 
    batch_size: int 
    n_actions: int 

    max_mem_size: int = 100_000
    eps_end: float = 0.01
    eps_dec = 5e-4

    @staticmethod
    def get_defaults():
        return AgentConfigs()
    

class Agent:

    def __init__(self, configs: AgentConfigs):

        self.configs: AgentConfigs = configs 

        self.action_space = list(range(self.configs.n_actions))

        self.mem_counter = 0 


        dqn_configs = DQNConfigs(
            lr=self.configs.lr,
            n_ations=self.configs.n_actions,
            input_dims=self.configs.input_dims,
        )

        self.q_eval_net = DeepQNetwork(
            configs= dqn_configs
        )

        self.state_memory = np.zeros((self.configs.max_mem_size, *self.configs.input_dims), dtype=np.float32)

        self.new_state_memory = np.copy(self.state_memory)

        self.action_memory = np.zeros(self.configs.max_mem_size, dtype=np.int32)

        self.reward_memory = np.zeros(self.configs.max_mem_size, stype=np.float32)

        self.terminal_memory = np.zeros(self.configs.max_mem_size, stype=bool)

    def store_transition(
        self, state, action, reward, state_, done
    ):
        
        index = self.mem_counter % self.configs.max_mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done 

        self.mem_counter += 1


    def choose_action(self, obsv):

        if np.random.rand() > self.configs.eps:
            state = torch.tensor([obsv], device=self.q_eval_net.device)
            action = self.q_eval_net(state).argmax()
        else:
            action = np.random.choice(self.action_space)

        return action 
    
    def learn(self):

        if self.mem_counter < self.configs.batch_size:
            return 
        
        self.q_eval_net.opt.zero_grad()

        # TODO: Continue The Learn Function



