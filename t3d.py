"""
T3D (Twin Delayed Deep Deterministic Policy Gradient) Implementation
For continuous control of autonomous car navigation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    """Actor network that outputs continuous actions (steering, speed)"""
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = torch.FloatTensor(max_action)
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = self.max_action.to(state.device) * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    """Twin Critic networks for Q-value estimation"""
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # First Critic Network (Q1)
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        
        # Second Critic Network (Q2)
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        """Forward pass through both critics"""
        xu = torch.cat([state, action], 1)
        
        # Q1
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        
        # Q2
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        
        return x1, x2
    
    def Q1(self, state, action):
        """Only Q1 for actor gradient computation"""
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


class ReplayBuffer:
    """Experience Replay Buffer for storing transitions"""
    
    def __init__(self, max_size=int(1e6)):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    
    def add(self, transition):
        """Add transition (s, a, r, s', done) to buffer"""
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        
        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )


class T3D:
    """T3D Agent for continuous control"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = torch.FloatTensor(max_action).to(self.device)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100):
        """Train the agent for one step"""
        self.total_it += 1
        
        # Sample batch from replay buffer
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            # Select next action from target actor
            next_action = self.actor_target(next_state)
            
            # Add Gaussian noise for target policy smoothing
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q-values using twin critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks (Polyak averaging)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return critic_loss.item(), actor_loss.item()
        
        return critic_loss.item(), None
    
    def save(self, filename):
        """Save model weights"""
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
    
    def load(self, filename):
        """Load model weights"""
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
