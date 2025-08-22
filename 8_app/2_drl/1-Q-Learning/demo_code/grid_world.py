"""
Grid World Q-Learning Example
============================

A simple grid world environment demonstrating basic principles of Q-Learning algorithm.

Environment Description:
- 4x4 grid world
- Agent starts from top-left corner (0,0)
- Goal is at bottom-right corner (3,3)
- Agent can perform 4 actions: up, down, left, right
- Reach goal gets +10 reward, hit wall or out of bounds gets -1 reward, other steps get -0.1 reward
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List

class GridWorldEnvironment:
    def __init__(self, size: int = 4):
        """
        Initialize grid world environment
        
        Args:
            size: Grid size (size x size)
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.current_pos = self.start_pos
        
        # Define actions: 0=up, 1=down, 2=left, 3=right
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        self.action_names = ["up", "down", "left", "right"]
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action
        
        Args:
            action: Action number (0-3)
            
        Returns:
            next_state: Next state (position)
            reward: Reward obtained
            done: Whether finished
        """
        # Calculate new position
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check boundaries
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.current_pos = (new_x, new_y)
            
            # Reach goal
            if self.current_pos == self.goal_pos:
                return self.current_pos, 10.0, True
            else:
                return self.current_pos, -0.1, False  # Movement penalty
        else:
            # Hit wall, position unchanged
            return self.current_pos, -1.0, False
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state coordinates to index"""
        return state[0] * self.size + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert index to state coordinates"""
        return (index // self.size, index % self.size)

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Q-Learning Agent
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Record training process
        self.training_history = []
    
    def choose_action(self, state_index: int) -> int:
        """Use ε-greedy strategy to choose action"""
        if random.random() < self.epsilon:
            # Explore: randomly choose action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: choose action with maximum Q-value
            return np.argmax(self.q_table[state_index])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                       next_state: int, done: bool):
        """Update Q-value"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-Learning update formula
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

def train_q_learning(episodes: int = 1000, visualize_interval: int = 100):
    """Train Q-Learning agent"""
    
    # Create environment and agent
    env = GridWorldEnvironment(size=4)
    agent = QLearningAgent(
        n_states=env.size * env.size,
        n_actions=4,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9
    )
    
    episode_rewards = []
    episode_steps = []
    
    print("Starting Q-Learning agent training...")
    
    for episode in range(episodes):
        state = env.reset()
        state_index = env.state_to_index(state)
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Maximum step limit
            # Choose action
            action = agent.choose_action(state_index)
            
            # Execute action
            next_state, reward, done = env.step(action)
            next_state_index = env.state_to_index(next_state)
            
            # Update Q-value
            agent.update_q_value(state_index, action, reward, next_state_index, done)
            
            # Update state
            state = next_state
            state_index = next_state_index
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % visualize_interval == 0:
            avg_reward = np.mean(episode_rewards[-visualize_interval:])
            avg_steps = np.mean(episode_steps[-visualize_interval:])
            print(f"Episode {episode + 1}: avg_reward = {avg_reward:.2f}, "
                  f"avg_steps = {avg_steps:.1f}, ε = {agent.epsilon:.3f}")
    
    return agent, episode_rewards, episode_steps, env

def test_trained_agent(agent: QLearningAgent, env: GridWorldEnvironment):
    """Test the trained agent"""
    print("\nTesting trained agent:")
    
    state = env.reset()
    state_index = env.state_to_index(state)
    path = [state]
    
    print(f"Starting position: {state}")
    
    for step in range(20):  # Maximum test steps
        # Use greedy strategy (no exploration)
        action = np.argmax(agent.q_table[state_index])
        action_name = env.action_names[action]
        
        next_state, reward, done = env.step(action)
        next_state_index = env.state_to_index(next_state)
        
        path.append(next_state)
        print(f"Step {step + 1}: action={action_name}, new_pos={next_state}, reward={reward:.1f}")
        
        if done:
            print(f"Successfully reached goal! Used {step + 1} steps")
            break
        
        state = next_state
        state_index = next_state_index
    
    return path

def visualize_results(agent: QLearningAgent, env: GridWorldEnvironment, 
                     episode_rewards: List[float], episode_steps: List[int]):
    """Visualize training results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward changes during training
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Reward Changes During Training')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # 2. Step changes during training
    axes[0, 1].plot(episode_steps)
    axes[0, 1].set_title('Step Changes During Training')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # 3. Q-table heatmap
    q_max = np.max(agent.q_table, axis=1).reshape(env.size, env.size)
    im1 = axes[1, 0].imshow(q_max, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title('Maximum Q-values for Each State')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    
    # Add value labels
    for i in range(env.size):
        for j in range(env.size):
            axes[1, 0].text(j, i, f'{q_max[i, j]:.1f}', 
                          ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 4. Policy visualization (arrow plot)
    policy = np.argmax(agent.q_table, axis=1).reshape(env.size, env.size)
    arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}
    
    # Create policy matrix for display
    policy_display = np.zeros((env.size, env.size))
    axes[1, 1].imshow(policy_display, cmap='gray', alpha=0.3)
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.goal_pos:
                axes[1, 1].text(j, i, 'G', ha='center', va='center', fontsize=16, 
                               fontweight='bold', color='red',
                               bbox=dict(boxstyle="circle,pad=0.1", facecolor="yellow", alpha=0.8))
            else:
                arrow = arrows[policy[i, j]]
                axes[1, 1].text(j, i, arrow, ha='center', va='center', fontsize=20)
    
    axes[1, 1].set_title('Learned Policy')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    axes[1, 1].set_xticks(range(env.size))
    axes[1, 1].set_yticks(range(env.size))
    
    plt.tight_layout()
    plt.show()

def print_q_table(agent: QLearningAgent, env: GridWorldEnvironment):
    """Print Q-table"""
    print("\n=== Final Q-table ===")
    action_names = ["up", "down", "left", "right"]
    
    for i in range(env.size):
        for j in range(env.size):
            state_index = env.state_to_index((i, j))
            print(f"\nState ({i},{j}):")
            for action in range(4):
                q_value = agent.q_table[state_index, action]
                print(f"  {action_names[action]}: {q_value:.3f}")

if __name__ == "__main__":
    # Train agent
    agent, rewards, steps, env = train_q_learning(episodes=1000, visualize_interval=100)
    
    # Test agent
    path = test_trained_agent(agent, env)
    
    # Print Q-table
    print_q_table(agent, env)
    
    # Visualize results
    visualize_results(agent, env, rewards, steps)
    
    print("\n=== Q-Learning Grid World Example Complete ===")
    print("This example demonstrates:")
    print("1. Basic implementation of Q-Learning algorithm")
    print("2. Exploration vs exploitation balance with ε-greedy strategy")
    print("3. Q-value update process")
    print("4. Visualization of training process")
    print("5. Final learned policy")