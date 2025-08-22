"""
Maze Solving Q-Learning Example (Fixed Version)
==============================================

A more complex maze environment demonstrating Q-Learning in complex state spaces.

Features:
- Support for custom maze design
- Contains walls, traps, rewards and other elements
- Implements path backtracking and visualization
- Compares different parameter settings
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Optional

class MazeEnvironment:
    def __init__(self, maze_map: List[List[str]]):
        """
        Initialize maze environment
        
        Args:
            maze_map: Maze map represented by characters:
                'S' - Start position
                'G' - Goal position
                '#' - Wall
                'T' - Trap (negative reward)
                'R' - Reward point (positive reward)
                ' ' - Passable area
        """
        self.maze = np.array(maze_map)
        self.height, self.width = self.maze.shape
        
        # Find start and goal positions
        start_positions = np.where(self.maze == 'S')
        goal_positions = np.where(self.maze == 'G')
        
        if len(start_positions[0]) == 0 or len(goal_positions[0]) == 0:
            raise ValueError("Maze must contain start point (S) and goal point (G)")
        
        self.start_pos = (start_positions[0][0], start_positions[1][0])
        self.goal_pos = (goal_positions[0][0], goal_positions[1][0])
        self.current_pos = self.start_pos
        
        # Define actions: 0=up, 1=down, 2=left, 3=right
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        self.action_names = ["up", "down", "left", "right"]
        
        # Reward settings
        self.rewards = {
            'G': 100,    # goal
            'T': -50,    # trap
            'R': 10,     # reward point
            ' ': -1,     # normal move
            'S': -1      # start point
        }
    
    def reset(self) -> Tuple[int, int]:
        """Reset to start position"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action"""
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check boundaries and walls
        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] != '#'):
            
            self.current_pos = (new_x, new_y)
            cell_type = self.maze[new_x, new_y]
            reward = self.rewards.get(cell_type, -1)
            
            # Check if goal is reached
            done = (self.current_pos == self.goal_pos)
            
            return self.current_pos, reward, done
        else:
            # Hit wall or out of bounds, position unchanged, penalty given
            return self.current_pos, -10, False
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state to index"""
        return state[0] * self.width + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert index to state"""
        return (index // self.width, index % self.width)

class AdvancedQLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = np.zeros((n_states, n_actions))
        
        # Record training history
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        
        # Visit count (for analyzing exploration)
        self.visit_count = np.zeros(n_states)
    
    def choose_action(self, state_index: int) -> int:
        """Improved action selection strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state_index])
    
    def update_q_value(self, state: int, action: int, reward: float,
                       next_state: int, done: bool):
        """Update Q-value"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        
        # Update visit count
        self.visit_count[state] += 1
    
    def decay_epsilon(self, episode: int, total_episodes: int):
        """Improved exploration rate decay"""
        # Linear decay
        self.epsilon = max(0.01, 0.9 - 0.89 * episode / total_episodes)

def create_example_mazes():
    """Create example mazes"""
    
    # Simple maze
    simple_maze = [
        ['S', ' ', '#', 'G'],
        [' ', ' ', '#', ' '],
        [' ', ' ', ' ', ' '],
        ['#', '#', '#', ' ']
    ]
    
    # Complex maze
    complex_maze = [
        ['S', ' ', '#', ' ', 'R', ' ', '#'],
        [' ', ' ', '#', ' ', '#', ' ', ' '],
        [' ', '#', ' ', ' ', ' ', ' ', '#'],
        [' ', '#', 'T', '#', '#', ' ', ' '],
        [' ', ' ', ' ', ' ', '#', ' ', 'T'],
        ['#', '#', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', '#', ' ', ' ', 'G']
    ]
    
    # Maze with multiple traps
    trap_maze = [
        ['S', ' ', ' ', ' ', ' ', ' '],
        [' ', 'T', '#', 'T', ' ', ' '],
        [' ', ' ', '#', ' ', ' ', 'R'],
        ['T', ' ', ' ', ' ', '#', ' '],
        [' ', ' ', 'T', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', 'G']
    ]
    
    return {
        'simple': simple_maze,
        'complex': complex_maze,
        'trap': trap_maze
    }

def train_maze_solver(maze_name: str = 'complex', episodes: int = 2000):
    """Train maze solver"""
    
    mazes = create_example_mazes()
    maze_map = mazes[maze_name]
    
    # Create environment and agent
    env = MazeEnvironment(maze_map)
    agent = AdvancedQLearningAgent(
        n_states=env.height * env.width,
        n_actions=4,
        alpha=0.15,
        gamma=0.95,
        epsilon=0.9
    )
    
    print(f"Starting training for {maze_name} maze solver...")
    print(f"Maze size: {env.height}x{env.width}")
    print(f"Start position: {env.start_pos}")
    print(f"Goal position: {env.goal_pos}")
    
    successful_episodes = 0
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        state_index = env.state_to_index(state)
        total_reward = 0
        steps = 0
        max_steps = env.height * env.width * 2  # reasonable maximum steps
        
        while steps < max_steps:
            action = agent.choose_action(state_index)
            next_state, reward, done = env.step(action)
            next_state_index = env.state_to_index(next_state)
            
            agent.update_q_value(state_index, action, reward, next_state_index, done)
            
            state = next_state
            state_index = next_state_index
            total_reward += reward
            steps += 1
            
            if done:
                successful_episodes += 1
                break
        
        # Record training history
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        agent.epsilon_history.append(agent.epsilon)
        
        # Decay exploration rate
        agent.decay_epsilon(episode, episodes)
        
        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Print progress
        if (episode + 1) % 200 == 0:
            recent_rewards = agent.episode_rewards[-200:]
            recent_steps = agent.episode_steps[-200:]
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_steps)
            success_rate = successful_episodes / (episode + 1) * 100
            
            print(f"Episode {episode + 1:4d}: "
                  f"avg_reward = {avg_reward:6.1f}, "
                  f"avg_steps = {avg_steps:5.1f}, "
                  f"success_rate = {success_rate:5.1f}%, "
                  f"ε = {agent.epsilon:.3f}")
    
    print(f"\nTraining completed! Total success rate: {successful_episodes/episodes*100:.1f}%")
    print(f"Best reward: {best_reward:.1f}")
    
    return agent, env

def test_maze_solver(agent: AdvancedQLearningAgent, env: MazeEnvironment, 
                    show_path: bool = True) -> List[Tuple[int, int]]:
    """Test the trained agent"""
    print("\nTesting trained agent:")
    
    state = env.reset()
    state_index = env.state_to_index(state)
    path = [state]
    total_reward = 0
    
    print(f"Start position: {state}")
    
    for step in range(100):  # maximum test steps
        # Use greedy strategy (no exploration)
        action = np.argmax(agent.q_table[state_index])
        action_name = env.action_names[action]
        
        next_state, reward, done = env.step(action)
        next_state_index = env.state_to_index(next_state)
        
        path.append(next_state)
        total_reward += reward
        
        if show_path:
            print(f"Step {step + 1}: {action_name} -> {next_state}, reward = {reward:.1f}")
        
        if done:
            print(f"\nSuccessfully found path!")
            print(f"Total steps: {step + 1}")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Path length: {len(path)}")
            break
        
        # Check for loops
        if path.count(next_state) > 3:
            print(f"\nLoop detected, stopping at step {step + 1}")
            break
        
        state = next_state
        state_index = next_state_index
    else:
        print(f"\nFailed to find path within step limit")
    
    return path

def visualize_maze_training(agent: AdvancedQLearningAgent, env: MazeEnvironment,
                          path: List[Tuple[int, int]] = None):
    """Visualize maze training results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training reward curve
    axes[0, 0].plot(agent.episode_rewards)
    axes[0, 0].set_title('Training Process - Reward Change')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # 2. Training steps curve
    axes[0, 1].plot(agent.episode_steps)
    axes[0, 1].set_title('Training Process - Steps Change')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # 3. Exploration rate change
    axes[0, 2].plot(agent.epsilon_history)
    axes[0, 2].set_title('Exploration Rate (epsilon) Change')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon Value')
    axes[0, 2].grid(True)
    
    # 4. Maze map
    maze_visual = np.zeros((env.height, env.width))
    colors = {'S': 0.2, 'G': 0.8, '#': 0, ' ': 0.5, 'T': 0.1, 'R': 0.7}
    
    for i in range(env.height):
        for j in range(env.width):
            cell = env.maze[i, j]
            maze_visual[i, j] = colors.get(cell, 0.5)
    
    im = axes[1, 0].imshow(maze_visual, cmap='RdYlBu_r', alpha=0.8)
    
    # Add maze labels
    for i in range(env.height):
        for j in range(env.width):
            cell = env.maze[i, j]
            if cell in ['S', 'G', 'T', 'R']:
                axes[1, 0].text(j, i, cell, ha='center', va='center', 
                               fontsize=12, fontweight='bold')
    
    # If there's a path, draw it
    if path and len(path) > 1:
        path_y = [p[1] for p in path]
        path_x = [p[0] for p in path]
        axes[1, 0].plot(path_y, path_x, 'r-', linewidth=3, alpha=0.7)
        axes[1, 0].plot(path_y, path_x, 'ro', markersize=4)
    
    axes[1, 0].set_title('Maze Map and Optimal Path')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    
    # 5. State visit heatmap
    visit_map = agent.visit_count.reshape(env.height, env.width)
    im2 = axes[1, 1].imshow(visit_map, cmap='YlOrRd', alpha=0.8)
    axes[1, 1].set_title('State Visit Frequency Heatmap')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # 6. Policy visualization
    policy_map = np.zeros((env.height, env.width))
    arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}  # Use ASCII arrows instead of Unicode
    
    axes[1, 2].imshow(policy_map, cmap='gray', alpha=0.3)
    
    for i in range(env.height):
        for j in range(env.width):
            if env.maze[i, j] != '#':
                state_index = env.state_to_index((i, j))
                if np.max(agent.q_table[state_index]) > -np.inf:
                    best_action = np.argmax(agent.q_table[state_index])
                    arrow = arrows[best_action]
                    axes[1, 2].text(j, i, arrow, ha='center', va='center', 
                                   fontsize=14, fontweight='bold')
                if (i, j) == env.goal_pos:
                    # Use simple text instead of emoji
                    axes[1, 2].text(j, i, 'G', ha='center', va='center', 
                                   fontsize=16, fontweight='bold', color='red',
                                   bbox=dict(boxstyle="circle,pad=0.1", facecolor="yellow", alpha=0.8))
    
    axes[1, 2].set_title('Learned Policy')
    axes[1, 2].set_xlabel('Column')
    axes[1, 2].set_ylabel('Row')
    axes[1, 2].set_xticks(range(env.width))
    axes[1, 2].set_yticks(range(env.height))
    
    plt.tight_layout()
    plt.show()

def compare_parameters():
    """Compare different parameter settings"""
    print("\n=== Parameter Comparison Experiment ===")
    
    mazes = create_example_mazes()
    maze_map = mazes['complex']
    
    # Different parameter settings
    param_sets = [
        {'alpha': 0.1, 'gamma': 0.9, 'name': 'Standard'},
        {'alpha': 0.3, 'gamma': 0.9, 'name': 'High LR'},
        {'alpha': 0.1, 'gamma': 0.99, 'name': 'High Gamma'},
        {'alpha': 0.05, 'gamma': 0.8, 'name': 'Conservative'}
    ]
    
    results = {}
    
    for params in param_sets:
        print(f"\nTraining {params['name']}...")
        
        env = MazeEnvironment(maze_map)
        agent = AdvancedQLearningAgent(
            n_states=env.height * env.width,
            n_actions=4,
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=0.9
        )
        
        # Training (fewer episodes for comparison)
        for episode in range(1000):
            state = env.reset()
            state_index = env.state_to_index(state)
            total_reward = 0
            
            for step in range(50):
                action = agent.choose_action(state_index)
                next_state, reward, done = env.step(action)
                next_state_index = env.state_to_index(next_state)
                
                agent.update_q_value(state_index, action, reward, next_state_index, done)
                
                state = next_state
                state_index = next_state_index
                total_reward += reward
                
                if done:
                    break
            
            agent.episode_rewards.append(total_reward)
            agent.decay_epsilon(episode, 1000)
        
        # Test performance
        test_rewards = []
        for _ in range(10):
            env.reset()
            state_index = env.state_to_index(env.current_pos)
            total_reward = 0
            
            for _ in range(50):
                action = np.argmax(agent.q_table[state_index])
                next_state, reward, done = env.step(action)
                state_index = env.state_to_index(next_state)
                total_reward += reward
                if done:
                    break
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        results[params['name']] = {
            'avg_reward': avg_test_reward,
            'training_rewards': agent.episode_rewards[-100:],  # Last 100 episodes
            'params': params
        }
        
        print(f"{params['name']}: Average test reward = {avg_test_reward:.2f}")
    
    # Visualize comparison results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training curve comparison
    for name, result in results.items():
        rewards = result['training_rewards']
        ax1.plot(range(len(rewards)), rewards, label=name, alpha=0.7)
    
    ax1.set_title('Training Curve Comparison - Different Parameters')
    ax1.set_xlabel('Episode (Last 100 episodes)')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Average performance comparison
    names = list(results.keys())
    avg_rewards = [results[name]['avg_reward'] for name in names]
    
    bars = ax2.bar(names, avg_rewards)
    ax2.set_title('Average Test Performance - Different Parameters')
    ax2.set_ylabel('Average Reward')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, reward in zip(bars, avg_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def print_maze_info(env: MazeEnvironment):
    """Print maze information"""
    print("\n=== Maze Information ===")
    print("Maze layout:")
    for i, row in enumerate(env.maze):
        print(f"  {i}: {' '.join(row)}")
    
    print(f"\nSymbol meanings:")
    print(f"  S: Start position {env.start_pos}")
    print(f"  G: Goal position {env.goal_pos}")
    print(f"  #: Wall")
    print(f"  T: Trap (reward: {env.rewards['T']})")
    print(f"  R: Reward point (reward: {env.rewards['R']})")
    print(f"  Space: Normal passage (reward: {env.rewards[' ']})")

if __name__ == "__main__":
    try:
        # 1. Train complex maze
        print("=== Maze Solving Q-Learning Example ===")
        agent, env = train_maze_solver('complex', episodes=1000)  # Reduced episodes for quick testing
        
        # Print maze information
        print_maze_info(env)
        
        # 2. Test agent
        path = test_maze_solver(agent, env, show_path=False)  # Hide detailed path to reduce output
        
        # 3. Visualize results
        print("\nGenerating visualization charts...")
        visualize_maze_training(agent, env, path)
        
        # 4. Parameter comparison experiment
        print("\nStarting parameter comparison experiment...")
        comparison_results = compare_parameters()
        
        print("\n=== Summary ===")
        print("This maze solving example demonstrates:")
        print("1. Q-Learning application in complex environments")
        print("2. Different reward mechanism design (goal, trap, reward point)")
        print("3. Training process monitoring and visualization")
        print("4. State visit analysis and policy visualization")
        print("5. Comparison experiments with different parameter settings")
        print("6. Path planning and optimal policy verification")
        
    except Exception as e:
        print(f"Program execution error: {e}")
        print("Please check if dependencies are correctly installed:")
        print("pip install numpy matplotlib")