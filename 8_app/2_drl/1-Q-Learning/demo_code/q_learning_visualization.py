"""
Q-Learning Learning Process Visualization
=========================================

This module provides tools for dynamic visualization of Q-Learning learning process, including:
1. Real-time Q-table update visualization
2. Agent learning trajectory animation
3. Convergence process analysis
4. Exploration-exploitation behavior visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Dict

class QLearningVisualizer:
    """Q-Learning process visualizer"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.training_data = {
            'q_tables': [],      # Q-table snapshots for each episode
            'paths': [],         # Paths for each episode
            'rewards': [],       # Rewards for each episode
            'exploration': [],   # Exploration behavior for each episode
            'convergence': []    # Q-value change magnitude
        }
        
    def record_episode(self, episode_num: int, path: List[Tuple[int, int]], 
                      total_reward: float, exploration_actions: int):
        """Record data for one episode"""
        # Save Q-table snapshot (every 10 episodes to save memory)
        if episode_num % 10 == 0:
            self.training_data['q_tables'].append(self.agent.q_table.copy())
        
        self.training_data['paths'].append(path)
        self.training_data['rewards'].append(total_reward)
        self.training_data['exploration'].append(exploration_actions)
        
        # Calculate Q-value change (convergence metric)
        if len(self.training_data['q_tables']) > 1:
            q_change = np.mean(np.abs(self.training_data['q_tables'][-1] - 
                                    self.training_data['q_tables'][-2]))
            self.training_data['convergence'].append(q_change)
    
    def create_interactive_dashboard(self):
        """Create interactive learning process dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Learning curve
        axes[0, 0].plot(self.training_data['rewards'], alpha=0.6)
        window = min(50, len(self.training_data['rewards']) // 4)
        if len(self.training_data['rewards']) > window:
            moving_avg = [np.mean(self.training_data['rewards'][max(0, i-window):i+1]) 
                         for i in range(len(self.training_data['rewards']))]
            axes[0, 0].plot(moving_avg, 'r-', linewidth=2)
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 2. Steps statistics  
        steps = [len(path) for path in self.training_data['paths']]
        axes[0, 1].plot(steps, alpha=0.6)
        if len(steps) > window:
            steps_avg = [np.mean(steps[max(0, i-window):i+1]) 
                        for i in range(len(steps))]
            axes[0, 1].plot(steps_avg, 'r-', linewidth=2)
        axes[0, 1].set_title('Steps per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # 3. Success rate statistics  
        if hasattr(self.env, 'goal_pos'):
            successes = [1 if hasattr(self.env, 'goal_pos') and path[-1] == self.env.goal_pos else 0 
                        for path in self.training_data['paths']]
            success_rate = [np.mean(successes[max(0, i-window):i+1]) 
                           for i in range(len(successes))]
            axes[1, 0].plot(success_rate, 'g-', linewidth=2)
            axes[1, 0].set_title(f'Success Rate (window={window})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True)
        else:
            # For simple grid world environment
            axes[1, 0].plot([])
            axes[1, 0].set_title('Success Rate (Not Available)')
            axes[1, 0].grid(True)
        
        # 4. Optimal path
        if len(self.training_data['paths']) > 0:
            # Find path with highest reward
            best_episode = np.argmax(self.training_data['rewards'])
            best_path = self.training_data['paths'][best_episode]
            
            # Draw environment (grid world or maze)
            if hasattr(self.env, 'maze'):
                # Maze environment
                maze_visual = np.zeros((self.env.height, self.env.width))
                colors = {'S': 0.2, 'G': 0.8, '#': 0, ' ': 0.5, 'T': 0.1, 'R': 0.7}
                
                for i in range(self.env.height):
                    for j in range(self.env.width):
                        cell = self.env.maze[i, j]
                        maze_visual[i, j] = colors.get(cell, 0.5)
                
                axes[1, 1].imshow(maze_visual, cmap='RdYlBu_r', alpha=0.8)
                
                # Draw optimal path
                if len(best_path) > 1:
                    path_x = [p[1] for p in best_path]
                    path_y = [p[0] for p in best_path]
                    axes[1, 1].plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8)
                    axes[1, 1].plot(path_x, path_y, 'ro', markersize=4)
            else:
                # Simple grid environment
                grid = np.zeros((self.env.size, self.env.size))
                axes[1, 1].imshow(grid, cmap='gray', alpha=0.3)
                
                if len(best_path) > 1:
                    path_x = [p[1] for p in best_path]
                    path_y = [p[0] for p in best_path]
                    axes[1, 1].plot(path_x, path_y, 'r-', linewidth=3)
                    axes[1, 1].plot(path_x, path_y, 'ro', markersize=6)
            
            axes[1, 1].set_title(f'Optimal Path (Episode {best_episode + 1})')
            axes[1, 1].set_xlabel('Column')
            axes[1, 1].set_ylabel('Row')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        self.print_training_statistics()
    
    def print_training_statistics(self):
        """Print training statistics"""
        print("\n=== Q-Learning Training Statistics ===")
        print(f"Total episodes: {len(self.training_data['rewards'])}")
        
        if len(self.training_data['rewards']) > 0:
            rewards = self.training_data['rewards']
            print(f"Average reward: {np.mean(rewards):.2f}")
            print(f"Maximum reward: {np.max(rewards):.2f}")
            print(f"Minimum reward: {np.min(rewards):.2f}")
            print(f"Final 100 episodes average reward: {np.mean(rewards[-100:]):.2f}")
        
        if len(self.training_data['paths']) > 0:
            steps = [len(path) for path in self.training_data['paths']]
            print(f"Average steps: {np.mean(steps):.1f}")
            print(f"Minimum steps: {np.min(steps)}")
            
            # Calculate success rate
            if hasattr(self.env, 'goal_pos'):
                successes = sum(1 for path in self.training_data['paths'] 
                              if path[-1] == self.env.goal_pos)
                print(f"Success rate: {successes/len(self.training_data['paths'])*100:.1f}%")
        
        if len(self.training_data['convergence']) > 0:
            final_convergence = self.training_data['convergence'][-10:]
            print(f"Final convergence level: {np.mean(final_convergence):.6f}")
        
        total_exploration = sum(self.training_data['exploration'])
        total_actions = sum(len(path) for path in self.training_data['paths'])
        if total_actions > 0:
            print(f"Overall exploration rate: {total_exploration/total_actions*100:.1f}%")


def train_with_visualization_grid(episodes=1000):
    """Training function with visualization for grid world"""
    
    # Import from grid world
    from grid_world import GridWorldEnvironment, QLearningAgent, train_q_learning
    
    # Create environment and agent
    env = GridWorldEnvironment(size=4)
    agent = QLearningAgent(
        n_states=env.size * env.size,
        n_actions=4,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9
    )
    
    # Create visualizer
    visualizer = QLearningVisualizer(env, agent)
    
    print("Starting grid world training with visualization...")
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        state_index = env.state_to_index(state)
        path = [state]
        total_reward = 0
        exploration_actions = 0
        steps = 0
        
        while steps < 100:  # Maximum step limit
            # Record if exploration action
            if np.random.random() < agent.epsilon:
                exploration_actions += 1
            
            action = agent.choose_action(state_index)
            next_state, reward, done = env.step(action)
            next_state_index = env.state_to_index(next_state)
            
            agent.update_q_value(state_index, action, reward, next_state_index, done)
            
            path.append(next_state)
            state = next_state
            state_index = next_state_index
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Record data
        visualizer.record_episode(episode, path, total_reward, exploration_actions)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode + 1}: avg_reward = {avg_reward:.2f}, "
                  f"avg_steps = {avg_steps:.1f}, ε = {agent.epsilon:.3f}")
    
    print("Training completed!")
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard()
    
    return visualizer


def train_with_visualization_maze(maze_name='complex', episodes=1000):
    """Training function with visualization for maze"""
    
    try:
        # Import from maze solver
        from maze_solver_example_fixed import MazeEnvironment, AdvancedQLearningAgent, create_example_mazes
        
        # Create environment
        mazes = create_example_mazes()
        maze_map = mazes[maze_name]
        env = MazeEnvironment(maze_map)
        
        # Create agent
        agent = AdvancedQLearningAgent(
            n_states=env.height * env.width,
            n_actions=4,
            alpha=0.15,
            gamma=0.95,
            epsilon=0.9
        )
        
        # Create visualizer
        visualizer = QLearningVisualizer(env, agent)
        
        print(f"Starting {maze_name} maze training with visualization...")
        
        for episode in range(episodes):
            state = env.reset()
            state_index = env.state_to_index(state)
            path = [state]
            total_reward = 0
            exploration_actions = 0
            
            for step in range(100):  # Maximum steps
                # Record if exploration action
                if np.random.random() < agent.epsilon:
                    exploration_actions += 1
                
                action = agent.choose_action(state_index)
                next_state, reward, done = env.step(action)
                next_state_index = env.state_to_index(next_state)
                
                agent.update_q_value(state_index, action, reward, next_state_index, done)
                
                path.append(next_state)
                state = next_state
                state_index = next_state_index
                total_reward += reward
                
                if done:
                    break
            
            # Record data
            visualizer.record_episode(episode, path, total_reward, exploration_actions)
            
            # Decay exploration rate
            agent.decay_epsilon(episode, episodes)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                recent_rewards = visualizer.training_data['rewards'][-100:]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1}: avg_reward = {avg_reward:.2f}")
        
        print("Training completed!")
        
        # Create interactive dashboard
        visualizer.create_interactive_dashboard()
        
        return visualizer
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Falling back to grid world visualization...")
        return train_with_visualization_grid(episodes)


if __name__ == "__main__":
    # Run training with visualization
    print("=== Q-Learning Visualization Learning Process ===")
    
    print("\nChoose environment:")
    print("1. Grid World (4x4)")
    print("2. Maze Environment")
    
    choice = input("Enter choice (1 or 2, default 1): ").strip()
    
    if choice == "2":
        # Train and visualize maze
        visualizer = train_with_visualization_maze('complex', episodes=500)
    else:
        # Train and visualize grid world
        visualizer = train_with_visualization_grid(episodes=500)
    
    print("\nVisualization module features:")
    print("1. Real-time Q-table change recording")
    print("2. Path learning process animation")
    print("3. Exploration-exploitation behavior analysis")
    print("4. Convergence monitoring")
    print("5. Interactive learning dashboard")
    print("6. Detailed statistical reports")