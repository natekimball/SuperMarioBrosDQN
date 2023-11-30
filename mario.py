from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
    
env = gym.make('SuperMarioBros-v2', apply_api_compatibility=True)#, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_shape = env.observation_space.shape
action_size = env.action_space.n

def train(agent, batch_size=32, timesteps=1000000, render=False, save_dir='models/mario'):
    high_scores = []
    record = 0
    done = True
    for step in tqdm(range(timesteps)):
        if step % 100 == 0:
            high_scores.append(record)
            if step%500 == 0:
                plot(high_scores)
        if done:
            state, info = env.reset()
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        record = max(record, info['score'])
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
    env.close()
    agent.save(save_dir)
    plot(high_scores)

def plot(high_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(high_scores)
    plt.xlabel('Every 100th time step')
    plt.ylabel('High Score')
    plt.savefig('out/mario_scores.png')
    plt.close()
    
def test(agent, render=True):
    done = True
    high_score = 0
    for step in tqdm(range(1000)):
        if done:
            state, info = env.reset()
        action = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        done = terminated or truncated
        high_score = max(high_score, info['score'])
    env.close()
    print("High score: ", high_score)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Super Mario Bros DQN')
    parser.add_argument('--time', type=int, help='Number of timesteps for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--load_model', type=str, help='Path to the model directory')
    parser.add_argument('--test', action='store_true', help='Enable test mode')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--save_dir', type=str, default='models/mario', help='Directory to save model to')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.load_model:
        agent = DQNAgent(state_shape, action_size, load_model=args.load_model)
    else:
        agent = DQNAgent(state_shape, action_size)
    if args.test:
        agent.test_mode = True
        test(agent, render=args.render)
    else:
        train(agent, batch_size=args.batch_size, timesteps=args.time, render=args.render, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
        
# # EPISODES = 1
# # scores = []
# # moving_avg_scores = []
# # for e in range(EPISODES):
# #     state, info = env.reset()
# #     done = False
# #     score = 0
# #     for time in range(5000):
# #         action = agent.act(state)
# #         next_state, reward, terminated, truncated, info = env.step(action)
# #         done = terminated or truncated
# #         agent.remember(state, action, reward, next_state, done)
# #         state = next_state
# #         score = info['score']
# #         if done:
# #             break
# #         if len(agent.memory) > batch_size:
# #             agent.replay(batch_size)
# #     print(f"episode: {e}/{EPISODES}, score: {score}, ε: {agent.epsilon:.2}")
# #     scores.append(score)
# #     moving_avg_scores.append(np.mean(scores[-100:]))

# # scores = []
# # moving_avg_scores = []
# # e = 1
# # done = True
# # for time in range(1000000):
# #     if done:
# #         state, info = env.reset()
# #     action = agent.act(state)
# #     next_state, reward, terminated, truncated, info = env.step(action)
# #     done = terminated or truncated
# #     agent.remember(state, action, reward, next_state, done)
# #     state = next_state
# #     if len(agent.memory) > batch_size:
# #         agent.replay(batch_size)
# #     if done:
# #         score = info['score']
# #         print(f"episode: {e}, score: {score}, ε: {agent.epsilon:.2}, time-step: {time}")
# #         scores.append(score)
# #         moving_avg_scores.append(np.mean(scores[-100:]))
# #         break
# #     e += 1
