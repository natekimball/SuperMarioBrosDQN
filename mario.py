from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def train(env, agent, batch_size=32, timesteps=1000000, render=False, save_dir='models/mario', record=0):
    f = open('temp','a')
    done = True
    for step in tqdm(range(timesteps)):
        if step % 100 == 0:
            f.write((str(record) + '\n'))
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

    f.write((str(record) + '\n'))
    f.close()
    env.close()
    agent.save(save_dir)
    plot()

def write(record, filename='temp'):
    with open(filename, 'a') as f:
        f.write(str(record) + '\n')

def plot(high_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(high_scores)
    plt.xlabel('Every 500th time step')
    plt.ylabel('High Score')
    plt.savefig('out/mario_scores.png')
    plt.close()
    
def plot():
    with open('temp', 'r') as f:
        scores = map(int, f.readlines())
        plot(scores)
    
def test(env, agent, render=True):
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
    parser.add_argument('--time', type=int, default=1000000, help='Number of timesteps for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--load_model', type=str, help='Path to the model directory')
    parser.add_argument('--test', action='store_true', help='Enable test mode')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--save_dir', type=str, default='models/mario', help='Directory to save model to')
    parser.add_argument('--no_epsilon', action='store_false', help='no epsilon greedy')
    parser.add_argument('--record', type=int, default=0, help='current record')
    return parser.parse_args()

def main():
    args = parse_arguments()
    env = gym.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human" if args.render else None)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    if args.load_model:
        agent = DQNAgent(state_shape, action_size, model_path=args.load_model, initial_epsilon=0 if args.no_epsilon else 1.0)
    else:
        agent = DQNAgent(state_shape, action_size)
    if args.test:
        agent.test_mode = True
        test(env, agent, render=args.render)
    else:
        train(env, agent, batch_size=args.batch_size, timesteps=args.time, render=args.render, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
