from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def train(env, agent, batch_size=64, timesteps=1000000, render=False, record=0):
    done = True
    score = 0
    for step in tqdm(range(timesteps)):
        if done:
            print(score)
            state, info = env.reset()
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        score = info['score']
        record = max(record, info['score'])
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    env.close()
    agent.save()
    
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
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--save_model_dir', type=str, default='models/mario', help='directory to save model in')
    parser.add_argument('--model', type=str, default='medium', help='model to create if no model loaded')
    return parser.parse_args()

def main():
    args = parse_arguments()
    env = gym.make('SuperMarioBros-v2', apply_api_compatibility=True, render_mode="human" if args.render else None)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    epsilon_greedy = not (args.no_epsilon or args.test)
    if args.load_model:
        agent = DQNAgent(state_shape, action_size, model_path=args.load_model, epsilon_greedy=epsilon_greedy, learning_rate=args.learning_rate, save_dir=args.save_model_dir)
    else:
        agent = DQNAgent(state_shape, action_size, epsilon_greedy=epsilon_greedy, learning_rate=args.learning_rate, save_dir=args.save_model_dir, model=args.model)
    if args.test:
        test(env, agent, render=args.render)
    else:
        train(env, agent, batch_size=args.batch_size, timesteps=args.time, render=args.render, record=args.record)

if __name__ == '__main__':
    main()
