import torch
import gym
from agent import DDPGAgent

env = gym.make('BipedalWalker-v3', render_mode="human")
num_steps = 1500


if __name__=="__main__":
    
    state_size = int(env.observation_space.shape[0])
    action_size = env.action_space.shape[0]
    action_lim = env.action_space.high[0]

    agent = DDPGAgent(env, state_size, action_size, action_lim, prate=0.0001, rate=0.001)

    agent.load_weights("saved_models")
    agent.is_training = False


    obs, info = env.reset()


    for step in range(num_steps):
        action = agent.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()