from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
# from stable_baselines.common.bit_flipping_env import BitFlippingEnv
import pybullet_ur5
import gym
import pybullet
import time

model_class = DDPG  # works also with SAC, DDPG and TD3

# N_BITS=100
env = gym.make("UR5DynamicReachEnv-v2")
pybullet.connect(pybullet.DIRECT)
env.render("human")

env.seed(0)
# env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
# model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
#                                                 verbose=1)
# Train the model
# model.learn(5e5)

# model.save("./her_bit_env")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./her_bit_env', env=env)

obs = env.reset()
for _ in range(10000):
    time.sleep(0.02)
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
