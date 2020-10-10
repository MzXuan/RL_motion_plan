import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)



# ------------bullet-------------

register(
    id='PyFetchReachEnv-v0',
    entry_point='pybullet_ur5.envs.fetch:FetchReachEnv',
    max_episode_steps=200,
    reward_threshold=20000.0,
)


register(
    id='UR5DynamicReachEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_dynamic_reach:UR5DynamicReachEnv',
    max_episode_steps=200,
    # max_episode_steps=400,
    reward_threshold=20000.0,
)

register(
    id='UR5DynamicTestEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_dynamic_test:UR5DynamicTestEnv',
    max_episode_steps=200,
    reward_threshold=20000.0,
)

register(
    id='UR5RealTestEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_human_real_env:UR5RealTestEnv',
    max_episode_steps=600,
    reward_threshold=20000.0,
)




def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs
