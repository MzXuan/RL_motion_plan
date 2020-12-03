import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)



# ------------bullet-------------


# fake robot and human
register(
    id='UR5DynamicReachPlannerEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_dynamic_reach_obs:UR5DynamicReachPlannerEnv',
    max_episode_steps=200,
    # max_episode_steps=400,
    reward_threshold=20000.0,
)

register(
    id='UR5DynamicReachEnv-v2',
    entry_point='pybullet_ur5.envs.ur5_dynamic_reach_obs:UR5DynamicReachObsEnv',
    max_episode_steps=400,
    # max_episode_steps=400,
    reward_threshold=20000.0,
)


# real robot and human
register(
    id='UR5HumanRealEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_human_env:UR5HumanRealEnv',
    max_episode_steps=8000,
    reward_threshold=20000.0,
)

register(
    id='UR5RealPlanTestEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_human_real_env:UR5RealPlanTestEnv',
    max_episode_steps=3000,
    reward_threshold=20000.0,
)


#real human, fake robot
register(
    id='UR5HumanEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_human_env:UR5HumanEnv',
    max_episode_steps=4000,
    reward_threshold=20000.0,
)

register(
    id='UR5HumanPlanEnv-v0',
    entry_point='pybullet_ur5.envs.ur5_human_env:UR5HumanPlanEnv',
    max_episode_steps=4000,
    reward_threshold=20000.0,
)






def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs
