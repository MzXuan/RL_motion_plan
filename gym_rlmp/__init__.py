from gym.envs.registration import register


register(
    id='FetchMotionPlan-v0',
    entry_point='gym_rlmp.envs:FetchMotionPlanEnv',
    max_episode_steps=400
)


