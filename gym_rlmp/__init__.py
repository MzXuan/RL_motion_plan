from gym.envs.registration import register


register(
    id='FetchMotionPlan-v0',
    entry_point='gym_rlmp.envs:FetchMotionPlanEnv',
    max_episode_steps=400
)


register(
    id='FetchJointPlan-v1',
    entry_point='gym_rlmp.envs:FetchPlanEnv',
    max_episode_steps=400
)