## bullet runner
python runner_bullet.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=3e5 --save_path=~/models/ur5_rnn_j_1205_03 --log_path=~/log/ur5_rnn_j_1205_03 --load_path=~/models/ur5_rnn_j_1201_01 
python runner_bullet.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=10e5 --save_path=~/models/ur5_mlp_c14_1126_l2 --log_path=~/log/ur5_mlp_c14_1126_l2
python runner_bullet.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=10e5 --save_path=~/models/ur5_mlp_c14_1118_l10 --log_path=~/log/ur5_mlp_c14_1118_l10


### bullet play
python runner_bullet.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/ur5_rnn_j_1205_03 --play

python runner_bullet.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/ur5_mlp_c14_1119_l2 --play


### bullet test with reference trajectory:
python runner_real.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/ur5_rnn_j_1205_03 --play

python runner_real.py --alg=her --env=UR5HumanEnv-v0 --env_type=robotics --num_timesteps=1 --load_path=~/models/ur5_mlp_c14_1126_l2 --play




## real robot test env: (BE CAREFUL!!!!)
python runner_real.py --alg=her --env=UR5DynamicReachEnv-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/ur5_rnn_c14_1116_1 --play

__
# her:
python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=2e5 --save_path=~/models/fetch_reach_her_5 --log_path=~/log/fetch_reach_her_5 --load_path=~/models/fetch_reach_her_5

python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=5000 --load_path=~/models/fetch_reach_her --play

python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/fetch_her --play

# test q value function:
python runner_test_q.py --alg=her --env=FetchDynamicTestReach-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/fetch_reach_her --play
+


#test reactive method:
1.  python test_reactive_method.py


# test data collection:
1.
python runner_test_load.py --alg=her --env=FetchDynamicCollectReach-v2 --env_type=robotics --num_timesteps=0 --load_path=~/models/fetch_reach_her_5 --play
2. train value function
goto spining up,
run ```python value_pi.py```

# test collision V function:
python runner_test_v.py --alg=her --env=FetchDynamicTestReach-v2 --env_type=robotics --num_timesteps=1 --load_path=~/models/fetch_reach_her --play --fpath=/home/xuan/Code/motion_style/spinningup/data/value_pi/value_pi_s0


# ddpg only:
train:
python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=2e5 --save_path=~/models/fetch_her --log_path=~/log/fetch_her 

test:
python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=0 --load_path=~/models/fetch_her --play

__



# how to train:

# ppo
python runner.py --alg=ppo2 --env=FetchMotionPlan-v0 --num_timesteps=1e7 --save_path=~/models/fetch_mp_fasp_app_ppo --log_path=~/log/fetch_mp_fasp_app_ppo --load_path=~/log/fetch_mp_fasp_app_ppo/checkpoints/00120 --num_env=8



#joint:
python runner.py --alg=ppo2 --env=FetchJointPlan-v1 --num_timesteps=1e7 --log_path=~/log/joint_mp_ppo --num_env=8


###test on reach env:
python runner.py --alg=ppo2 --env=FetchReach-v1 --num_timesteps=1e7 --save_path=~/models/reach_ppo --log_path=~/log/reach_ppo--num_env=8


### play:
python runner.py --alg=ppo2 --env=FetchMotionPlan-v0 --num_timesteps=1 --save_path=~/models/fetch_mp_fasp_app_ppo --load_path=~/log/fetch_mp_fasp_app_ppo/checkpoints/00120  --play



## from demonstration
python runner.py --alg=her --env=FetchMotionPlan-v0 --save_path=~/models/reach_ddpg --log_path=~/log/reach_ddpg --num_timesteps=2.5e6 --demo_file=/home/xuan/Code/motion_style/data_fetch_random_100.npz


# ddpg (not working)
python runner.py --alg=ddpg --env=FetchMotionPlan-v0 --num_timesteps=2e6 --save_path=~/models/fetch_mp_fasp_app_ddpg --log_path=~/log/fetch_mp_fasp_app_ddpg --num_env=8
