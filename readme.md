# her:
python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=20000 --save_path=~/models/fetch_reach_her --log_path=~/log/fetch_reach_her

python runner.py --alg=her --env=FetchDynamicReach-v2 --env_type=robotics --num_timesteps=5000 --load_path=~/models/fetch_reach_her --play

python runner.py --alg=her --env=FetchDynamicTestReach-v2 --env_type=robotics --num_timesteps=5000 --load_path=~/models/fetch_reach_her --play


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
