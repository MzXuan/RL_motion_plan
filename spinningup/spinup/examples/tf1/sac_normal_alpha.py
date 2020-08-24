from spinup.utils.run_utils import ExperimentGrid
from spinup import safe_rl
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac_normal')
    eg.add('env_name', 'UR5HumanCollisionEnv-v0')
    eg.add('alpha', [0.2, 0.02, 2])
#     eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 1000)
    eg.run(safe_rl)