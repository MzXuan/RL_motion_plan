import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def df_plot(dfs, x, ys, ylim=None, legend_loc='best'):
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.style.use('ggplot')
    if ylim:
        plt.ylim(ylim)

    plt.plot(dfs[x]/3600, dfs[ys], linewidth=1, label=ys)
    plt.xlabel(x)
    plt.legend(loc=legend_loc)
    plt.show()

def main(fpath):
    filepath = os.path.join(fpath,"progress.csv")
    print("read file from {}".format(filepath))
    dataframes = []

    data = pd.read_csv(filepath)
    print(data.head(10))
    # data['train/episode'] = data['train/episode']*4000000
    # data['test/episode'] = data['test/episode'] * 4000000
    #for ppo
    # df_plot(data, 'time_elapsed', 'policy_entropy')
    # df_plot(data, 'time_elapsed', 'eprewmean')
    # df_plot(data, 'time_elapsed', 'explained_variance', ylim=(-1, 1))
    # df_plot(data, 'time_elapsed', 'pred_loss')
    # df_plot(data, 'time_elapsed', 'origin_rew')

    #for her and ddpg
    df_plot(data, 'train/episode', 'train/success_rate', ylim=(0,1))
    df_plot(data, 'test/episode', 'test/success_rate', ylim=(0,1))
    # df_plot(data, 'time_elapsed', 'eprewmean')
    # df_plot(data, 'time_elapsed', 'explained_variance', ylim=(-1, 1))

    # df_plot(data, 'time_elapsed', 'lr')
    # df_plot(data, 'time_elapsed', 'policy_loss')
    # df_plot(data, 'time_elapsed', 'value_loss')
    # df_plot(data, 'time_elapsed', 'approxkl')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default=None)
    args = parser.parse_args()

    main(args.fpath)


