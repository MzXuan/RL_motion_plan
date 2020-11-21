import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
def make_plots(logdir):

    df = pd.read_csv(os.path.join(logdir, "progress.csv"))
    print(df.head(5))
    sns.lineplot(data=df, x="train/episode", y="train/success_rate")
    plt.show()



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir',type=str, default=None)
    args = parser.parse_args()

    make_plots(args.logdir)


if __name__ == "__main__":
    main()