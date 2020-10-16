from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd



def load_csv():
    df=pd.read_csv('/home/xuan/Documents/human_data.csv', sep=',',header=None)
    print(df.head(5))
    return df


#todo: dt need to be updated from origin time steps
def kf_filter():
    dt = 1/30
    kf = KalmanFilter(dim_x=6, dim_z=3, dim_u=3)
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0,  0],
                     [0, 0, 0, 0, 1,  0],
                     [0, 0, 0, 0, 0,  1]])

    kf.B = np.array([[0.5*dt**2, 0, 0],
                     [0, 0.5*dt**2, 0],
                     [0, 0, 0.5*dt**2],
                     [dt, 0, 0],
                     [0, dt, 0],
                     [0, 0, dt]])

    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0]])
    return kf



#todo: initial value for filter

data_frame = load_csv()
timesteps = (data_frame.loc[:,0]).to_numpy()
elbow = (data_frame.loc[:, 1:4]).to_numpy()
hand = (data_frame.loc[:, 4:7]).to_numpy()



