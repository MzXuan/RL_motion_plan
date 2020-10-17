from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import time


def load_csv():
    df=pd.read_csv('/home/xuan/Documents/human_data.csv', sep=',',header=None)
    print(df.head(5))
    return df


#todo: dt need to be updated from origin time steps
def kf_filter():
    dt = 1/30
    kf = KalmanFilter(dim_x=6, dim_z=6, dim_u=3)
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


    # kf.H = np.array([[1, 0, 0, 0, 0, 0],
    #                 [0, 1, 0, 0, 0, 0],
    #                 [0, 0, 1, 0, 0, 0]])

    kf.H = np.eye(6)

    # kf.P = np.eye(6)*0.01

    kf.P = np.array([[1.00199823e+00, 0.00000000e+00, 0.00000000e+00, 6.65341280e-05,
                           0.00000000e+00, 0.00000000e+00],
                          [0.00000000e+00, 1.00199823e+00, 0.00000000e+00, 0.00000000e+00,
                           6.65341280e-05, 0.00000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 1.00199823e+00, 0.00000000e+00,
                           0.00000000e+00, 6.65341280e-05],
                          [6.65341280e-05, 0.00000000e+00, 0.00000000e+00, 1.00199602e+00,
                           0.00000000e+00, 0.00000000e+00],
                          [0.00000000e+00, 6.65341280e-05, 0.00000000e+00, 0.00000000e+00,
                           1.00199602e+00, 0.00000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 6.65341280e-05, 0.00000000e+00,
                           0.00000000e+00, 1.00199602e+00]]) #covariance matrix
    kf.Q = np.eye(6)*0.5#process matrix (小=相信模型）



    kf.R = np.array([[0.2, 0, 0, 0, 0, 0],
                         [0, 0.2, 0, 0, 0, 0],
                         [0, 0, 0.2, 0, 0, 0],
                         [0, 0, 0, 3, 0, 0],
                         [0, 0, 0, 0, 3, 0],
                         [0, 0, 0, 0, 0, 3]]) #measurement uncertantity


    return kf

#
# def butter_lowpass_filter(data, cutoff, fs, order):
#     nyq = 0.5 * fs  # Nyquist Frequency
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data, padlen=0)
#     return y

data_frame = load_csv()
timesteps = (data_frame.loc[:,0]).to_numpy()
# timesteps = timesteps[:]-timesteps[0]

elbow = (data_frame.loc[:, 1:3]).to_numpy()
hand = (data_frame.loc[:, 4:6]).to_numpy()

kf_elbow = kf_filter()
kf_hand = kf_filter()

pos_hand = hand[0, :]
vel_hand=acc_hand = np.zeros(3)

kf_hand.x = np.concatenate([pos_hand, vel_hand]).T


measure_x = []
filter_x = []
pred_x = []
acc_list = []
vel_list = []



for i in range(1, len(timesteps)):
    #simulate online process
    pos_elbow = elbow[i, :]
    pos_hand = hand[i, :]

    if timesteps[i] == timesteps[i-1]:
        vel_elbow = acc_elbow = vel_hand = acc_hand = np.zeros(3)
        dt = 1/30
    else:
        # vel_elbow = elbow[i,:]/(timesteps[i]-timesteps[i-1])
        # acc_elbow = vel_elbow/(timesteps[i]-timesteps[i-1])
        vel_hand = (hand[i, :]-hand[i-1,:]) / (timesteps[i] - timesteps[i - 1])
        acc_hand = vel_hand / (timesteps[i] - timesteps[i - 1])
        dt = (timesteps[i] - timesteps[i - 1])

    if np.linalg.norm(vel_hand)>3:
        vel_hand = 3*vel_hand/np.linalg.norm(vel_hand)
    if np.linalg.norm(acc_hand)>10:
        acc_hand = 10*acc_hand/np.linalg.norm(acc_hand)


    acc_list.append(acc_hand)
    vel_list.append(vel_hand)

    # z =np.expand_dims(np.concatenate([pos_hand, vel_hand]), axis=1)
    z = np.concatenate([pos_hand, vel_hand]).T

    s_time = time.time()

    # #prepare
    kf_hand.F = np.array([[1, 0, 0, dt, 0, 0],
                                 [0, 1, 0, 0, dt, 0],
                                 [0, 0, 1, 0, 0, dt],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])

    kf_hand.B = np.array([[0.5 * dt ** 2, 0, 0],
                                 [0, 0.5 * dt ** 2, 0],
                                 [0, 0, 0.5 * dt ** 2],
                                 [dt, 0, 0],
                                 [0, dt, 0],
                                 [0, 0, dt]])

    # update
    kf_hand.predict(u=acc_hand)
    kf_hand.update(z=z)

    # print("x current is: ", z)
    measure_x.append(z.copy())
    filter_x.append(kf_hand.x_post.copy())
    pred_x.append(kf_hand.get_prediction(u=acc_hand)[0])

    print(time.time()-s_time)



    # print("p", kf_hand.P)
    # print("q", kf_hand.Q)
    # print("R", kf_hand.R)
    # print("K", kf_hand.K)
    #
    # print("-------------------------------------")


measure_x = np.asarray(measure_x)
filter_x = np.asarray(filter_x)
pred_x = np.asarray(pred_x)


for j in range(3):
    plt.plot(timesteps[1:], measure_x[:,j], 'r--')
    plt.plot(timesteps[1:], filter_x[:,j], 'bs')
    plt.plot(timesteps[2:], pred_x[:-1,j], 'g^')

    plt.show()








