from kinect2.client import Kinect2Client
import time
import json
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt


def kf_filter():
    dt = 1 / 30
    kf = KalmanFilter(dim_x=6, dim_z=6, dim_u=3)

    kf.x = np.zeros(6)
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

    kf.B = np.array([[0.5 * dt ** 2, 0, 0],
                     [0, 0.5 * dt ** 2, 0],
                     [0, 0, 0.5 * dt ** 2],
                     [dt, 0, 0],
                     [0, dt, 0],
                     [0, 0, dt]])

    kf.H = np.eye(6)

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
                      0.00000000e+00, 1.00199602e+00]])  # covariance matrix
    kf.Q = np.eye(6)*0.2   # process matrix (小=相信模型）

    kf.R = np.array([[0.2, 0, 0, 0, 0, 0],
                     [0, 0.2, 0, 0, 0, 0],
                     [0, 0, 0.2, 0, 0, 0],
                     [0, 0, 0, 3, 0, 0],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 0, 3]])  # measurement uncertantity

    return kf





class HumanModel(object):
# real time information for human model
    def __init__(self, ip="192.168.0.10"):
        self.kinect = Kinect2Client(ip)
        self.joints = {}
        self.joint_velocity = {}
        self.joint_accerlation = {}
        self.joint_queue = {}

        # self.dt = 1/30
        self.human_id = None
        self.last_time_stamp = time.time()
        self.filters = {'ElbowLeft': kf_filter(), 'HandLeft': kf_filter()} #elbow, hand
        self.filter_joint_name = ['ElbowLeft', 'HandLeft']

        self.kinect.skeleton.set_callback(self.callback_skeleton)
        self.kinect.skeleton.start()



    def reset(self):
        print("reset")
        # self.kinect.skeleton.stop()
        print("stop")
        self.joints = {}
        self.joint_velocity = {}
        self.joint_accerlation = {}
        self.joint_queue = {}
        # self.dt = 1/30
        self.human_id = None
        self.last_time_stamp = time.time()
        self.filters = {'ElbowLeft': kf_filter(), 'HandLeft': kf_filter()}  # elbow, hand

        print("start")

        # self.kinect.skeleton.start()



    def get_joint_state(self, joint_name):
        return [self.joints[joint_name],self.joint_velocity[joint_name]]


    def callback_skeleton(self, msg):


        human_id = list(msg.keys())[0]

        current_time =time.time()

        dt = 0.033


        dt = current_time-self.last_time_stamp
        #
        if dt>1:
            self.reset()

        # print("human id: ", human_id)
        value_dict = msg[human_id]

        # check is this the previous human
        if self.human_id is None:
            self.human_id = human_id
        else:
            if self.human_id != human_id:
                print("human id not match")
                return

        #copy data to joint info
        for joint_name, value in value_dict.items():

            try:
                last_joint_position = self.joints[joint_name].copy()
                last_joint_velocity = self.joint_velocity[joint_name].copy()
                self.joints[joint_name] = np.asarray(
                    [value['Position']['X'], value['Position']['Y'], value['Position']['Z']])
            except:
                last_joint_position = self.joints[joint_name] = np.asarray(
                    [value['Position']['X'], value['Position']['Y'], value['Position']['Z']])
                last_joint_velocity = np.zeros(3)


            self.joint_velocity[joint_name] = (self.joints[joint_name] - last_joint_position) / dt
            self.joint_accerlation[joint_name] = (self.joint_velocity[joint_name]-last_joint_velocity) / dt


            # --- add kalman filter----#
            if joint_name in self.filter_joint_name:

                # print("self.joint velocity", self.joint_velocity)

                vel = self.joint_velocity[joint_name]
                acc = self.joint_accerlation[joint_name]


                if np.linalg.norm(vel) > 3:
                    vel = 3.0 * vel / np.linalg.norm(vel)
                if np.linalg.norm(acc) > 20:
                    acc = 20.0 * acc / np.linalg.norm(acc)


                #prepare kalman filter
                self.filters[joint_name].F = \
                                np.array([[1, 0, 0, dt, 0, 0],
                                 [0, 1, 0, 0, dt, 0],
                                 [0, 0, 1, 0, 0, dt],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])

                self.filters[joint_name].B =\
                                 np.array([[0.5 * dt ** 2, 0, 0],
                                 [0, 0.5 * dt ** 2, 0],
                                 [0, 0, 0.5 * dt ** 2],
                                 [dt, 0, 0],
                                 [0, dt, 0],
                                 [0, 0, dt]])

                z= np.concatenate([self.joints[joint_name], vel]).T

                # print("z", z)
                # history_joint = self.filters[joint_name].x.copy()[:3]
                #update kalman filter
                self.filters[joint_name].predict(acc)
                self.filters[joint_name].update(z)
                current_joint = self.filters[joint_name].x.copy()



                #predict step
                next_joint = self.filters[joint_name].get_prediction(acc)[0]
                next_next_joint = self.filters[joint_name].get_prediction(acc, x=next_joint)[0]

                # print("history joint", history_joint)
                # print("current joint", current_joint)
                self.joint_queue[joint_name] = [current_joint[:3], next_joint[:3], next_next_joint[:3]]



                # print("sef.joint queue", self.joint_queue)

            self.last_time_stamp = current_time

        # print("value dict", self.joints)
        # print("joint velocity", self.joint_velocity['HandRight'])

if __name__ == "__main__":
    hm = HumanModel()
    #todo: save to csv

    joint_name = ['ElbowLeft', 'HandLeft']
    csv_data = []
    last_time = hm.last_time_stamp
    measure = []
    filter =[]
    predict = []
    further_predict = []
    timestep = []
    while True:
        try:
            time.sleep(0.015)

            if hm.last_time_stamp!=last_time:
                print("sef.joint queue", hm.joint_queue)
                try:
                    measure.append(hm.joints['HandLeft'])
                    filter.append(hm.joint_queue['HandLeft'][0])
                    predict.append(hm.joint_queue['HandLeft'][1])
                    further_predict.append(hm.joint_queue['HandLeft'][2])
                    timestep.append(last_time)
                except:
                    pass
            #     csv_data.append(np.concatenate([
            #         np.asarray([hm.last_time_stamp]), hm.joints['ElbowLeft'], hm.joints['HandLeft']]))
            last_time = hm.last_time_stamp

        except KeyboardInterrupt:
            measure=np.asarray(measure)
            filter = np.asarray(filter)
            predict = np.asarray(predict)
            further_predict = np.asarray(further_predict)
            timestep=np.asarray(timestep)
            for j in range(3):
                plt.plot(timestep[:], measure[:, j], 'r--')
                plt.plot(timestep[:], filter[:, j], 'bs')
                plt.plot(timestep[:], predict[:, j], 'g^')
                plt.plot(timestep[:], further_predict[:,j],'y^')

                plt.show()

            # np.savetxt("/home/xuan/Documents/human_data.csv", np.asarray(csv_data), delimiter=",")
            # print("save csv successfully")
            raise
