from kinect2.client import Kinect2Client
import time
import json
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

import pickle

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
    kf.Q = np.eye(6)*0.4

    kf.R = np.array([[0.5, 0, 0, 0, 0, 0],
                     [0, 0.5, 0, 0, 0, 0],
                     [0, 0, 0.5, 0, 0, 0],
                     [0, 0, 0, 3, 0, 0],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 0, 3]])  # measurement uncertantity

    return kf





class HumanModel(object):
# real time information for human model
    def __init__(self, ip="192.168.0.10"):
        self.kinect = Kinect2Client(ip)
        self.joints = {}
        # self.joints_orientation = {}
        self.joint_velocity = {}
        self.joint_accerlation = {}
        self.joint_queue = {}

        # self.dt = 1/30
        self.human_id = None
        self.last_time_stamp = time.time()
        self.filters = {'ElbowLeft': kf_filter(), 'HandLeft': kf_filter(),
                        'ElbowRight': kf_filter(), 'HandRight': kf_filter()}  # elbow, hand
        self.filter_joint_name = ['ElbowLeft', 'HandLeft', 'ElbowRight', 'HandRight']

        self.count = 0
        self.kinect.skeleton.set_callback(self.callback_skeleton)
        self.kinect.skeleton.start()




    def reset(self):
        print("reset")
        # self.kinect.skeleton.stop()
        print("stop")
        self.joints = {}
        # self.joints_orientation = {}
        self.joint_velocity = {}
        self.joint_accerlation = {}
        self.joint_queue = {}
        # self.dt = 1/30
        self.human_id = None
        self.last_time_stamp = time.time()
        self.filters = {'ElbowLeft': kf_filter(), 'HandLeft': kf_filter(),
                        'ElbowRight': kf_filter(), 'HandRight': kf_filter()}  # elbow, hand

        print("start")

        # self.kinect.skeleton.start()


    def get_joint_state(self, joint_name):
        return [self.joints[joint_name][:3], self.joint_velocity[joint_name]]


    def callback_skeleton(self, msg):

        human_id = list(msg.keys())[0]

        # print("self.human_id is: ", self.human_id)
        # check is this the previous human
        if self.human_id is None:
            self.human_id = human_id
        elif self.human_id != human_id:
            print("!!!!!!!!human id not match!!!!!!current id is: {} ".format(self.human_id))
            return
        current_time =time.time()

        dt = current_time-self.last_time_stamp
        #
        if dt>1:
            self.reset()

        # print("human id: ", human_id)
        value_dict = msg[human_id]


        #copy data to joint info
        for joint_name, value in value_dict.items():

            # self.joints[joint_name] = np.asarray(
            #     [value['Position']['X'], value['Position']['Y'], value['Position']['Z'],
            #      value['Orientation']['X'], value['Orientation']['Y'],
            #      value['Orientation']['Z'], value['Orientation']['W']])
            try:
                last_joint_position = self.joints[joint_name][:3].copy()
                last_joint_velocity = self.joint_velocity[joint_name].copy()
                last_joint_acceleration = self.joint_accerlation[joint_name].copy()
                self.joints[joint_name] = np.asarray(
                    [value['Position']['X'], value['Position']['Y'], value['Position']['Z'],
                     value['Orientation']['X'], value['Orientation']['Y'],
                     value['Orientation']['Z'], value['Orientation']['W']])

            except:
                #first data
                self.joints[joint_name] = np.asarray(
                    [value['Position']['X'], value['Position']['Y'], value['Position']['Z'],
                     value['Orientation']['X'], value['Orientation']['Y'],
                     value['Orientation']['Z'], value['Orientation']['W']])

                last_joint_position = self.joints[joint_name][:3]
                last_joint_velocity = np.zeros(3)
                last_joint_acceleration = np.zeros(3)
                if joint_name in self.filter_joint_name:
                    self.filters[joint_name].x = np.concatenate([last_joint_position.flatten(), last_joint_velocity.flatten()])


            # if np.linalg.norm(self.joints[joint_name] - last_joint_position) >0.5 or \
            if (last_joint_position==0).all():
                self.joint_velocity[joint_name] = np.zeros(3)
                self.joint_accerlation[joint_name] = np.zeros(3)
            elif (last_joint_velocity==0).all():
                self.joint_velocity[joint_name] = (self.joints[joint_name][:3] - last_joint_position) / dt
                self.joint_accerlation[joint_name] = np.zeros(3)
            else:
                self.joint_velocity[joint_name] = (self.joints[joint_name][:3] - last_joint_position) / dt
                self.joint_accerlation[joint_name] = 0.5* ((self.joint_velocity[joint_name]-last_joint_velocity) / dt + \
                                                     last_joint_acceleration)



            # --- add kalman filter----#
            if joint_name in self.filter_joint_name:

                # print("self.joint velocity", self.joint_velocity)
                # if joint_name == "ElbowRight":

                    # print("--------dt{}--- count{}----".format(dt, self.count))
                    # print("position {}, velocity {} and acceleration {}".format(self.joints[joint_name], self.joint_velocity[joint_name],self.joint_accerlation[joint_name]))


                vel = self.joint_velocity[joint_name]
                acc = self.joint_accerlation[joint_name]


                if np.linalg.norm(vel) > 2:
                    vel = 2.0 * vel / np.linalg.norm(vel)
                elif np.linalg.norm(acc) > 10:
                    acc = 10.0 * acc / np.linalg.norm(acc)


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

                z= np.concatenate([self.joints[joint_name][:3], vel]).T

                # print("z", z)
                # history_joint = self.filters[joint_name].x.copy()[:3]
                #update kalman filter
                self.filters[joint_name].predict(acc)
                self.filters[joint_name].update(z)
                current_joint = self.filters[joint_name].x.copy()

                # acc_new = (current_joint[3:6]-last_joint_velocity)/dt

                #predict step
                next_joint = self.filters[joint_name].get_prediction(acc)[0]
                next_next_joint = self.filters[joint_name].get_prediction(acc, x=next_joint)[0]


                # print("history joint", history_joint)
                # print("current joint", current_joint)
                self.joint_queue[joint_name] = [current_joint[:3], next_joint[:3], next_next_joint[:3]]

                # print("sef.joint queue", self.joint_queue)

            self.last_time_stamp = current_time
            self.count +=1
        print("value dict", self.joints)
        # print("joint velocity", self.joint_velocity['HandRight'])


#todo: save human motion;  load human motion

if __name__ == "__main__":
    hm = HumanModel()
    #todo: save to csv

    joint_name = ['ElbowLeft', 'HandLeft']
    last_time = hm.last_time_stamp
    measure = []
    filter =[]
    predict = []
    further_predict = []
    timestep = []

    joint_data_lst = []
    while True:
        try:
            time.sleep(0.015)

            if hm.last_time_stamp!=last_time:
                # print("sef.joint queue", hm.joint_queue)
                try:
                    measure.append(hm.joints['HandLeft'])
                    filter.append(hm.joint_queue['HandLeft'][0])
                    predict.append(hm.joint_queue['HandLeft'][1])
                    further_predict.append(hm.joint_queue['HandLeft'][2])
                    timestep.append(hm.count)

                    joint_data_lst.append(hm.joints.copy())
                    # print("joint queue", hm.joint_queue)
                except:
                    pass

            last_time = hm.last_time_stamp

        except KeyboardInterrupt:
            hm.kinect.skeleton.stop()
            with open('/home/xuan/demos/human_test_6.pkl', 'wb') as handle:
                pickle.dump(joint_data_lst, handle, protocol=2)
                print("save successfully")

            measure=np.asarray(measure)
            filter = np.asarray(filter)
            predict = np.asarray(predict)
            further_predict = np.asarray(further_predict)
            timestep=np.asarray(timestep)


            for j in range(3):
                plt.plot(timestep[:measure.shape[0]], measure[:, j], 'r-')
                plt.plot(timestep[:filter.shape[0]], filter[:, j], 'bs--')
                plt.plot(timestep[1:predict.shape[0]], predict[:-1, j], 'g^-')
                plt.plot(timestep[2:further_predict.shape[0]], further_predict[:-2,j],'y^--')

                plt.show()


            raise
