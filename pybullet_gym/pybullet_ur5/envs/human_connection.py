from kinect2.client import Kinect2Client
import time
import json
import numpy as np

# class Joint(object):
#     def __init__(self, name, position):
#         self.joint_name = name
#         self.position = position


class HumanModel(object):
# real time information for human model
    def __init__(self, ip="192.168.0.10"):
        self.kinect = Kinect2Client(ip)
        self.kinect.skeleton.set_callback(self.callback_skeleton)
        self.kinect.skeleton.start()
        self.joints = {}
        self.joint_velocity = {}
        self.dt = 1/30
        self.human_id = None


    #todo: add function
    def get_joint_state(self, joint_name):
        return [self.joints[joint_name],self.joint_velocity[joint_name]]



    def callback_skeleton(self, msg):
        human_id = list(msg.keys())[0]
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

                self.joints[joint_name] = np.asarray([value['Position']['X'], value['Position']['Y'], value['Position']['Z']])
                self.joint_velocity[joint_name] = (self.joints[joint_name]-last_joint_position)/self.dt
            except:
                self.joints.update({joint_name:np.asarray([value['Position']['X'], value['Position']['Y'], value['Position']['Z']])})
                self.joint_velocity.update({joint_name:0})

        # print("value dict", self.joints)
        # print("joint velocity", self.joint_velocity['HandRight'])

if __name__ == "__main__":
    hm = HumanModel()
    while True:
        try:
            time.sleep(0.005)
            print("hm.joints", hm.joints)

        except KeyboardInterrupt:
            raise
