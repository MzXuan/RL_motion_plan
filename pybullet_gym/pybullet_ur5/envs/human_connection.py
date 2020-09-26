from kinect2.client import Kinect2Client
import time
import json


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
        self.human_id = None


    def callback_skeleton(self, msg):
        human_id = list(msg.keys())[0]
        print("human id: ", human_id)
        value_dict = msg[human_id]

        # check is this the previous human
        if self.human_id is None:
            self.human_id = human_id
        else:
            if self.human_id !=human_id:
                return

        #copy data to joint info
        for joint_name, value in value_dict.items():
            try:
                self.joints[joint_name] = value['Position']
            except:
                self.joints.update({joint_name:value})



if __name__ == "__main__":
    hm = HumanModel()
    while True:
        try:
            time.sleep(0.005)
        except KeyboardInterrupt:
            raise
