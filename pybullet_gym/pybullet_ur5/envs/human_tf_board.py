import pickle
import numpy as np
import tf
import rospy

class FileHuman(object):
    def __init__(self, file):
        try:
            with open(file, 'rb') as handle:
                self.joint_queue_list = pickle.load(handle)
            print("load human data successfully")
            self.data_length = len(self.joint_queue_list)
        except:
            print("!!!!!!!!!!!!!!fail to load data !!!!!!!")
            exit()

        self.index = 0
        # self.joint_queue = self.joint_queue_list[0]

        self.update_joint_queue()


    def update_joint_queue(self):
        print("self.index: ", self.index)
        if self.index > self.data_length-1:
            self.index = np.random.randint(low=0, high=self.data_length - 1)
        self.joint_queue = self.joint_queue_list[self.index]

        self.index += 1


if __name__ == "__main__":
    file_human = FileHuman(file='/home/xuan/demos/human_data_normal_py2.pkl')
    name_list = ["SpineBase", "SpineMid","SpineShoulder", "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft"]

    print(file_human.joint_queue["SpineBase"])

    rospy.init_node('fixed_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)


    while not rospy.is_shutdown():
        # for name in name_list:
        for i in range(len(name_list)):
            if i == 0:
                name = name_list[i]
                name2 = "world"
            else:
                name = name_list[i]
                name2 = name_list[i-1]
            joint = file_human.joint_queue[name]
            br.sendTransform((joint[:3]),
                             (joint[3], joint[4], joint[5],joint[6]),  #x,y,z, w
                             rospy.Time.now(),
                             name,
                            "world")
        rospy.sleep(0.1)
        file_human.update_joint_queue()