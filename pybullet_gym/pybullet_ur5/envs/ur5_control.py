from urx import Robot
import numpy as np
import pyquaternion
import math3d as m3d
import time

class UR5Control():

    def __init__(self, ip='192.168.0.3', use_rt=True):
        self.rob = Robot(ip, use_rt=use_rt)
        self.rob.set_tcp((0, 0, 0, 0, 0, 0))
        self.rob.set_payload(0.1, (0, 0, 0))

    def get_joint_state(self):
        robjp, robjv = self.rob.getj(getv=True)

        return robjp, robjv

    def get_tool_state(self):
        p = self.rob.get_pos().array
        ori = self.rob.get_orientation().quaternion.array #x,y,z,w
        return np.concatenate([p, ori])

    def set_tool_velocity(self):
        return 0

    def set_joint_position(self, target):
        self.rob.movej(target, acc=0.2, vel=0.05)


    def set_joint_velocity(self, target):
        target = target.clip(min=-0.08, max=0.08)
        self.rob.speedj(target,acc=0.3, min_time=10)


    def stop(self):
        self.rob.stop()


    def close(self):
        self.stop()
        self.rob.close()



if __name__ == '__main__':
    ur5 = UR5Control(ip='192.168.0.3')
    robjp, robjv = ur5.get_joint_state() #radius
    print("robot joint position is: ", robjp)
    print("robot joint velocity is: ", robjv)


    tool_state = ur5.get_tool_state()
    print("tool state, ", tool_state)

    target_v = np.asarray(robjv)-0.02
    ur5.set_joint_velocity(target_v)

    while True:
        try:
            print("target v", target_v)
            robjp, robjv = ur5.get_joint_state()
            print("robv", robjv)

        except KeyboardInterrupt:
            ur5.close()
            raise

    # target_p=np.asarray(robjp)-0.1
    #
    # print("target_p is:", target_p)
    # ur5.set_joint_position(target_p)
    # print("move success fully")

    # rob = Robot("192.168.0.3")
    # rob.x  # returns current x
    #
    # trans = rob.get_pose()
    # print("robot current tans is: ", trans)


