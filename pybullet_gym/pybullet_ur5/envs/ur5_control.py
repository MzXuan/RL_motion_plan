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
        pose = self.rob.get_tcp_speed()
        vel = pose.array
        p,rot = self.rob.get_tcp_pose()
        return p, vel, rot

    def get_tool_state_2(self):
        pos = self.rob.get_pos().array
        return pos

    def set_tool_velocity(self, velocity):
        # ---- prevent from singaularity
        p, vel,_ = self.get_tool_state()
        if np.linalg.norm(p) > 0.85:
            # print("warining!!! close to singularity!!!!!")
            velocity = np.zeros(6)
            velocity[:3] = -p/np.linalg.norm(p)*0.1
        # print("p, velocity", p ,velocity)
        # ---normal speedl command
        self.rob.speedl(velocities=velocity, acc=0.2, min_time=2)


    def set_joint_position(self, target, wait=True):
        self.rob.movej(target, acc=0.3, vel=0.2, wait=wait)


    def set_joint_positions(self, target_list, wait=False):
        self.rob.movejs(target_list, acc=0.3, vel=0.2, wait=wait)

    def servo_joint_position(self, target, wait=True):
        #not working
        self.rob.servoj(target, wait=wait)


    def set_joint_velocity(self, target):
        # target = target.clip(min=-0.04, max=0.04)
        print("target velocity", target)
        self.rob.speedj(target,acc=0.2, min_time=10)


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


    #
    tool_vel = [-0.01, 0, +0.05, 0, 0, 0]
    # target_v = np.asarray(robjv)+0.02
    # ur5.set_joint_velocity(target_v)


    while True:
        try:

            ur5.set_tool_velocity(tool_vel)
            # print("target v", target_v)
            # robjp, robjv = ur5.get_joint_state()
            # p, vel = ur5.get_tool_state()
            # print("postion, {} and vel {}".format(p, vel))
            #
            # ur5.set_joint_velocity(target_v)
            # print("robv", robjv)

        except KeyboardInterrupt:
            ur5.close()
            raise
    #
    # target_v = np.asarray(robjv)-0.02
    # # ur5.set_joint_velocity(target_v)
    #
    # while True:
    #     try:
    #         # print("target v", target_v)
    #         robjp, robjv = ur5.get_joint_state()
    #         test = ur5.get_tool_state()
    #         ur5.set_joint_velocity(target_v)
    #         # print("robv", robjv)
    #
    #     except KeyboardInterrupt:
    #         ur5.close()
    #         raise

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


