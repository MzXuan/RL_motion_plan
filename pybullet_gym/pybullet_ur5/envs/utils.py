import numpy as np
from pyquaternion import Quaternion


def quaternion_to_euler(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw])


def quaternion_inv(q):
    #x,y,z,w

    q_con = [-q[0],-q[1],-q[2], -q[3]]
    q_inv = q_con/np.linalg.norm(q)
    return q_inv

def quaternion_diff(q1, q2):
    #q1 and q2 is x, y, z,w

    #for quaternion library
    q10 = Quaternion(q1[0],q1[1],q1[2],q1[3])
    q20 = Quaternion(q2[0],q2[1],q2[2],q2[3])
    q = q20 * q10.inverse

    # print("q10 {} and q20 {}".format(q10, q20))
    angle = 2 * np.arccos(abs(q[0]))
    return angle



if __name__ == '__main__':

    q1 = Quaternion(np.asarray([0.707, 0, 0.707, 0]))
    q2 = Quaternion(0.707, 0, 0.707, 0)
    q= q2*q1.inverse
    angle = 2*np.arccos(abs(q[0]))
    print("new q is {} and angle is {}".format(q, angle))
    # print("quaternion diff", quaternion_diff(q1, q2))