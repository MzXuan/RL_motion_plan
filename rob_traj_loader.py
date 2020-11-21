from pybullet_ur5.envs.ur5_control import UR5Control
import pickle

from gym_rlmp.envs.ws_path_gen import WsPathGen
import time

import numpy as np


def main():
    try:
        with open('/home/xuan/demos/demo5.pkl', 'rb') as handle:
            data = pickle.load(handle)
        print("load data successfully")
    except:
        print("fail to load data, stop")
        return

    print("length of data is: ", len(data))
    return data

def move_to_start(ur5, data):
    start_joint = data[0]['robjp']
    print(f"move to {start_joint}, please be careful")
    ur5.set_joint_position(start_joint, wait=True)
    print(f"success move to {start_joint}")

def move_along_path(ur5, ws_path_gen, dt=0.02):
    toolp,_,_ = ur5.get_tool_state()
    next_goal, next_vel = ws_path_gen.next_goal(toolp, 0.08)

    ref_vel = (next_goal-toolp)
    print(f"current goal {toolp}, next goal {next_goal}, next_vel {next_vel}, ref_vel {ref_vel}")
    tool_vel = np.zeros(6)
    tool_vel[:3]=ref_vel
    ur5.set_tool_velocity(tool_vel)




if __name__ == '__main__':
    # ur5 = UR5Control(ip='192.168.0.3')
    data = main()
    #move to start
    # move_to_start(ur5, data)

    # move along path

    # ws_path_gen = WsPathGen(data[0]['toolp'], data[-1]['toolp'])
    # ws_path_gen.path = [data[i]['toolp'] for i in range(len(data))]

    path = [data[i]['toolp'] for i in range(len(data))]

    print(data[-1]['robjp'])
    # vel_path = [data[i]['toolv'] for i in range(len(data))]
    # ws_path_gen = WsPathGen(path, vel_path)
    # last_time = time.time()
    # while True:
    #     try:
    #         dt= time.time()-last_time
    #         move_along_path(ur5, ws_path_gen, dt)
    #         last_time = time.time()
    #     except KeyboardInterrupt:
    #         ur5.close()
    #         raise







