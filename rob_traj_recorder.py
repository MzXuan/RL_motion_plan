from pybullet_ur5.envs.ur5_control import UR5Control
import pickle
import time
import numpy as np

if __name__ == '__main__':
    ur5 = UR5Control(ip='192.168.0.3')
    # todo: record robot moving data


    robjp, robjv = ur5.get_joint_state() #radius
    print("robot joint position is: ", robjp)
    print("robot joint velocity is: ", robjv)


    tool_state = ur5.get_tool_state()
    print("tool state, ", tool_state)


    # press to start
    value = input("Press any key to start recording:\n")
    print(f'You entered {value}, now start recording')
    # press to stop

    step = 0
    start_time = time.time()
    data = []
    while True:
        try:
            robjp, robjv = ur5.get_joint_state()  # radius
            tool_state = ur5.get_tool_state()
            timestep = time.time()-start_time

            print("robot joint position is: ", robjp)
            print("robot joint velocity is: ", robjv)
            print("tool state, ", tool_state)
            data.append({'timestep':timestep, 'robjp':robjp, 'robjv':robjv,
                         'toolp': tool_state[0],'toolv': tool_state[1]})
            time.sleep(0.02)

        except KeyboardInterrupt:
            idx = 0
            for i in range(len(data)-1):
                if np.linalg.norm(np.array(data[i]['robjp'])-
                                  np.array(data[i+1]['robjp']))>0.01:
                    idx = i
                    break

            with open('/home/xuan/demos/task_demo3.pkl', 'wb') as handle:
                pickle.dump(data[idx:], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("save successfully")
            ur5.close()
            raise