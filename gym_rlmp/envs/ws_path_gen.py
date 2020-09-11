import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations


class WsPathGen():
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.sample_path(start, end)


    def sample_path(self, start, end):
        waypoints = np.linspace(start, end, num=10)
        waypoints += (np.random.rand(waypoints.shape[0],waypoints.shape[1])-0.5)*0.05
        self.path = waypoints


    def next_goal(self, center, r):
        dists = np.linalg.norm((self.path-center),axis=1)
        print("dist is: ", dists)
        indices = np.where(dists < r)[0]

        print("indices", indices)

        inverse_indices = np.flip(indices)

        for i in inverse_indices: # from end to start
            if i+1 < len(dists): # not meet limit
                print("current i is: ", i)
                p_insect = self.calculate_interaction(center, r, self.path[i], self.path[i+1])
                if p_insect is not None:
                    print("returned intersection is: ", p_insect)
                    return p_insect
            else:
                #the end, return the last way point
                return self.path[-1]

        #no interection, return the waypoint with minimum distance
        # todo: or expand r please
        ####!!!!todo!!!! problem here######
        idx = np.argmin(dists)
        return self.path[idx]


    def calculate_interaction(self, center, r, p0, p1):
        # check intersection
        # C*t^2+B*t+A =0
        xc, yc, zc = center[0], center[1], center[2]
        x0, y0, z0 = p0[0], p0[1], p0[2]
        x1, y1, z1 = p1[0], p1[1], p1[2]

        A=(x0-xc)**2+(y0-yc)**2+(z0-zc)**2-r**2
        C=(x0-x1)**2+(y0-y1)**2+(z0-z1)**2
        B=(x1-xc)**2+(y1-yc)**2+(z1-zc)**2-A-C-r**2

        coeff = [C, B, A]
        t_roots = np.roots(coeff)

        t_cons = []
        for t in t_roots:
            if t >=0 and t <=1:
                t_cons.append(t)
        try:
            t=max(t_cons)
        except:
            print("not found valid solution")
            return None
        else:
            print("solved root are: ", t_roots)
            print("t_cons are: ", t_cons)
            #find intersection
            point_intersect = p0*[1-t]+p1*t
            print("p0 {} and p1 {}".format(p0,p1))
            print("point_intersect is: ", point_intersect)
            return point_intersect



def create_ball(center, r):
    # Create a sphere
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:10j, 0.0:2.0 * pi:10j]
    x = r * sin(phi) * cos(theta) + center[0]
    y = r * sin(phi) * sin(theta) + center[1]
    z = r * cos(phi) + center[2]
    return x,y,z


def plot_result(waypoints, center,  goal):
    #---- plot ---#
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    bx, by, bz = create_ball(center, 0.05)
    ax.plot_surface(
        bx, by, bz, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2],'-*')

    #print intersection point
    ax.plot([goal[0]], [goal[1]], [goal[2]],'+')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0,0.8)
    ax.set_ylim(0, 0.8)
    ax.set_zlim(0, 0.8)
    plt.show()




if __name__ == '__main__':
    start = [0.5,0.6,0.1]
    end =[0.1,0.5,0.3]
    ws_path_gen = WsPathGen(start, end)
    waypoints = ws_path_gen.path

    center = waypoints[0,:]
    radius = 0.05

    for _ in range(0,20):
        ws_path_gen.next_goal(center, radius)
        goal = ws_path_gen.next_goal(center, radius)
        plot_result(waypoints, center, goal)
        center = goal+np.random.uniform(-0.05, 0.05)




