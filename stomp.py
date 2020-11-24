import numpy as np
import matplotlib.pyplot as plt

import time

MIN_COST_DIFFERENCE = 1e-8
MAX_ITERATION_STEPS = 20
STOP_TOLERANCE = 1e-2
class STOMP:

    def __init__(self, D, N, K, initial_traj = None, collision_fn=None, ik_fn=None):
        '''
        D: dof
        N: steps, include start and end
        K: number of noisy trajectory
        '''
        n = N+2
        A_k = np.eye(n-1)
        A = -2 * np.eye(n)
        A[0:n-1,1:n] = A[0:n-1, 1:n] + A_k
        A[1:n, 0:n-1] = A[1:n, 0:n-1] + A_k
        self.A = A[:, 1:n-1]

        self.R = np.matmul(self.A.T,self.A)

        self.R_inv = np.linalg.inv(self.R)
        self.M = self.R_inv/np.max(self.R_inv,axis=1)/5
        self.N = N
        self.K = K
        self.D = D
        self.Q = initial_traj

        if collision_fn is None:
            self.collision_fn = lambda x: 10 if np.linalg.norm(x)<1.5 else 0
        else:
            self.collision_fn = collision_fn

        if ik_fn is None:
            self.ik_fn = lambda x: x
        else:
            self.ik_fn = ik_fn



    def plan(self, q0, qN):
        if self.Q is None:
            self.Q = np.linspace(q0, qN, self.N) #N*D
            self.Q_cost = np.sum(self.compute_rollout_cost(self.Q)) + self.compute_controll_cost(self.Q)

        step = 0
        while step<MAX_ITERATION_STEPS:
            Q_update = self.run_single_iteration()
            update_cost = np.sum(self.compute_rollout_cost(Q_update)) + self.compute_controll_cost(Q_update)

            step+=1
            #stop or not?
            print("last_cost is", self.Q_cost)
            print("update cost", update_cost)

            if abs(update_cost-self.Q_cost) < STOP_TOLERANCE:
                break
            self.Q = Q_update
            self.Q_cost = update_cost

        #----for debug-----#
        # plot_obst()
        # plot_traj(self.Q.copy())
        return self.Q.copy()


    def run_single_iteration(self):
        # plot_obst()
        #todo: update step

        ts = time.time()
        #1. create K noisy trajectory:
        noisy_paramters, noisy_trajectories = self.generate_noisy_parameter() #K*N*D

        #2. compute cost of rollouts
        Q_update = self.update_parameter(noisy_paramters, noisy_trajectories)
        print("solving time is:", time.time() - ts)

        # plot_obst()
        # plot_rollouts(noisy_trajectories)
        # plot_traj(Q_update)
        return Q_update



    def generate_noisy_parameter(self):
        noisy_paramters = []
        noisy_trajectories = []
        for k in range(self.K):
            # ek = np.random.multivariate_normal(np.zeros(self.N), 1/self.N*self.R_inv/np.max(self.R_inv),self.D)
            ek = np.random.multivariate_normal(np.zeros(self.N),  1/self.N*self.R_inv / np.max(self.R_inv), self.D)
            # ek = np.random.multivariate_normal(np.zeros(self.N), self.R_inv, self.D)
            ek[:,0] = 0
            ek[:,-1] = 0
            ek=ek.T
            noisy_paramters.append(ek)
            noisy_trajectories.append(ek+self.Q.copy())

        return np.array(noisy_paramters), np.array(noisy_trajectories)



    def update_parameter(self, noisy_paramters, noisy_trajectories):
        #compute state costs
        cost_matrix = []
        for rollout in noisy_trajectories:
            cost_matrix.append(self.compute_rollout_cost(rollout))
        cost_matrix=np.array(cost_matrix) #K*N
        #compute probability
        probabilities = self.compute_probabilities(cost_matrix)
        #calculate update parameters
        # #--------------------or debug--------------------------
        # for n in range(self.N):
        #     for k in range(self.K):
        #         if probabilities[k,n] !=0:
        #             plt.text(noisy_trajectories[k,n,0], noisy_trajectories[k,n,1], str("%.2f" % probabilities[k,n]), fontsize=12)

        normal_delta_theta = self.compute_delta_parameters(probabilities, noisy_paramters)


        return self.Q.copy()+normal_delta_theta


    def compute_rollout_cost(self, rollout):
        '''
        rollout:N*D
        '''
        state_costs = []
        for i in range(self.N):
            q = rollout[i,:]
            state_costs.append(self.compute_state_cost(q))
        return np.array(state_costs)


    def compute_delta_parameters(self, probabilities, noisy_paramters):
        delta_theta = \
            [np.sum(probabilities * noisy_paramters[:,:,i],axis=0)   for i in range(self.D)]
        # delta_theta = np.sum(probabilities*noisy_paramters, axis=0)
        delta_theta = np.array(delta_theta).T
        normal_delta_theta = np.matmul(self.M, delta_theta)

        normal_delta_theta[0,:] = 0
        normal_delta_theta[-1, :] = 0

        # print("probabilities", probabilities)
        # print("delta_theta", delta_theta)
        # print("delta theta: ", normal_delta_theta)

        return normal_delta_theta


    def compute_probabilities(self, cost_matrix):
        '''
        cost_matrix: K*N
        '''
        exp_cost_matrix = np.zeros((self.K, self.N))
        for n in range(self.N): #for column in cost matrix
            c_n = cost_matrix[:,n]
            c_max = np.max(c_n)
            c_min = np.min(c_n)
            denom = c_max - c_min
            if denom < MIN_COST_DIFFERENCE:
                denom = MIN_COST_DIFFERENCE
            exp_cost_matrix[:,n]= np.exp(-10*(c_n-c_min)/denom) #equation 11
        # print("exp cost matrix: ", exp_cost_matrix)

        probabilities = np.zeros((self.K, self.N))
        for k in range(self.K):
            #follow equation 11
            probabilities[k,:] = (exp_cost_matrix[k,:]/(np.sum(exp_cost_matrix,axis=0)+MIN_COST_DIFFERENCE))
        return probabilities


    def compute_state_cost(self, q):
        #todo: add state cost
        conf = self.ik_fn(q)
        is_collision = self.collision_fn(conf)
        # print("cost collision", is_collision)
        if is_collision :
            cost_collision = 10
        else:
            cost_collision = 0
        cost =cost_collision
        return cost

    def compute_controll_cost(self, Q):
        control_cost = np.linalg.norm(np.matmul(self.A, Q), ord=2)
        return control_cost



def plot_obst():
    circle = plt.Circle((0, 0), 1.5, color='y')
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    ax.add_artist(circle)

def plot_rollouts(rollouts):
    for r in rollouts:
        plt.plot(r[:,0], r[:,1], 'r-')

def plot_traj(last_traj):

    plt.plot(last_traj[:,0], last_traj[:,1], 'b-')
    plt.show()



