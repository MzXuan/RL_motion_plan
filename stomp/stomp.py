import numpy as np
import matplotlib.pyplot as plt

# class ROLLOUT:
#     def __init__(self, parameters, theta):
#         self.noisy_paramters = parameters
#         self.theta_parameters = theta + parameters #theta+ek
#
#     def set_state_costs(self, state_costs):
#         self.state_costs = state_costs
#         print("state costs: ", self.state_costs)
#
#

MIN_COST_DIFFERENCE = 1e-8
MAX_ITERATION_STEPS = 10
STOP_TOLERANCE = 1e-2
class STOMP:

    def __init__(self, D, N, K, initial_traj = None, collision_fn = None):
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
        self.M = 1/N * self.R_inv/np.max(self.R_inv,axis=1)
        self.N = N
        self.K = K
        self.D = D
        self.Q = initial_traj
        if collision_fn is None:
            self.collision_fn = lambda x: 1 if np.linalg.norm(x)>1.5 else 0


    def plan(self, q0, qN):
        if self.Q is None:
            self.Q = np.linspace(q0, qN, self.N) #N*D
            self.Q_cost = np.sum(self.compute_rollout_cost(self.Q)) + self.compute_controll_cost(self.Q)


        step = 0
        while step<MAX_ITERATION_STEPS:
            Q_update = self.run_single_iteration()
            update_cost = np.sum(self.compute_rollout_cost(Q_update)) + self.compute_controll_cost(Q_update)
            step+=1
            self.Q = Q_update

            #stop or not?

            print("update cost", update_cost)
            print("Q_cost", self.Q_cost)
            if abs(update_cost-self.Q_cost) < STOP_TOLERANCE:
                break


        return self.Q.copy()


    def run_single_iteration(self):
        #todo: update step
        #1. create K noisy trajectory:
        noisy_paramters, noisy_trajectories = self.generate_noisy_parameter() #K*N*D

        #2. compute cost of rollouts
        Q_update = self.update_parameter(noisy_paramters, noisy_trajectories)

        # debug: print noise_Q and check it
        plot_rollouts(noisy_trajectories, self.N)
        plot_traj(self.Q.copy(), self.N)


        return Q_update


    def generate_noisy_parameter(self):
        noisy_paramters = []
        noisy_trajectories = []
        for k in range(self.K):
            # ek = np.random.multivariate_normal(np.zeros(self.N), 1/self.N*self.R_inv/np.max(self.R_inv),self.D)
            ek = np.random.multivariate_normal(np.zeros(self.N),  self.R_inv / np.max(self.R_inv), self.D)
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
            exp_cost_matrix[:,n]= np.exp(-10*(c_n-c_min)/denom)


        probabilities = np.zeros((self.K, self.N))
        for k in range(self.K):
            #follow equation 11
            probabilities[k,:] = (exp_cost_matrix[k,:]/np.sum(exp_cost_matrix,axis=0))
        return probabilities


    def compute_state_cost(self, q):
        #todo: add state cost
        # cost = np.random.uniform(0,1)
        cost = self.collision_fn(q)
        return cost

    def compute_controll_cost(self, Q):
        control_cost = np.linalg.norm(np.matmul(self.A, Q), ord=2)
        return control_cost



def plot_rollouts(rollouts, steps):
    for r in rollouts:
        plt.plot(r[:,0], r[:,1], 'r-')

def plot_traj(last_traj, steps):
    plt.plot(last_traj[:,0], last_traj[:,1], 'b-')
    plt.show()



