import numpy as np
from stomp import STOMP



if __name__ == "__main__":
    D = 2
    N = 40
    K = 10
    stomp_planner = STOMP(D, N, K)

    stomp_planner.plan([-10,-10],[10,10])


