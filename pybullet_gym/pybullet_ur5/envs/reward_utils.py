import os, sys, glob, gc, joblib
import numpy as np
import math


def point_distance(point1,point2):
	return np.square(point1-point2).mean()

def cost_func(d0, di):
	l = np.arctan(20*(di-d0))
	return l


def distance(point, goals):
	d = [np.linalg.norm(point - g) for g in goals]
	return d


def delta_distance(poins_list,goals):
	delta_d0 = np.diff([np.linalg.norm(p - goals[0]) for p in poins_list])
	delta_d1 = np.diff([np.linalg.norm(p - goals[1]) for p in poins_list])
	delta_d2 = np.diff([np.linalg.norm(p - goals[2]) for p in poins_list])
	delta_d = [[d0, d1, d2] for d0, d1, d2 in zip(delta_d0,delta_d1,delta_d2)]
	return delta_d


def fast_decrease_distance_reward(obs, infos, steps):
	delta_distance = obs[20:]
	dim = obs[20:].shape
	length = int(dim[0]/3)
	delta_distance = np.reshape(obs[20:],(length, 3))

	#remove zero element
	for idx, a in enumerate(delta_distance):
		if not a.any(): # all zeros
			delta_distance=delta_distance[:idx]
			break

	delta_d0 = delta_distance[:,0].mean()
	delta_d1 = delta_distance[:,1].mean()
	delta_d2 = delta_distance[:,2].mean()

	return np.exp(-steps/30)*min([delta_d1-delta_d0, delta_d2-delta_d0])

	# if delta_d0 < delta_d1 and delta_d0<delta_d2:
	# 	return 1
	# else:
	# 	return 0




# def fast_decrease_distance_reward(obs_positions, infos):
#
#
# 	goals = np.concatenate(([infos['goal']], infos['alternative_goals']), axis=0)
# 	state_obs_lst = np.reshape(obs_positions[0:39],(3,13))
# 	points_list = state_obs_lst[:,0:3]
#
# 	for idx, a in enumerate(points_list):
# 		if not a.any(): # all zeros
# 			points_list=points_list[:idx]
# 			break
#
# 	delta_d0, delta_d1, delta_d2= delta_distance(points_list, goals)
# 	if delta_d0 < delta_d1 and delta_d0<delta_d2:
# 		return 1
# 	else:
# 		return 0

	# c1 = cost_func(delta_d0,delta_d1)
	# c2 = cost_func(delta_d0,delta_d2)
	# min_reward = min([c1, c2])
	#
	# print("delta d0 {} d1 {} d2 {} and c1 {} c2 {}".format(delta_d0, delta_d1, delta_d2, c1,c2))
	# print("task reward:",min_reward)
	# return min_reward


def point_goal_reward(batched_seqs, x_starts, batched_goals, batch_alternative_goals):
	'''
	:param point:
	:param goals:
	:return:
	'''
	rewards = []
	n_envs = len(batched_seqs)
	for idx in range(0, n_envs):
		seq = batched_seqs[idx]
		total_length = len(seq)
		if total_length < 1:
			rewards.append(0.0)
		else:
			dis_list = []
			true_goal = batched_goals[idx] - x_starts[idx][-3:]
			alternative_goals = batch_alternative_goals[idx].reshape((3, 3)) - x_starts[idx][-3:]
			point = seq[-1, -3:]
			d0 = np.linalg.norm(point - true_goal)

			for g in alternative_goals:
				if np.linalg.norm(g - true_goal) < 1e-7:
					pass
				else:
					dis_list.append(np.linalg.norm(point - g))
			rew = reward_dist(d0, dis_list, total_length)
			rewards.append(rew)
	return np.asarray(rewards)


def path_goal_reward(path, alternative_goals, g0, total_length):
	dis_list = []
	idx_list =[]
	idx_0 = np.linalg.norm((path - g0), axis=1).argmin()
	idx_list.append(idx_0)
	for g in alternative_goals:
		if np.linalg.norm(g - g0) < 1e-7:
			pass
		else:
			idx_i = np.linalg.norm((path - g), axis=1).argmin()
			idx_list.append((idx_i))
	idx_min = np.asarray(idx_list).argmin()  # find closest point in predlicted trajectory
	# print("the id list is {} abd the minimum idx is: {} ".format(idx_list, idx_min))

	point = path[idx_min]
	d0 = np.linalg.norm(point - g0)
	for g in alternative_goals:
		if np.linalg.norm(g - g0) < 1e-7:
			pass
		else:
			dis_list.append(np.linalg.norm(point - g))
	return(reward_dist(d0, dis_list,total_length))


def reward_dist(d0, dis, t):
	rew = []
	for d in dis:
		theta = 1 if d0 < d else -1
		rew.append(theta*math.log(abs(d0-d)/abs(d0+1)+1))
		# rew.append(theta*math.log(abs(d0-d)/abs(d0+d)+1))
	# print("distance is {} and reward list is: {} ".format(dist, rew))
	min_rew = np.asarray(rew).min()
	time_scale = math.exp(-t / 30)
	# time_scale = 1
	return (time_scale*min_rew)


def GetRandomGoal(dims):
	goals = []
	limite = [-3, 3]
	max_min = limite[1] - limite[0]
	min = limite[0]
	for _ in range(5):
		goals.append(np.random.random(dims)*max_min+min)
	return goals


if __name__ == '__main__':
	GetRandomGoal(3)

	#    path = np.random.rand(10,3)
	# goals = []
	# for _ in range(5):
	# 	goals.append(np.random.rand(3))

	# goal_idx = find_goal(path, goals)