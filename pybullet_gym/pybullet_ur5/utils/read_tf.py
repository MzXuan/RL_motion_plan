import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc('font', family='Times New Roman')

'''
process and draw tfboard result
'''


def mean_arr(arr, win=10):
	arr_len = len(arr)
	new_arr = np.zeros_like(arr)
	for i in range(arr_len - win):
		new_arr[i] = np.mean(arr[i:i + win])
	return new_arr


def max_arr(arr, win=10):
	arr_len = len(arr)
	new_arr = np.zeros_like(arr)
	for i in range(arr_len - win):
		if i == 0:
			new_arr[i] == arr[i]
			continue

		if np.mean(arr[i:i + win]) < arr[i]:
			new_arr[i] = arr[i] + np.std(arr[i:i + win])
		else:
			new_arr[i] = new_arr[i - 1]

	return new_arr


def min_arr(arr, win=10):
	arr_len = len(arr)
	new_arr = np.zeros_like(arr)
	for i in range(arr_len - win):
		if i == 0:
			new_arr[i] == arr[i]
			continue

		if np.mean(arr[i:i + win]) > arr[i]:
			new_arr[i] = arr[i] - np.std(arr[i:i + win])
		else:
			new_arr[i] = new_arr[i - 1]

	return new_arr


def get_plot_list(arr, win=10):
	means, mins, maxs = mean_arr(arr, win), min_arr(arr, win), max_arr(arr, win)
	arr_list = [means[:-win], mins[:-win], maxs[:-win]]
	return arr_list




#----------------------------smoothing and drawing----------------#
def smoothing(dataset, smoothingWeight=0.8):
	set_smoothing =[]
	for idx, d in enumerate(dataset):
		if idx==0:
			last = d
		else:
			d_smoothed = last * smoothingWeight + (1 - smoothingWeight) * d;
			last=d_smoothed
		set_smoothing.append(last)
	return set_smoothing

def reading(csv_list, step, column_list, smoothingWeight=0.8):
	df_list =[]

	# step="epoch"
	# colnames = ["epoch", "train/success_rate_rnn"]
	
	for f in csv_list:

		for c in column_list:
			df = pd.read_csv(f + ".csv")
			values = df[c].tolist()
			df[step]-=df[step][0]
			df['value'] = values
			df['smoothvalue']=smoothing(values,smoothingWeight)
			df['mdoelname']=f

			df_list.append(df)

	return df_list


def read_eprew():

	csv_list = ["success_rate_1220"]
	# column_list =["train/success_rate_rnn", "train/success_rate_mlp", "train/success_rate_1127_mlp", "train/success_rate_joint_space"]
	column_list = ["success_rate_GRU+MLP", "success_rate_MLP_2steps", "success_rate_MLP_6steps"]
	step = 'train/episode'
	color = ['red','green','blue','orange']


	df_c = reading(csv_list, step, column_list, 0.5)


	fig, ax = plt.subplots()

	win = 8

	for i, df in enumerate(df_c):
		rews = df["value"].tolist()
		rew_list = get_plot_list(rews, win)
		print(type(rew_list))
		print(rew_list)
		print(np.shape(rew_list))
		sns.tsplot(time=df[step].tolist()[:-win], data=rew_list, color=color[i], linestyle='-', condition=column_list[i])

	ax.set_xlabel('Episode', fontsize=20)
	ax.set_ylabel('Episode success rate', fontsize=20)


	plt.tight_layout()
	plt.legend(fontsize =14)
	plt.grid()
	plt.show()

	# fig.savefig('eprew.pdf', dpi=1000)
	fig.savefig('success_rate.jpg', dpi=600)


def read_loss():
	csv_list = ["qc_progress"]
	# column_list =["train/success_rate_rnn", "train/success_rate_mlp", "train/success_rate_1127_mlp", "train/success_rate_joint_space"]
	column_list = ["stats_Qc_loss/mean"]
	step = 'train/episode'
	color = ['blue', 'orange']

	df_c = reading(csv_list, step, column_list, 0.0)

	fig, ax = plt.subplots()

	win = 1

	for i, df in enumerate(df_c):
		rews = df["value"].tolist()
		rew_list = get_plot_list(rews, win)
		print(type(rew_list))
		print(rew_list)
		print(np.shape(rew_list))
		sns.lineplot(data=df, x='train/episode', y=column_list[i],linewidth=4,color=color[i])

	ax.set_xlabel('Episode', fontsize=20)
	ax.set_ylabel('Episode loss of collision value ', fontsize=20)

	plt.tight_layout()
	plt.legend(fontsize=14)
	plt.grid()
	plt.show()

	# fig.savefig('eprew.pdf', dpi=1000)
	fig.savefig('Qc_loss.jpg', dpi=600)


def read_path_error():
	csv_list = ["tracking error - end-effector_nohuman"]
	# csv_list = ["tracking error - end-effector_human"]
	# column_list = ["dynamic_goal_no_human", "dynamic_goal_human_nearby", "final_goal_no_human"]

	column_list = ["Our method","Previous RL based method","Previous RL based method +dymanic goal","ITOMP with STOMP optimizer"]

	step = 'step'
	color = ['red', 'lightblue','darkblue','green']
	line = ['-', '--', '--', '-.']
	df_c = reading(csv_list, step, column_list, 0.2)
	fig, ax = plt.subplots()

	for i, df in enumerate(df_c):
		steps=np.asarray(range(100))

		y_raw = df[column_list[i]].values
		y_raw = y_raw[~np.isnan(y_raw)]
		x_raw = np.linspace(0,100,len(y_raw))

		y = np.interp(steps, x_raw, y_raw)
		plt.plot(steps, y, color=color[i], label=column_list[i], linestyle=line[i], linewidth=3)

	ax.set_xlabel('Percent', fontsize=18)
	ax.set_ylabel('End-effector tracking error / m', fontsize=20)

	plt.tight_layout()
	plt.legend(fontsize=12, loc=1)
	plt.grid()
	plt.show()

	fig.savefig('eef_error.jpg', dpi=1000)

def read_j_path_error():
	csv_list = ["tracking error - joint_no_human"]
	# csv_list = ["tracking error - joint_human"]
	# column_list = ["dynamic_goal_no_human", "dynamic_goal_human_nearby", "final_goal_no_human"]

	column_list = ["Our method", "Previous RL based method", "Previous RL based method +dymanic goal",
				   "ITOMP with STOMP optimizer"]
	step = 'step'
	color = ['red', 'lightblue','darkblue','green']
	line = ['-','--','--','-.']
	df_c = reading(csv_list, step, column_list, 0.2)
	fig, ax = plt.subplots()


	for i, df in enumerate(df_c):
		steps=np.asarray(range(100))
		y_raw = df[column_list[i]].values
		y_raw = y_raw[~np.isnan(y_raw)]
		x_raw = np.linspace(0, 100, len(y_raw))

		y = np.interp(steps, x_raw, y_raw)
		plt.plot(steps, y, color=color[i], label=column_list[i], linestyle=line[i], linewidth=3)

	ax.set_xlabel('Percent', fontsize=18)
	ax.set_ylabel('Joint tracking error', fontsize=20)


	# for i, df in enumerate(df_c):
	# 	plt.plot(df['step'], df[column_list[i]], color=color[i], label=column_list[i], linestyle=line[i], linewidth=3)
	#
	# ax.set_xlabel('Step', fontsize=18)
	# ax.set_ylabel('Joint tracking error', fontsize=20)

	plt.tight_layout()
	plt.legend(fontsize=12, loc=1)
	plt.grid()
	plt.show()

	# fig.savefig('eprew.pdf', dpi=1000)
	fig.savefig('joint_error.jpg', dpi=1000)




def main():

	# read_eprew()
	# read_loss()
	read_path_error()
	read_j_path_error()



if __name__ == "__main__":
	main()
