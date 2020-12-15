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

	csv_list = ["success_rate"]
	# column_list =["train/success_rate_rnn", "train/success_rate_mlp", "train/success_rate_1127_mlp", "train/success_rate_joint_space"]
	column_list = ["train/success_rate_rnn", "train/success_rate_mlp"]
	step = 'epoch'
	color = ['red','green','blue','orange']


	df_c = reading(csv_list, step, column_list, 0.5)


	fig, ax = plt.subplots()

	win = 10

	for i, df in enumerate(df_c):
		rews = df["value"].tolist()
		rew_list = get_plot_list(rews, win)
		print(type(rew_list))
		print(rew_list)
		print(np.shape(rew_list))
		sns.tsplot(time=df[step].tolist()[:-win], data=rew_list, color=color[i], linestyle='-', condition=column_list[i])
	

	# for i, df in enumerate(df_c):
	# 	ax.plot(df['Step'], df['smoothvalue'], '-', linewidth=2, color = color[i], label=df['mdoelname'][0])
	# 	ax.plot(df['Step'], df['Value'], '-', linewidth=4, color = color[i], alpha=.2, label="")
	# ax.tick_params(axis='both', which='major', labelsize=16)

	ax.set_xlabel('Steps', fontsize=20)
	ax.set_ylabel('Epoch task reward', fontsize=20)


	plt.tight_layout()
	plt.legend(fontsize =14)
	plt.grid()
	plt.show()

	# fig.savefig('eprew.pdf', dpi=1000)
	fig.savefig('success_rate.jpg', dpi=600)



def seq2seq(csvname):
	colnames=["Step", "Value"]
	color = ['orange','red','green','blue']
	model_name = ['Predictor_iter1','Predictor_iter2','Predictor_iter3']

	df = pd.read_csv(csvname+".csv")

	steps = df[colnames[0]].tolist()
	values = df[colnames[1]].tolist()

	df['Step']-=df['Step'][0]

	df['smoothvalue']=smoothing(values)

	fig, ax = plt.subplots()

	# for i in range(len(steps)//30):
	# 	end = 30*(i+1)
	# 	if i == 0:
	# 		sta = 0
	# 	else:
	# 		sta = 30*i-1
	# 	sns.tsplot(time=df["Step"][sta:], data=df['smoothvalue'][sta:], color=color[i], linestyle='-', condition=model_name[i])
		# ax.plot(df['Step'][sta:end], df['Value'][sta:end], '-', linewidth=2, color = color[i], alpha=.2, label="")

	for i in range(len(steps)//30):
		end = 30*(i+1)
		if i == 0:
			sta = 0
		else:
			sta = 30*i-1
		ax.plot(df['Step'][sta:end], df['smoothvalue'][sta:end], '-', linewidth=2, color = color[i], label=model_name[i])
		ax.plot(df['Step'][sta:end], df['Value'][sta:end], '-', linewidth=2, color = color[i], alpha=.2, label="")
	
	ax.tick_params(axis='both', which='major', labelsize=14)

	ax.set_xlabel('Steps', fontsize=24)
	ax.set_ylabel('Validation loss', fontsize=24)


	plt.tight_layout()
	plt.legend(fontsize =20)
	plt.grid()
	# plt.show()
	fig.savefig('valloss.jpg', dpi=600)


	# fig.savefig('val_loss.pdf', dpi=1000)




def main():

	read_eprew()
	# read_pred()
	# seq2seq('seq2seq_val_loss')


if __name__ == "__main__":
	main()
