import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


TITLE_FONTSIZE = 20
AXES_LAB_FONTSIZE = 16
AXES_TICK_FONTSIZE = 14


def main():
	confidence_scores_by_task = {
		"listops": load_conf_score_df("listops"),
		"arithmetic": load_conf_score_df("arithmetic"),
		"algebra": load_conf_score_df("algebra"),
	}

	plot_conf_scores(confidence_scores_by_task)


def load_conf_score_df(task_name):
	return pd.read_csv(f'../out/in_len_v_conf_scores/{task_name}_solve_1000_input_len_Vs_conf_scores.csv')


def plot_conf_scores(dfs):
	fig, axes = plt.subplots(3, 1, figsize=(10, 10))#, sharex=True, sharey=True)
	max_len_input_per_task = get_max_len_input_per_task()

	for ax, (task_name, df) in zip(axes.flat, dfs.items()):
		ax = sns.scatterplot(
				data=df,
				x='input_len',
				y='avg_conf_score',
				ax=ax,
				linewidth=0,
				edgecolors='face',
			)
		ax.axvline(max_len_input_per_task[task_name], color='gray', linestyle='--')
		ax.set_title(format_task_name(task_name.capitalize()), fontsize=TITLE_FONTSIZE)
		ax.set_xlabel('')
		ax.set_ylabel('Avg Confidence Score', fontsize=AXES_LAB_FONTSIZE)
		ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
		ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

	axes[-1].set_xlabel('Input Lenght', fontsize=AXES_LAB_FONTSIZE)
	plt.tight_layout()
	plt.savefig('../out/plots/input_len_v_conf_scores.pdf', bbox_inches='tight')


def get_max_len_input_per_task():
	return {
		"listops": pd.read_csv('../datasets/listops_controlled_select/train.csv')['X'].apply(len).max(),
		"arithmetic": pd.read_csv('../datasets/arithmetic_controlled_select/train.csv')['X'].apply(len).max(),
		"algebra": pd.read_csv('../datasets/algebra_controlled_select/train.csv')['X'].apply(len).max(),
	}

def format_task_name(task_name):
	if task_name == 'Listops':
		return 'ListOps'
	else:
		return task_name

if __name__ == '__main__':
	main()
