import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ANNOT_FONTSIZE = 25
AXES_LAB_FONTSIZE = 20
AXES_TICK_FONTSIZE = 17
TITLE_FONTSIZE = 27


def main():

    accuracy_tables_by_task = {
        "listops": {
            "ndr": load_table_ndr('listops'),
            "gpt": load_table_gpt('listops'),
            "ours": load_table_ours('listops'),
            "ours_w": load_table_ours('listops', window=True)
        },
        "arithmetic": {
            "ndr": load_table_ndr('arithmetic'),
            "gpt": load_table_gpt('arithmetic'),
            "ours": load_table_ours('arithmetic'),
            "ours_w": load_table_ours('arithmetic', window=True)
        },
        "algebra": {
            "ndr": load_table_ndr('algebra'),
            "gpt": load_table_gpt('algebra'),
            "ours": load_table_ours('algebra'),
            "ours_w": load_table_ours('algebra', window=True)
        },
    }
    
    plot_accuracy_tables_listops(accuracy_tables_by_task['listops'])
    plot_accuracy_tables_arit_alg(accuracy_tables_by_task['arithmetic'], 'arithmetic')
    plot_accuracy_tables_arit_alg(accuracy_tables_by_task['algebra'], 'algebra')

    accuracy_tables_zscot = {
        'listops': load_table_zero_shot_cot('listops'),
        'arithmetic': load_table_zero_shot_cot('arithmetic'),
        'algebra': load_table_zero_shot_cot('algebra'),
    }

    plot_accuracy_tables_zero_shot_cot(accuracy_tables_zscot)


def load_table_ndr(task_name):
    df = pd.read_csv(f'../out/ndr_accuracy_tables/{task_name}.csv', index_col=0)
    return revert_rows_order(reformat_floats(df))

def load_table_gpt(task_name):
    df = pd.read_csv(f'../gpt/output/accuracy_tables/gpt4_{task_name}_self_consistency.csv', index_col=0)
    return revert_rows_order(reformat_floats(df.dropna(axis=1)))

def load_table_zero_shot_cot(task_name):
    df = pd.read_csv(f'../gpt/output/accuracy_tables/gpt4_{task_name}_zero_shot_cot.csv', index_col=0)
    return revert_rows_order(reformat_floats(df.dropna(axis=1)))

def load_table_ours(task_name, window=False):
    if window:
        df = pd.read_csv(f'../out/ours_accuracy_tables/window/{task_name}.csv', index_col=0)
        return revert_rows_order(reformat_floats(df))
    else:
        df = pd.read_csv(f'../out/ours_accuracy_tables/no_window/{task_name}.csv', index_col=0)
        return revert_rows_order(reformat_floats(df))

def reformat_floats(df):
    return df.astype(str).map(lambda x: x.replace(',', '.')).astype(float)

def revert_rows_order(df):
    return df.iloc[::-1]

def plot_accuracy_tables_listops(accuracy_tables):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)
    
    for (model_name, table), ax in zip(accuracy_tables.items(), axes.flat):
        ax = sns.heatmap(table.iloc[::-1,::-1].T, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': ANNOT_FONTSIZE}, cbar=False, square=True)
        ax.set_title(format_model_name(model_name), fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    axes[1, 0].set_xlabel('nesting', fontsize=AXES_LAB_FONTSIZE)
    axes[1, 1].set_xlabel('nesting', fontsize=AXES_LAB_FONTSIZE)
    axes[0, 0].set_ylabel('arguments', fontsize=AXES_LAB_FONTSIZE)
    axes[1, 0].set_ylabel('arguments', fontsize=AXES_LAB_FONTSIZE)
    axes[1, 1].set_title('NRS\n(Dynamic Windowing)', fontsize=TITLE_FONTSIZE)
    
    plt.savefig('../out/plots/accuracy_tables_listops.pdf', bbox_inches='tight')

def plot_accuracy_tables_arit_alg(accuracy_tables, task_name):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    for (model_name, table), ax in zip(accuracy_tables.items(), axes.flat):
        ax = sns.heatmap(table.iloc[::-1].T, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': ANNOT_FONTSIZE}, cbar=False, square=True)
        ax.set_title(format_model_name(model_name), fontsize=TITLE_FONTSIZE)
        ax.set_ylabel('arguments', fontsize=AXES_LAB_FONTSIZE)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    axes[-1].set_xlabel('nesting', fontsize=AXES_LAB_FONTSIZE)
    
    plt.savefig(f'../out/plots/accuracy_tables_{task_name}.pdf', bbox_inches='tight')


def plot_accuracy_tables_zero_shot_cot(accuracy_tables_by_task):
    base_fontsize = 8
    
    fig = plt.figure(layout="constrained", figsize=(4, 2.2))
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    axes = [ax1, ax2, ax3]

    for task_name, ax in zip(['listops', 'arithmetic', 'algebra'], axes):
        table = accuracy_tables_by_task[task_name]
        if task_name == 'listops':
            ax = sns.heatmap(table.iloc[::-1,::-1].T, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': base_fontsize-2}, cbar=False, square=True)
        else:
            ax = sns.heatmap(table.iloc[::-1].T, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': base_fontsize-2}, cbar=False, square=True)
        
        ax.set_title(format_task_name(task_name.capitalize()), fontsize=base_fontsize)
        ax.tick_params(axis="x", labelsize=base_fontsize-4)
        ax.tick_params(axis="y", labelsize=base_fontsize-4)
        ax.set_ylabel('arguments', fontsize=base_fontsize-3)

    axes[0].set_xlabel('nesting', fontsize=base_fontsize-3)
    axes[-1].set_xlabel('nesting', fontsize=base_fontsize-3)

    plt.savefig(f'../out/plots/accuracy_tables_zscot.pdf', bbox_inches='tight')


def format_model_name(model_name):
    return {
        'ndr': 'NDR',
        'gpt': 'GPT-4',
        'ours': 'NRS',
        'ours_w': 'NRS (Dynamic Windowing)'
    }[model_name]


def format_task_name(task_name):
    if task_name == 'Listops':
        return 'ListOps'
    else:
        return task_name


if __name__ == '__main__':
    main()