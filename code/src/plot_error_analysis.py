import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import numpy as np


TITLE_FONTSIZE = 20
AXES_LAB_FONTSIZE = 16
AXES_TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
MARKERS_FONTSIZE = 10


def main():
    error_stats_by_task = {
        "listops": build_error_stats_df("listops"),
        "arithmetic": build_error_stats_df("arithmetic"),
        "algebra": build_error_stats_df("algebra"),
    }
    plot_error_analysis(error_stats_by_task)


def build_error_stats_df(task_name):
    dfs = []
    for n_multi in [10, 100, 1000]:
        df = pd.read_csv(f'../out/error_analysis/{task_name}_solve_{n_multi}_error_analysis.csv', index_col=0)
        df = df.rename(columns={'sub_expression': 'selector/missing', 'parentheses': 'selector/corrupted'})
        df = df.rename(lambda x: x.replace('O','A').replace('_', '\n'), axis=0)
        dfs.append(df)
    dfs.append(pd.read_csv(f'../out/error_analysis/{task_name}_solve_1000_window_error_analysis.csv', index_col=0))    
    return dfs


def plot_error_analysis(stats_dfs):
    fig = plt.figure(layout="constrained", figsize=(10,9))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2, ax3]

    spades = mlines.Line2D([], [], color='gray', marker='$♠$', linestyle='None',
                            markersize=MARKERS_FONTSIZE, label='10')
    clubs = mlines.Line2D([], [], color='gray', marker='$♣$', linestyle='None',
                            markersize=MARKERS_FONTSIZE, label='100')
    diamonds = mlines.Line2D([], [], color='gray', marker='$♦$', linestyle='None',
                            markersize=MARKERS_FONTSIZE, label='1000')
    hearts = mlines.Line2D([], [], color='gray', marker='$♥$', linestyle='None',
                            markersize=MARKERS_FONTSIZE, label='1000+DW')

    for ax, (task_name, stats_df) in zip(axes, stats_dfs.items()):
        ax = plot_clustered_stacked(stats_df, ['10', '100', '1000'], title=task_name.capitalize(), H='/', ax=ax, handles=[spades, clubs, diamonds, hearts])
        ax.set_xticks(range(len(stats_df[0].index)), stats_df[0].index, rotation=0, fontsize=AXES_TICK_FONTSIZE)
        ax.set_xlim(-0.6,len(stats_df[0].index))
        ax.set_ylabel('% Errors', fontsize=AXES_LAB_FONTSIZE)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)
    
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax3.set_ylim(0, 1.1)
    ax1.set_xlabel('Complexity', fontsize=AXES_LAB_FONTSIZE)
    ax2.set_xlabel('Complexity', fontsize=AXES_LAB_FONTSIZE)
    ax3.set_xlabel('Complexity', fontsize=AXES_LAB_FONTSIZE)

    ax3.set_ylabel('')
    ax3.set_yticks([], [])

    plt.tight_layout()
    plt.savefig('../out/plots/errors_analysis.pdf', bbox_inches='tight')


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", ax=None, handles=None, **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    if ax is None:
        ax = plt.subplot(111)

    for df in dfall : # for each data frame
        ax = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=ax,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots
    texts=[]
    h,l = ax.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        heights = [0]*n_ind
        for j, pa in enumerate(h[i:i+n_col]):
            for rect_idx, rect in enumerate(pa.patches): # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                # rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))
                heights[rect_idx] += rect.get_height()
                if j == 2:
                    texts.append(ax.text(rect.get_x() + rect.get_width() / 2.0, heights[rect_idx], '♠♣♦♥'[i//3], ha='center', va='bottom', size=MARKERS_FONTSIZE, color='gray'))
    ax.set_title(format_task_name(title), fontsize=TITLE_FONTSIZE)

    if title == 'Listops':
        l1 = ax.legend(handles + h[:n_col], ['10','100','1000','1000+DW'] + l[:n_col], loc='best', fontsize=LEGEND_FONTSIZE, ncol=2, handleheight=2.0, labelspacing=0.1)
        ax.add_artist(l1)

    return ax


def format_task_name(task_name):
    if task_name == 'Listops':
        return 'ListOps'
    else:
        return task_name


if __name__ == '__main__':
    main()
