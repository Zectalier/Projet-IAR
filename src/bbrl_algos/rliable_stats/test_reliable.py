import numpy as np

# See warnings only once
import warnings


import logging

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import seaborn as sns

# Matplotlib params
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import matplotlib.patches as mpatches

warnings.filterwarnings("default")

sns.set_style("white")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

rc("text", usetex=False)

# from scipy.stats import find_repeats

colors = sns.color_palette("colorblind")
xlabels = ["dqn", "ppo"]
color_idxs = [0, 3, 4, 2, 1, 7, 8]
color_dict = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))


def run_aggregates(data_file1, data_file2):
    data1 = np.expand_dims(np.array(np.loadtxt(data_file1)), axis=1)
    data2 = np.expand_dims(np.array(np.loadtxt(data_file2)), axis=1)
    score_dict = {"dqn": data1, "ppo": data2}
    algorithms = ["dqn", "ppo"]

    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=50000
    )

    # fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_interval_estimates,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=algorithms,
        colors=color_dict,
        # xlabel_y_coordinate=-0.16,
        xlabel="Score",
    )
    plt.savefig("dqn_vs_ppo_aggregates.pdf")
    plt.show()
    plt.close()


def run_perf_profiles(data_file1, data_file2):
    data1 = np.expand_dims(np.array(np.loadtxt(data_file1)), axis=1)
    data2 = np.expand_dims(np.array(np.loadtxt(data_file2)), axis=1)

    min_value1 = np.min(data1)
    min_value2 = np.min(data2)
    min_value = min(min_value1, min_value2)

    max_value1 = np.max(data1)
    max_value2 = np.max(data2)
    max_value = max(max_value1, max_value2)

    score_dict = {"dqn": data1, "ppo": data2}

    tau = np.linspace(min_value, max_value, 201)
    # Higher value of reps corresponds to more accurate estimates but are slower
    # to compute. `reps` corresponds to number of bootstrap resamples.
    reps = 2000

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, tau, reps=reps
    )

    fig, ax = plt.subplots(ncols=1, figsize=(8, 7))

    plot_utils.plot_performance_profiles(
        score_distributions,
        tau,
        performance_profile_cis=score_distributions_cis,
        colors=color_dict,
        xlabel=r"Score $(\tau)$",
        labelsize="xx-large",
        ax=ax,
    )

    ax.axhline(0.5, ls="--", color="k", alpha=0.4)
    plt.savefig("dqn_vs_ppo_perf_profiles.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    data_file1 = "./data_files/sac_hc_final_perfs.txt"
    data_file2 = "./data_files/td3_hc_final_perfs.txt"

    # data_file1 = './data_files/sac_hc_all_perfs.txt'
    # data_file2 = './data_files/td3_hc_all_perfs.txt'

    run_aggregates(data_file1, data_file2)
    run_perf_profiles(data_file1, data_file2)
