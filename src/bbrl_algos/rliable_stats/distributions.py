import numpy as np
import matplotlib

font = {"family": "normal", "size": 70}
matplotlib.rc("font", **font)


distributions_list = ["normal", "lognormal", "bimod", "td3", "sac"]


def sample(perf1, perf2, distrib, size, shift=0.0, std_ratio="single", median=False):
    """
    Samples from a distribution among the list (normal, log-normal, bimodal, td3, sac)
    All distributions are centered in 0 (mean=0 or median=0) depending on the boolean median.

    :param distrib: (str) tells which distribution to sample from
    :param size: (int) sample size
    :param shift: (float) relative effect size, the distribution is shifted by shift * std_pooled
    :param std_ratio: (str) either 'single' for std=1 or 'double' for std=2.
        Note that sac and td3 std are not affected by this
    :param median: (bool) if true, the distribution median is 0, if false, the distribution mean is 0
    :return: a sample of size size.
    """

    if distrib == "normal":
        if std_ratio == "double":
            std_factor = 2
            pooled_std = np.sqrt((1**2 + 2**2) / 2)
        else:
            std_factor = 1
            pooled_std = 1
        return np.random.normal(loc=shift * pooled_std, scale=std_factor, size=size)

    elif distrib == "lognormal":
        if not median:
            if std_ratio == "single":
                loc = -1.2695
                std = 0.691
                pooled_std = 1
            else:
                loc = -1.601
                std = 0.9712
                pooled_std = np.sqrt((1**2 + 2**2) / 2)
        else:
            if std_ratio == "single":
                loc = -1
                std = 0.691
                pooled_std = 1
            else:
                loc = -1
                std = 0.9712
                pooled_std = np.sqrt((1**2 + 2**2) / 2)
        return (
            np.random.lognormal(mean=0, sigma=std, size=size) + loc + shift * pooled_std
        )
    elif distrib == "bimod":
        if std_ratio == "double":
            std_factor = 2.17
            pooled_std = np.sqrt((1**2 + 2**2) / 2)
        else:
            std_factor = 1
            pooled_std = 1

        out = []
        for _ in range(size):
            if np.random.random() < 0.5:
                out.append(
                    np.random.normal(
                        loc=-0.9 * std_factor + shift * pooled_std, scale=0.45
                    )
                )
            else:
                out.append(
                    np.random.normal(
                        loc=0.9 * std_factor + shift * pooled_std, scale=0.45
                    )
                )
        return np.array(out)

    elif distrib == "sac":
        pooled_std = np.sqrt((np.std(perf1) ** 2 + np.std(perf2) ** 2) / 2)
        inds = np.random.randint(0, perf1.shape[0], size=size)
        if median:
            align = np.median(perf1)
        else:
            align = perf1.mean()
        return np.array(perf1[inds]) - align + shift * pooled_std

    elif distrib == "td3":
        pooled_std = np.sqrt((np.std(perf1) ** 2 + np.std(perf2) ** 2) / 2)
        inds = np.random.randint(0, perf2.shape[0], size=size)
        if median:
            align = np.median(perf2)
        else:
            align = perf2.mean()
        return np.array(perf2[inds]) - align + shift * pooled_std
    else:
        raise NotImplementedError


def get_distribution_pairs(study, distributions_pair_idx):
    """
    Get str ids for distribution to compare in a given study. Set the std_ratio depending on the study.
    :param study: (str) describes the current study
    :param distributions_pair_idx: (list of tuples) each element is a tuple describing the index of the two
        distributions to compare.
    :return: list of tuples. Each tuple is of size two, contains the two string ids of two distributions
        to compare.
    """
    if study == "equal_dist_equal_var":
        distrib_list = []
        for distrib in distributions_list:
            distrib_list.append((distrib, distrib))
        std_ratio = ("single", "single")
    elif study == "equal_dist_unequal_var":
        distrib_list = []
        for distrib in distributions_list:
            distrib_list.append((distrib, distrib))
        std_ratio = ("single", "double")
    elif study == "unequal_dist_equal_var":
        distrib_idx = distributions_pair_idx
        distrib_list = []
        for idx in distrib_idx:
            distrib1 = distributions_list[idx[0]]
            distrib2 = distributions_list[idx[1]]
            distrib_list.append((distrib1, distrib2))
        std_ratio = ("single", "single")
    elif study == "unequal_dist_unequal_var_1":
        distrib_idx = distributions_pair_idx
        distrib_list = []
        for idx in distrib_idx:
            distrib1 = distributions_list[idx[0]]
            distrib2 = distributions_list[idx[1]]
            distrib_list.append((distrib1, distrib2))
        std_ratio = ("single", "double")
    elif study == "unequal_dist_unequal_var_2":
        distrib_idx = distributions_pair_idx
        distrib_list = []
        for idx in distrib_idx:
            distrib1 = distributions_list[idx[0]]
            distrib2 = distributions_list[idx[1]]
            distrib_list.append((distrib1, distrib2))
        std_ratio = ("double", "single")
    else:
        raise NotImplementedError

    return distrib_list, std_ratio
