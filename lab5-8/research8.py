import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def ci_mean_t(data, alpha):
    n = len(data)
    m = np.mean(data)
    s = np.std(data)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    d = s * t / np.sqrt(n - 1)
    return m - d, m + d


def ci_std_t(data, alpha):
    n = len(data)
    s = np.std(data)
    return s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1)), s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(alpha / 2, n - 1))


def ci_mean_asymp(data, alpha):
    n = len(data)
    m = np.mean(data)
    s = np.std(data)
    u = stats.norm.ppf(1 - alpha / 2)
    d = s * u / np.sqrt(n)
    return m - d, m + d


def ci_std_asymp(data, alpha):
    n = len(data)
    s = np.std(data)
    u = stats.norm.ppf(1 - alpha / 2)
    m4 = stats.moment(data, 4)
    e = m4 / s ** 4 - 3
    U = u * np.sqrt((e + 2) / n)
    return s / np.sqrt(1 + U), s / np.sqrt(1 - U)


def lab8():
    data20 = np.random.standard_normal(20)
    data100 = np.random.standard_normal(100)

    t_mean_20 = ci_mean_t(data20, 0.05)
    t_mean_100 = ci_mean_t(data100, 0.05)
    t_std_20 = ci_std_t(data20, 0.05)
    t_std_100 = ci_std_t(data100, 0.05)

    asymp_mean_20 = ci_mean_asymp(data20, 0.05)
    asymp_mean_100 = ci_mean_asymp(data100, 0.05)
    asymp_std_20 = ci_std_asymp(data20, 0.05)
    asymp_std_100 = ci_std_asymp(data100, 0.05)

    print(t_mean_20, t_std_20)
    print(t_mean_100, t_std_100)
    print(asymp_mean_20, asymp_std_20)
    print(asymp_mean_100, asymp_std_100)

    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace = 0.5)
    ax[0].set_ylim([0,1])
    ax[0].hist(data20, 10, density = 1, edgecolor = 'black')

    max_mu20 = max(t_mean_20[1], asymp_mean_20[1])
    min_mu20 = min(t_mean_20[0], asymp_mean_20[0])
    max_sigma20 = max(t_std_20[1], asymp_std_20[1])

    ax[0].axvline(x=min_mu20, label='max\u03BC ', color='green')
    ax[0].axvline(x=max_mu20, label='min\u03BC', color='red')
    ax[0].axvline(x=min_mu20 - max_sigma20, label='min\u03BC - max\u03C3', color='black')
    ax[0].axvline(x=max_mu20 + max_sigma20,label= 'max\u03BC + max\u03C3', color='orange')
    ax[0].legend()
    ax[0].set_title('N(0,1) hist, n = 20')

    max_mu100 = max(t_mean_100[1], asymp_mean_100[1])
    min_mu100 = min(t_mean_100[0], asymp_mean_100[0])
    max_sigma100 = max(t_std_100[1], asymp_std_100[1])

    ax[1].set_ylim([0,1])
    ax[1].hist(data100, 10, density = 1, edgecolor = 'black')
    ax[1].axvline(x=min_mu100, label='max\u03BC ', color='green')
    ax[1].axvline(x=max_mu100, label='min\u03BC', color='red')
    ax[1].axvline(x=min_mu100 - max_sigma100, label='min\u03BC - max\u03C3', color='black')
    ax[1].axvline(x=max_mu100 + max_sigma100,label= 'max\u03BC + max\u03C3', color='orange')
    ax[1].legend()
    ax[1].set_title('N(0,1) hist, n = 100')

    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.2) 

    ax[0][0].plot([q for q in t_mean_20], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[0][0].plot([q for q in t_mean_100], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[0][0].set_ylim([0,1])
    ax[0][0].set_title('Classic approach')
    ax[0][0].legend()

    ax[0][1].plot([q for q in t_std_20], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[0][1].plot([q for q in t_std_100], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[0][1].set_ylim([0,1])
    ax[0][1].set_title('Classic approach')
    ax[0][1].legend()

    ax[1][0].plot([q for q in asymp_mean_20], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'm interval, n = 20')
    ax[1][0].plot([q for q in asymp_mean_100], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'm interval, n = 100')
    ax[1][0].set_ylim([0,1])
    ax[1][0].set_title('Asymptotic approach')
    ax[1][0].legend()

    ax[1][1].plot([q for q in asymp_std_20], [0.3, 0.3], color='r', marker = '.', linewidth = 1, label = 'sigma interval, n = 20')
    ax[1][1].plot([q for q in asymp_std_100], [0.6, 0.6], color='blue', marker = '.', linewidth = 1, label = 'sigma interval, n = 100')
    ax[1][1].set_ylim([0,1])
    ax[1][1].set_title('Asymptotic approach')
    ax[1][1].legend()

    plt.show()


if __name__ == "__main__":
    lab8()