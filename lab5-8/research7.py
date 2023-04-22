import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def chi_table(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(f'mu={mu}, sigma={sigma}')

    k = int(np.floor(1.72 * len(data)**(1/3)))
    borders = np.linspace(np.floor(np.min(data)), np.ceil(np.max(data)), k-1)
    borders = np.insert(borders, 0, -np.inf)
    borders = np.append(borders, np.inf)
    
    table = []
    table.append(['\hline i', 'Границы $\Delta_i$', '$n_i$', '$p_i$', 
                  '$np_i$', '$n_i - np_i$', '$\\frac{(n_i - np_i)^2}{np_i}$'])
    
    ns = []
    ps = []
    nps = []
    n_sub_nps = []
    ress = []
   
    for i in range(len(borders) - 1):
        left = borders[i]
        right = borders[i + 1]
        
        n = ((left < data) & (data <= right)).sum()
        ns.append(n)
        
        p = stats.norm.cdf(right) - stats.norm.cdf(left)
        ps.append(p)
        
        np_ =  len(data) * p
        nps.append(np_)
        
        n_sub_np = n - np_
        n_sub_nps.append(n_sub_np)
        
        res = n_sub_np ** 2 / np_
        ress.append(res)
        
        table.append([i + 1, f'({round(left, 2)}, {round(right, 2)}]', 
                      round(n, 2), round(p, 2), round(np_, 2), round(n_sub_np, 2), round(res, 2)])
    table.append(['$\sum$', '-', sum(ns), sum(ps), round(sum(nps)), round(sum(n_sub_nps)), round(sum(ress), 2)])
    return table


def write_table(path, table):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        for row in table:
            f.write(" & ".join([str(i) for i in row]) + "\\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}")


def lab7():
    data = np.random.standard_normal(100)
    write_table('lab5-8/task3_data/task3_normal.tex', chi_table(np.random.standard_normal(100)))
    write_table('lab5-8/task3_data/task3_laplace.tex', chi_table(np.random.laplace(0, 1 / np.sqrt(2), 20)))


if __name__ == "__main__":
    lab7()