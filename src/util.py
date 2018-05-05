from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename, columns=['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']):
    return pd.read_csv(filename, compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)[columns]


def draw_fig(N=3, aide, random, random_grid)
    N = N
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3   # the width of the bars
    fig, ax = plt.subplots()
    
    #men_means = (388, 458, 588)
    aide = aide
    rects1 = ax.bar(ind, aide, width, color='r')
    
    #women_means = (1200, 3596, 4009)  
    random = random
    rects2 = ax.bar(ind + width, random, width, color='y')
    
    #n_means = (3409, 2304, 8800)
    random_grid = random_grid
    rects3 = ax.bar(ind + 2 * width, random_grid,  width, color='b')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Example')
    ax.set_title('Number of samples require to make F1 score >= 0.7')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Small', 'Medium', 'Larger'))
    
    ax.legend((rects1[0], rects2[0], rects3[0]), ('AIDE', 'Random', 'Random Grid'))
    
    plt.show
    plt.save('8d.svg')

if __name__ == '__main__':
    data_path = "../data/sdss_100k.csv.gz"
    columns = ['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']
    data = load_data(data_path, columns)
    print(data.shape)
