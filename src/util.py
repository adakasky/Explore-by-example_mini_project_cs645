from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd


def load_data(filename):
    return pd.read_csv(filename, compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)


if __name__ == '__main__':
    data_path = "../data/sdss_100k.csv.gz"
    data = load_data(data_path)
    print(data.shape)
