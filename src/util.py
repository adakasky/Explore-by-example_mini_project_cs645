from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


def load_data(filename, columns=['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']):
    return pd.read_csv(filename, compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)[columns]


if __name__ == '__main__':
    data_path = "../data/sdss_100k.csv.gz"
    columns = ['rowc', 'colc', 'ra', 'field', 'fieldid', 'dec']
    data = load_data(data_path, columns)
    print(data.shape)
