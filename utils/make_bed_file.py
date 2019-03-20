''' Generates bed files from data or randomly sampled from genome '''

import sys
from os.path import join
import pandas as pd
import numpy as np


GenRange = {1: 248e6, 2: 242e6, 3: 198e6, 4: 190e6,
            5: 181e6, 6: 170e6, 7: 159e6, 8: 145e6,
            9: 138e6, 10: 133e6, 11: 135e6, 12: 133e6,
            13: 114e6, 14: 107e6, 15: 101e6, 16: 90e6,
            17: 83e6, 18: 80e6, 19: 58e6, 20: 64e6,
            21: 46e6, 22: 50e6
}


def pull_bed_data(df):
    bed = df[['chr','pos','rs']]
    return bed


def mpra_to_bed(data_dir):

    fin = 'E116_train.csv'
    fout = 'E116.bed'

    data = pd.read_csv(join(data_dir, fin))
    bed = pull_bed_data(data)
    bed['chr'] = bed['chr'].map(lambda x: 'chr{0}'.format(x))
    bed['pos1'] = bed['pos'] + 1
    bed = bed[['chr','pos','pos1','rs']]

    bed.to_csv(join(data_dir, fout), sep='\t', index=False, header=False)


def random_bed_for_semisup(data_dir, n_samples=200000):

    total_range = sum([x for x in GenRange.values()])

    chrs = []
    samples = []
    for chr in range(1, 23):
        select = int(n_samples * GenRange[chr] / total_range)
        bps = np.random.randint(low=0, high=GenRange[chr], size=select)
        samples.append(bps)
        chrs.extend([chr] * select)

    samples = np.concatenate(samples)
    chrs = np.array(chrs)

    bed = pd.DataFrame({'chr': chrs,
                    'pos': samples,
                    'pos1': samples+1,
                    'rs': ['ul{0}'.format(x+1) for x in range(len(samples))]})
    bed['chr'] = bed['chr'].map(lambda x: 'chr{0}'.format(x))

    bed.to_csv(join(data_dir, 'semisup.bed'), sep='\t', index=False, header=False)


if __name__ == '__main__':

    # mpra_to_bed()
    random_bed_for_semisup('./')
