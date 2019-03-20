import os
import pandas as pd
import numpy as np

from config import Config as cfg
from config import GenomeDatasets, SeqDatasets
from utils.data_utils import split_train_test
from model.data_loaders import load_train_set, load_test_set


def load_labeled_data(dataset):

    data = pd.read_csv(os.path.join(cfg.VARIANTS_DIR, dataset[0]), delimiter="\t")
    bench = pd.read_csv(os.path.join(cfg.VARIANTS_DIR, dataset[1]), delimiter="\t")

    ### setup mpra/epigenetic data ###
    data_prepared = (data.assign(chr=data['chr'].apply( lambda x: int(x[3:]) ))
                         .sort_values(['chr','pos'])
                         .reset_index()
                         .drop('index', axis=1))

    ### setup benchmark data ###
    # modify index column to extract chr and pos information
    chr_pos = (bench.reset_index()
                    .loc[:, 'index']
                    .str
                    .split('-', expand=True)
                    .astype(int))

    # update benchmark data with chr and pos columns
    bench_prepared = (bench.reset_index()
                           .assign(chr=chr_pos[0])
                           .assign(pos=chr_pos[1])
                           .drop('index', axis=1)
                           .sort_values(['chr','pos'])
                           .reset_index()
                           .drop('index', axis=1))

    # put chr and pos columns in front for readability
    reordered_columns = ['chr','pos'] + bench_prepared.columns.values.tolist()[:-2]
    bench_prepared = bench_prepared[reordered_columns]

    return data_prepared, bench_prepared


def convert_seq_fa_to_df(loc):
    chr_list = []
    pos1_list = []
    pos2_list = []
    seq_list = []

    f = open(loc)
    while True:
        # read seq file 2 lines at a time because position and sequence are on separate lines
        line1 = f.readline()
        line2 = f.readline()
        if not line2:
            break

        chunk = line1.split('   ')[0]
        chr_raw, pos_raw = chunk.split(':')
        chr = int(chr_raw[4:])
        pos1, pos2 = map(int, pos_raw.split('-'))
        seq = line2.strip()

        chr_list.append(chr)
        pos1_list.append(pos1)
        pos2_list.append(pos2)
        seq_list.append(seq)
    f.close()

    seq_df = pd.DataFrame({'chr': chr_list, 'pos1': pos1_list, 'pos2': pos2_list, 'seq': seq_list})
    return seq_df


def merge_seq_with_mpra(dataset):

    train = load_train_set(dataset)
    test = load_test_set(dataset)

    seq_df = convert_seq_fa_to_df(os.path.join(cfg.SEQ_DIR, SeqDatasets))
    seq_df['pos'] = seq_df['pos1'] + 9

    train_seq = pd.merge(train[['chr','pos','rs','Label']], seq_df[['chr','pos','seq']], how='inner')
    test_seq = pd.merge(test[['chr','pos','rs','Label']], seq_df[['chr','pos','seq']], how='inner')

    return train_seq, test_seq


def build(dataset):

    os.makedirs(cfg.TRAIN_DIR, exist_ok=True)
    os.makedirs(cfg.TEST_DIR, exist_ok=True)
    os.makedirs(cfg.BENCH_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    data, bench = load_labeled_data(GenomeDatasets[dataset])
    bench.to_csv(os.path.join(cfg.BENCH_DIR, '{0}_benchmarks.csv'.format(dataset)), index=False)

    train, test = split_train_test(data, test_frac=0.15, seed=100)
    train.to_csv(os.path.join(cfg.TRAIN_DIR, '{0}_train.csv'.format(dataset)), index=False)
    test.to_csv(os.path.join(cfg.TEST_DIR, '{0}_test.csv'.format(dataset)), index=False)

    train_seq, test_seq = merge_seq_with_mpra(dataset)
    train_seq.to_csv(os.path.join(cfg.TRAIN_DIR, '{0}_seq_train.csv'.format(dataset)), index=False)
    test_seq.to_csv(os.path.join(cfg.TEST_DIR, '{0}_seq_test.csv'.format(dataset)), index=False)


if __name__ == '__main__':
    build('E116')
    # build('E118')
    # build('E123')
