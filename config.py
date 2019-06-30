import os


GenomeDatasets = {'E116': ('LabelData_CellPaperE116.txt', 'TestData_MPRA_E116_unbalanced.txt'),
                  'E118': ('LabelData_KellisE118.txt', 'TestData_MPRA_E118.txt'),
                  'E123': ('LabelData_KellisE123.txt', 'TestData_MPRA_E123.txt')
}

SeqDatasets = 'MPRA_all_variants_19bp.fa'


class Config:
    HOME_DIR = os.path.abspath(os.path.dirname(__file__))

    # DATA_DIR = os.path.join(HOME_DIR, 'data')
    DATA_DIR = 'C:/Users/fredl/Documents/datasets/functional_variants/'

    VARIANTS_DIR = os.path.join(DATA_DIR, 'original', 'variants')
    SEQ_DIR = os.path.join(DATA_DIR, 'original', 'seq')

    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    BENCH_DIR = os.path.join(DATA_DIR, 'bench')

    OUTPUT_DIR = os.path.join(HOME_DIR, 'experiments')


class SemiSup:
    ROLLMEAN_FILESIZE = 2050000000
    ROLLMEAN_LINES = 199990
    N_FEATURES = 1016
