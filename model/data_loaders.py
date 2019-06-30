import os
import pandas as pd
import numpy as np
from keras.utils import Sequence

from config import Config as cfg
from config import SemiSup as ssup


def load_train_set(dataset, valley=False):
    ''' convenience function for loading processed train or test splits of a dataset.
        dataset: 'E116', 'E118', etc.
    '''
    train = pd.read_csv(os.path.join(cfg.TRAIN_DIR, '{0}_train.csv'.format(dataset)))

    if valley:
        val_train = pd.read_csv(os.path.join(cfg.TRAIN_DIR, '{0}_valley_train.csv'.format(dataset)))
        assert len(train) == len(val_train)
        assert np.alltrue(train.rs == val_train.rs)
        train = pd.concat([train, val_train.iloc[:,5:]], axis=1)

    return train


def load_test_set(dataset, valley=False):
    test = pd.read_csv(os.path.join(cfg.TEST_DIR, '{0}_test.csv'.format(dataset)))

    if valley:
        val_test = pd.read_csv(os.path.join(cfg.TEST_DIR, '{0}_valley_test.csv'.format(dataset)))
        assert len(test) == len(val_test)
        assert np.alltrue(test.rs == val_test.rs)
        test = pd.concat([test, val_test.iloc[:,5:]], axis=1)

    return test


def load_seq_train_set(dataset):
    train_seq = pd.read_csv(os.path.join(cfg.TRAIN_DIR, '{0}_seq_train.csv'.format(dataset)))
    return train_seq


def load_seq_test_set(dataset):
    test_seq = pd.read_csv(os.path.join(cfg.TEST_DIR, '{0}_seq_test.csv'.format(dataset)))
    return test_seq


def load_benchmark(dataset):
    ''' loads benchmark table corresponding to the dataset
    '''
    bench = pd.read_csv(os.path.join(cfg.BENCH_DIR, '{0}_benchmarks.csv'.format(dataset)))
    return bench


#
# def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
# 	# open the CSV file for reading
# 	f = open(inputPath, "r")
#
# 	# loop indefinitely
# 	while True:
# 		# initialize our batches of images and labels
# 		images = []
# 		labels = []
#
# 		# keep looping until we reach our batch size
# 		while len(images) < bs:
# 			# attempt to read the next line of the CSV file
# 			line = f.readline()
#
# 			# check to see if the line is empty, indicating we have
# 			# reached the end of the file
# 			if line == "":
# 				# reset the file pointer to the beginning of the file
# 				# and re-read the line
# 				f.seek(0)
# 				line = f.readline()
#
# 				# if we are evaluating we should now break from our
# 				# loop to ensure we don't continue to fill up the
# 				# batch from samples at the beginning of the file
# 				if mode == "eval":
# 					break
#
# 			# extract the label and construct the image
# 			line = line.strip().split(",")
# 			label = line[0]
# 			image = np.array([int(x) for x in line[1:]], dtype="uint8")
# 			image = image.reshape((64, 64, 3))
#
# 			# update our corresponding batches lists
# 			images.append(image)
# 			labels.append(label)
#
#
class SemisupGenerator(Sequence):
    def __init__(self, dataset, batch_size=256,
                    dim=(ssup.N_FEATURES,), shuffle=True, evaluate=False):

        train_labeled = load_train_set(dataset)
        # train_unlabeled = load_unlabeled_set(dataset)

        t = load_test_set(dataset='E116')
        self.labels = train_labeled.Labels.values
        self.batch_size = batch_size
        self.shuffle = shuffle

        unlabeled_file = os.path.join(cfg.DATA_DIR, 'bigwig', 'rand_rollmean.csv')

        # open file and skip header
        self.f = open(unlabeled_file, 'r')
        header = self.f.readline()



    def _check_headers_align(self, header, train_labeled):
        h1 = header[8:].strip().split(',')
        h2 = train_labeled.iloc[:,4:].columns.values.tolist()
        assert h1 == h2


    def __len__(self):
        return int(np.floor(ssup.ROLLMEAN_LINES / self.batch_size))

    def __getitem__(self, index):

        unl_idx = np.random.randint(low=0, high=ssup.ROLLMEAN_FILESIZE, size=len(index))
        unl_idx
        unlabeled_rows = []
        for r in unl_idx:
            self.f.seek(r)
            self.f.readline()
            unlabeled_rows.append(self.f.readline())



        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]


        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np
#
# class MY_Generator(Sequence):
#
#     def __init__(self, image_filenames, labels, batch_size):
#         self.image_filenames, self.labels = image_filenames, labels
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return np.ceil(len(self.image_filenames) / float(self.batch_size))
#
#     def __getitem__(self, idx):
#         batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
#
#         return np.array([
#             resize(imread(file_name), (200, 200))
# for file_name in batch_x]), np.array(batch_y)
#
#
#
#  my_training_batch_generator = My_Generator(training_filenames, GT_training, batch_size)
#    my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)
#
#    model.fit_generator(generator=my_training_batch_generator,
#                                           steps_per_epoch=(num_training_samples // batch_size),
#                                           epochs=num_epochs,
#                                           verbose=1,
#                                           validation_data=my_validation_batch_generator,
#                                           validation_steps=(num_validation_samples // batch_size),
#                                           use_multiprocessing=True,
#                                           workers=16,
# max_queue_size=32)
