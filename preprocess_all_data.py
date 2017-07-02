"""
Data processing - oryginal, contrast and grayscale images 
"""

from __future__ import division

import os
from glob import glob

import numpy as np
import pandas as pd

from data_preprocessing import preprocess_one_image

# Just put here paths to files downloaded from kaggle
TRAIN_DATA = ".../train"
TEST_DATA = ".../test"

type_1_ids = []
type_2_ids = []
type_3_ids = []

type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])[100:]

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA) + 1:-4] for s in test_files])

tile_size = 1024, 1280
tile_size1 = 224, 224


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or \
                    image_type == "Type_2" or \
                    image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def train_data(type_1_ids, type_2_ids, type_3_ids):
    complete_cuts = []
    for k, type_ids in enumerate([type_1_ids, type_2_ids, type_3_ids]):
        print(' Start: Type_%i' % (k + 1))
        complete_cut = []

        m = len(type_ids)
        train_ids = sorted(type_ids)
        counter = 0

        for i in range(m):
            image_id = train_ids[counter]
            counter += 1
            print counter, '/', m

            complete_cut.append(preprocess_one_image(
                get_filename(image_id, 'Type_%i' % (k+1))))

        complete_cuts.append(complete_cut)

    complete_cuts[0] = np.array(complete_cuts[0], dtype=np.float32)
    complete_cuts[1] = np.array(complete_cuts[1], dtype=np.float32)
    complete_cuts[2] = np.array(complete_cuts[2], dtype=np.float32)

    X_train_cut = complete_cuts[2]
    X_train_cut = np.concatenate((X_train_cut,complete_cuts[1],complete_cuts[2]), axis=0)

    y_train_cut = np.concatenate((np.ones(complete_cuts[0].shape[0], dtype=np.int32),
                          np.full(complete_cuts[1].shape[0], 2, dtype=np.int32),
                          np.full(complete_cuts[2].shape[0], 3, dtype=np.int32)
                         ))
    y_train_cut = y_train_cut - 1
    pd.to_pickle(X_train_cut, 'X_train_cut.pkl')
    pd.to_pickle(y_train_cut, 'y_train_cut.pkl')


def test_data(test_files, test_ids):
    complete_cuts = []
    count = 1
    print "Start: test"
    for i in range(len(test_ids)):
        print count, '/', len(test_ids)
        count += 1
        image_id = test_ids[i]


        complete_cuts.append(preprocess_one_image(
            get_filename(image_id, 'Test')))

    X_test_cut = np.array(complete_cuts, dtype=np.float32)
    X_test_file_name_cut = np.array(test_ids)

    pd.to_pickle(X_test_cut, 'X_test_cut.pkl')
    pd.to_pickle(X_test_file_name_cut, 'X_test_file_name_cut.pkl')



if __name__ == "__main__":
    train_data(type_1_ids, type_2_ids, type_3_ids)
    test_data(test_files, test_ids)

    print "End"
