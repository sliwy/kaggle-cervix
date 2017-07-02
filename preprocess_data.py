"""
Data processing - oryginal, contrast and grayscale images 
"""

from __future__ import division

import math
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import mixture

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
hist_size = 30
X = np.zeros((len(test_ids), 3 * hist_size), dtype=np.float32)
n_classes = 4
RESIZED_IMAGES = {}


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
    print os.path.join(data_path, "{}.{}".format(image_id, ext))
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (
    image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i - position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif (height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area = maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r - 1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (
    int(maxArea[3] + 1 - maxArea[0] / abs(maxArea[1] - maxArea[2])), maxArea[2],
    maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img, cut):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, cut, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ff = np.zeros((gray.shape[0], gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1] / 2), int(gray.shape[0] / 2)),
                  1)

    rect = maxRect(ff)
    img_crop = img[min(rect[0], rect[2]):max(rect[0], rect[2]),
               min(rect[1], rect[3]):max(rect[1], rect[3])]
    cv2.rectangle(ff, (min(rect[1], rect[3]), min(rect[0], rect[2])),
                  (max(rect[1], rect[3]), max(rect[0], rect[2])), 3, 2)

    return img_crop


def Ra_space(img, Ra_ratio=1, a_upper_threshold=300, a_lower_threshold=0):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w * h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w / 2 - i) * (w / 2 - i) + (h / 2 - j) * (h / 2 - j))
            Ra[i * h + j, 0] = R
            a = min(imgLab[i][j][1], a_upper_threshold)
            a = max(imgLab[i][j][1], a_lower_threshold)
            Ra[i * h + j, 1] = a

    if Ra_ratio != 1:
        Ra[:, 0] /= max(Ra[:, 0])
        Ra[:, 0] *= Ra_ratio
        Ra[:, 1] /= max(Ra[:, 1])

    return Ra


def compute_histogram(img, hist_size=100):
    hist = cv2.calcHist([img], [0], mask=None, histSize=[hist_size],
                        ranges=(0, 255))
    hist = cv2.normalize(hist, dst=hist)
    return hist


def plt_st(l1, l2):
    plt.figure(figsize=(l1, l2))


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
            print counter + 1, '/', m
            counter += 1

            img = get_image_data(image_id, 'Type_%i' % (k + 1))
            img = cropCircle(img, 30)

            Ra = Ra_space(img)
            g = mixture.GMM(n_components=2, covariance_type='diag',
                            random_state=0, init_params='kmeans')

            mask_color = [0, 0, 0]
            g.fit(Ra)
            labels = g.predict(Ra)
            boolean_image_mask = np.array(labels).reshape(img.shape[0],
                                                          img.shape[1])

            outer_cluster_label = boolean_image_mask[0, 0]

            new_image = img.copy()

            for ii in range(boolean_image_mask.shape[0]):
                for jj in range(boolean_image_mask.shape[1]):
                    if boolean_image_mask[ii, jj] == outer_cluster_label:
                        new_image[ii, jj] = mask_color

            img_masked = new_image.copy()

            img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);

            _, thresh_mask = cv2.threshold(img_masked_gray, 0, 255, 0)
            _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)

            main_contour = \
            sorted(contours_mask, key=cv2.contourArea, reverse=True)[0]

            x, y, w, h = cv2.boundingRect(main_contour)

            ### CUT ###

            img = img[y:(y + h), x:(x + w)]

            img = cv2.resize(img, dsize=tile_size1)

            ### CONTRAST ###
            img_yuv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])


            complete_cut.append(img[:, :, :])

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

        img = get_image_data(image_id, 'Test')
        img = cropCircle(img, 30)

        Ra = Ra_space(img)

        g = mixture.GMM(n_components=2, covariance_type='diag', random_state=0,
                        init_params='kmeans')
        mask_color = [0, 0, 0]
        g.fit(Ra)
        labels = g.predict(Ra)
        boolean_image_mask = np.array(labels).reshape(img.shape[0],
                                                      img.shape[1])

        outer_cluster_label = boolean_image_mask[0, 0]

        new_image = img.copy()

        for ii in range(boolean_image_mask.shape[0]):
            for jj in range(boolean_image_mask.shape[1]):
                if boolean_image_mask[ii, jj] == outer_cluster_label:
                    new_image[ii, jj] = mask_color

        img_masked = new_image.copy()

        img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);

        _, thresh_mask = cv2.threshold(img_masked_gray, 0, 255, 0)
        _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

        main_contour = sorted(contours_mask, key=cv2.contourArea, reverse=True)[
            0]

        x, y, w, h = cv2.boundingRect(main_contour)

        ### CUT ###

        img = img[y:(y + h), x:(x + w)]
        img = cv2.resize(img, dsize=tile_size1)
        complete_cuts.append(img[:, :, :])

    X_test_cut = np.array(complete_cuts, dtype=np.float32)
    X_test_file_name_cut = np.array(test_ids)

    pd.to_pickle(X_test_cut, 'X_test_cut.pkl')
    pd.to_pickle(X_test_file_name_cut, 'X_test_file_name_cut.pkl')



if __name__ == "__main__":
    train_data(type_1_ids, type_2_ids, type_3_ids)
    test_data(test_files, test_ids)

    print "End"
