import pandas as pd
from sklearn.cross_validation import train_test_split
import cv2
import numpy as np
from sklearn.cluster import KMeans

FILE_PATH = '../data/'

def compute_histogram(img, hist_size=100):
    hist = cv2.calcHist([img], [0], mask=None, histSize=[hist_size], ranges=(0, 255))
    hist = cv2.normalize(hist, dst=hist)
    return hist

if __name__== '__main__':

    X_train = pd.read_pickle(FILE_PATH + 'X_train')
    X_additional = pd.read_pickle(FILE_PATH + 'X_additional')
    X_all = np.concatenate([X_train, X_additional])

    y_train = pd.read_pickle(FILE_PATH + 'y_train')
    y_additional = pd.read_pickle(FILE_PATH + 'y_additional')
    y_all = np.concatenate([y_train, y_additional])


    hist_size = 100
    X = np.zeros((X_all.shape[0], 3*hist_size), dtype=np.float32)

    print "Computing histograms..."
    for i in range(X_all.shape[0]):

        img = X_all[i,:,:,:]


        proc = cv2.GaussianBlur(img.copy(), (7, 7), 0)
        hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        val = hsv[:,:,2]

        hist_hue = compute_histogram(hue, hist_size)
        hist_sat = compute_histogram(sat, hist_size)
        hist_val = compute_histogram(val, hist_size)
        X[i, 0:hist_size] = hist_hue[:,0]
        X[i, hist_size:2*hist_size] = hist_sat[:,0]
        X[i, 2*hist_size:] = hist_val[:,0]

    print "Computing KMeans..."
    kmeans = KMeans(n_clusters=100)
    # Looking for clusters of similar patients to make the validation correct
    labels = kmeans.fit_predict(X)

    X_val = X_all[labels < 15]
    y_val = y_all[labels < 15]
    X_train = X_all[labels >= 15]
    y_train = y_all[labels >= 15]
    print "Validation set shape: ", X_val.shape, ' ,', y_val.shape
    print "Training set shape: ", X_train.shape, ' ,', y_train.shape
    pd.to_pickle(X_val, FILE_PATH + 'X_val_kmeans.pkl')
    pd.to_pickle(y_val, FILE_PATH + 'y_val_kmeans.pkl')
    pd.to_pickle(X_train, FILE_PATH + 'X_train_kmeans.pkl')
    pd.to_pickle(X_val, FILE_PATH + 'X_val_kmeans.pkl')

