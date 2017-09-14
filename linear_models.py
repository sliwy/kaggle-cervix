# md321825
# schemat.jpeg - wg tego diagramu probowalem na wyrywki testowac rozne modele liniowe 
#	recznie zmieniajac zestawy i zakresy danych oraz kernele i parametry klasyfikacji
# dane.jpeg - dla jasnosci dane

import numpy as np
import pylab as pl
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

def pokaz(set_name, pics = np.s_[0::], set_info = False, do_plot = True, return_both = False):
	# kombinacje nazwa: (train ; test) + ( ; _gray ; _cut ; _contrast)
	data = pickle.load( open( "dane/train/X_" + set_name + ".jpg", "rb" ) )
	labels = pickle.load( open( "dane/train/Y_" + set_name + ".jpg", "rb" ) )
	# wytnij kawalek
	if type(pics) == int:
		slice_data = np.s_[pics:pics+1:]
		slice_labels = np.s_[pics:pics+1]
		starting_pic_number = pics
	elif type(pics) == tuple:
		slice_data = np.s_[pics[0]:pics[1]:]
		slice_labels = np.s_[pics[0]:pics[1]]
		starting_pic_number = pics[0]
	else:
		print 'BLAD PICS'
		return
	data = data[slice_data]
	labels = labels[slice_labels]
	# pisz info
	if set_info:
		print 'data shape:', np.shape(data)
		print 'labels shape:', np.shape(labels)
	# narysuj
	if do_plot:
		# wymiary kartki
		n_rows = 3
		n_cols = 4
		pics_per_page = n_rows * n_cols
		# ile stron
		n_pics = np.shape(data)[0]
		n_pages = n_pics / pics_per_page + 1
		n = 0
		for page in range(n_pages):
			figure = pl.figure(figsize=(20, 10))
			for n_pic in range(pics_per_page):
				pl.subplot(n_rows, n_cols, n_pic + 1)
				if (n < n_pics):
					pl.imshow(data[n], cmap = ( 'gray' if len(np.shape(data)) == 3 else 'jet'))
					pl.title("NR: " + str(n + starting_pic_number) + " | TYP: " + str(labels[n]) )
					n += 1
			pl.show()
	# zwroc
	if return_both:
		return data, labels


data, labels = pokaz("train_gray", pics = (100, 400) , set_info = True, do_plot = False, return_both = True)
# krawedzie klas: 250, 1031

data = data.reshape((np.shape(data)[0], -1))

# reczny split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)
print "wymiary x_train, y_train :", x_train.shape, y_train.shape
print "wymiary x_test, y_test :", x_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
print clf.score(x_test, y_test)

clf = svm.SVC(kernel='linear', C=0.9)
print cross_val_score(clf, data, labels, cv=6)
