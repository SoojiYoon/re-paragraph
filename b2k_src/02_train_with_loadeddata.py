# /usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import datetime

# classifier
from sklearn.linear_model import LogisticRegression
print '=' * 50 
classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')

print '\tLoading labels...', datetime.datetime.now()
train_labels = pickle.load(open('pkl_train_labels_2'))

print '\tLoading sentence...', datetime.datetime.now()
train_sen_arrays = pickle.load(open('pkl_train_sen_arrays_2'))



b2k_classifier_file = 'b2k_classifier_2'

print '\tClassification start...', datetime.datetime.now()

classifier.fit(train_sen_arrays, train_labels)

print '\tClassification end...', datetime.datetime.now()
output = open(b2k_classifier_file, 'wb')
pickle.dump(classifier, output)
output.close()


