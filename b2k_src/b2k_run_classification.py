# /usr/bin/python
# -*- coding: utf-8 -*-
# 다른 서버에서 분류기 작성만. iMac에서 메모리 모자람

# elvis 서버 실행 경우
# kekeeo@elvis:~/b2k$ python b2k_run.py 
# ==================================================
# 	Loading labels... 2019-01-03 14:49:22.090395
# 	Loading sentence... 2019-01-03 14:49:29.772486
# 	Classification start... 2019-01-03 14:55:45.585947
# 	Classification end... 2019-01-04 10:25:46.951743


import pickle
import datetime

# classifier
from sklearn.linear_model import LogisticRegression
print '=' * 50 
classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')

print '\tLoading labels...', datetime.datetime.now()
train_labels = pickle.load(open('pkl_train_labels_0'))

print '\tLoading sentence...', datetime.datetime.now()
train_sen_arrays = pickle.load(open('pkl_train_sen_arrays_0'))



b2k_classifier_file = 'b2k_classifier_0'

print '\tClassification start...', datetime.datetime.now()

classifier.fit(train_sen_arrays, train_labels)

print '\tClassification end...', datetime.datetime.now()
output = open(b2k_classifier_file, 'wb')
pickle.dump(classifier, output)
output.close()
