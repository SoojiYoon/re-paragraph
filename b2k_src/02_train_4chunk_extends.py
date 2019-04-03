#!/usr/bin/python
# -*- coding: utf-8 -*-
# korean encoding
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics import classification_report

import pickle
import datetime
import util_b2k
import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read('b2k.conf')

# -------------------------------------------------------------------------
# # for inputs
sep = Config.get('Ingredient', 'sep')
iterno = Config.get('Ingredient', 'iterno')

# ds_test_filename = Config.get('Ingredient', 'b2k_gold_test_filename')
ds_corpus_filename = Config.get('Ingredient', 'b2k_ds_labeled_senetences_prefix') + str(iterno)

d2v_size = int(Config.get('Ingredient', 'd2v_size'))
d2v_min = int(Config.get('Ingredient', 'd2v_min'))
d2v_window = int(Config.get('Ingredient', 'd2v_window'))
d2v_epoch = int(Config.get('Ingredient', 'd2v_epoch'))

# # for outputs
output_report_filename = Config.get('Ingredient', 'b2k_train_report_prefix') + str(iterno)

d2v_model_sh_file		= Config.get('Ingredient', 'b2k_d2v_model_sh') + str(iterno)
d2v_model_st_file		= Config.get('Ingredient', 'b2k_d2v_model_st') + str(iterno)
d2v_model_oh_file 		= Config.get('Ingredient', 'b2k_d2v_model_oh') + str(iterno)
d2v_model_ot_file		= Config.get('Ingredient', 'b2k_d2v_model_ot') + str(iterno)
b2k_classifier_file		= Config.get('Ingredient', 'b2k_classifier') + str(iterno)
b2k_train_example		= Config.get('Ingredient', 'b2k_train_example') + str(iterno)
b2k_train_label			= Config.get('Ingredient', 'b2k_train_label') + str(iterno)

# -------------------------------------------------------------------------

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

def read_b2k_model_data(x, y, sh, st, th, tt, label):
	# x = x.replace("[", "").replace("]", "")
	sbj = x.split('<e1>')[1].split('</e1>')[0]
	obj = x.split('<e2>')[1].split('</e2>')[0]

	sbj_head = x.split('<e1>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + sbj
	sbj_tail = x.split('<e1>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
	obj_head = x.split('<e2>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + obj
	obj_tail = x.split('<e2>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')

	sh.append((sbj_head.split(' '), y[0:-7]+'_sbj_head')) 
	st.append((sbj_tail.split(' '), y[0:-7]+'_sbj_tail')) 
	th.append((obj_head.split(' '), y[0:-7]+'_obj_head')) 
	tt.append((obj_tail.split(' '), y[0:-7]+'_obj_tail')) 
	label.append((y[0:-7]))


print '[START]', sys.argv[0], datetime.datetime.now()

# ============================================================================
label_part = read_data(ds_corpus_filename)

train_sh = []
train_st = []
train_oh = []
train_ot = []
train_labels = []

tagged_train_data = [(read_b2k_model_data(row[0], row[1], train_sh, train_st, train_oh, train_ot, train_labels), row[1]) for row in label_part]


from collections import namedtuple
DistantlySupervisedSentence = namedtuple('DistantlySupervisedSentence', 'words tags')

print "\tVectoring (cross-validation: train data part)...", datetime.datetime.now()
labeled_sen_train_sh = [DistantlySupervisedSentence(d, [c]) for d, c in train_sh]
labeled_sen_train_st = [DistantlySupervisedSentence(d, [c]) for d, c in train_st]
labeled_sen_train_oh = [DistantlySupervisedSentence(d, [c]) for d, c in train_oh]
labeled_sen_train_ot = [DistantlySupervisedSentence(d, [c]) for d, c in train_ot]


def build_doc2vec_model(input):
	model = Doc2Vec(min_count=d2v_min, window=d2v_window, size=d2v_size, workers=8, alpha=0.025, min_alpha=0.025)
	model.build_vocab(input)
	for epoch in range(d2v_epoch):
		print "\t\t epoch: ", epoch, datetime.datetime.now()
		model.train(input, total_examples=model.corpus_count, epochs=1)
		model.alpha -= 0.002  			# decrease the learning rate
		model.min_alpha = model.alpha  	# fix the learning rate, no decay
	return model

print '\tBuilding model...', datetime.datetime.now()
sen_model_sh = build_doc2vec_model(labeled_sen_train_sh)
sen_model_st = build_doc2vec_model(labeled_sen_train_st)
sen_model_oh = build_doc2vec_model(labeled_sen_train_oh)
sen_model_ot = build_doc2vec_model(labeled_sen_train_ot)

sen_model_sh.save(d2v_model_sh_file)
sen_model_st.save(d2v_model_st_file)
sen_model_oh.save(d2v_model_oh_file)
sen_model_ot.save(d2v_model_ot_file)

sen_model_sh_array 	= [sen_model_sh.infer_vector(doc.words) for doc in labeled_sen_train_sh]
sen_model_st_array 	= [sen_model_st.infer_vector(doc.words) for doc in labeled_sen_train_st]
sen_model_oh_array 	= [sen_model_oh.infer_vector(doc.words) for doc in labeled_sen_train_oh]
sen_model_ot_array 	= [sen_model_ot.infer_vector(doc.words) for doc in labeled_sen_train_ot]


import numpy as np
train_sen_arrays = np.column_stack((sen_model_sh_array, sen_model_st_array, sen_model_oh_array,sen_model_ot_array))

sen_model_sh_array 	= None
sen_model_st_array 	= None
sen_model_oh_array 	= None
sen_model_ot_array 	= None
sen_model_sh = None
sen_model_st = None
sen_model_oh = None
sen_model_ot = None

print '\t\t Dumping train_sen_arrays...', datetime.datetime.now()
output = open(b2k_train_example, 'wb')
pickle.dump(train_sen_arrays, output)
output.close()



print '\t\t Dumping train_labels...', datetime.datetime.now()
output = open(b2k_train_label, 'wb')
pickle.dump(train_labels, output)
output.close()



# classifier
print '\tClassification...', datetime.datetime.now()
from sklearn.linear_model import LogisticRegression
print '=' * 50 
classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
classifier.fit(train_sen_arrays, train_labels)


output = open(b2k_classifier_file, 'wb')
pickle.dump(classifier, output)
output.close()


# test_part = util_b2k.read_data(ds_test_filename)

# test_sh = []
# test_st = []
# test_oh = []
# test_ot = []
# test_labels = []
# tagged_test_data = [(read_b2k_model_data(row[0], row[1], test_sh, test_st, test_oh, test_ot, test_labels), row[1]) for row in label_part]


# labeled_sen_test_sh = [DistantlySupervisedSentence(d, [c]) for d, c in test_sh]
# labeled_sen_test_st = [DistantlySupervisedSentence(d, [c]) for d, c in test_st]
# labeled_sen_test_oh = [DistantlySupervisedSentence(d, [c]) for d, c in test_oh]
# labeled_sen_test_ot = [DistantlySupervisedSentence(d, [c]) for d, c in test_ot]

# test_model_sh_array 	= [sen_model_sh.infer_vector(doc.words) for doc in labeled_sen_test_sh]
# test_model_st_array 	= [sen_model_st.infer_vector(doc.words) for doc in labeled_sen_test_st]
# test_model_oh_array 	= [sen_model_oh.infer_vector(doc.words) for doc in labeled_sen_test_oh]
# test_model_ot_array 	= [sen_model_ot.infer_vector(doc.words) for doc in labeled_sen_test_ot]

# test_sen_arrays = np.column_stack((test_model_sh_array, test_model_st_array, test_model_oh_array, test_model_ot_array))



# print '\t' , classifier.score(test_sen_arrays, test_labels)

# pred = classifier.predict(test_sen_arrays)

# print classification_report(pred, test_labels)

# fo = open(output_report_filename, 'wb')
# fo.write('%s\n'% (classifier.score(test_sen_arrays, test_labels)))	
# fo.write('%s' % (classification_report(pred, test_labels)))	
# fo.close()

print '[END]', sys.argv[0], datetime.datetime.now()
