# /usr/bin/python
# -*- coding: utf-8 -*-
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics import classification_report
import util_b2k
import re

import pickle
import datetime

import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read('b2k.conf')

# ======================================================================
sep = Config.get('Ingredient', 'sep')
iterno = Config.get('Ingredient', 'iterno')

d2v_size = int(Config.get('Ingredient', 'd2v_size'))
d2v_min = int(Config.get('Ingredient', 'd2v_min'))
d2v_window = int(Config.get('Ingredient', 'd2v_window'))
d2v_epoch = int(Config.get('Ingredient', 'd2v_epoch'))

d2v_model_sh_file   = Config.get('Ingredient', 'b2k_d2v_model_sh') + str(iterno)
d2v_model_st_file   = Config.get('Ingredient', 'b2k_d2v_model_st') + str(iterno)
d2v_model_oh_file   = Config.get('Ingredient', 'b2k_d2v_model_oh') + str(iterno)
d2v_model_ot_file   = Config.get('Ingredient', 'b2k_d2v_model_ot') + str(iterno)
b2k_classifier   = Config.get('Ingredient', 'b2k_classifier') + str(iterno)

isSemeval_flag = Config.get('WiseKB', 'isSemeval')
isLive_flag = Config.get('WiseKB', 'isLive')
b2k_input_doc = Config.get('WiseKB', 'extract_input_path')
predict_output_filename = Config.get('WiseKB', 'extract_output_path')
predict_output_all_filename = Config.get('WiseKB', 'extract_output_all_path')

# ======================================================================

class multival_dict(dict):
  def __setitem__(self, key, value):
    self.setdefault(key, []).append(value)


def ensure_listvalue(expectation):
  import operator
  if len(expectation) > 0:
    return '[' +  max(expectation.iteritems(), key=operator.itemgetter(1))[0] + ']'
  else:
    return None

def get_centrality_in_list(givenlist):
  import networkx as nx
  G = nx.DiGraph()
  Edges = []
  
  entity_set = set([])
  for x in givenlist:
    for entity in re.findall('(\\[.*?\\])', x):
      entity_set.add(entity)
  
  import itertools
  for (e1,e2) in set(itertools.combinations(entity_set,2)):
    # print a, b
    a = e1.replace('[','').replace(']','')
    b = e2.replace('[','').replace(']','')
    try:
      weight_ab = len(kb_memory[a+sep+b])
      Edges.append((a, b, weight_ab))
    except KeyError:
      try:
        weight_ab = len(kb_memory[b+sep+a])
        Edges.append((b, a, weight_ab))
      except KeyError:
        continue
  G.add_weighted_edges_from(Edges)
  centraility_out   = nx.betweenness_centrality(G)

  out_  = ensure_listvalue(centraility_out)
  
  return out_

def find_entity_term(sentence):
  entity_tag = re.compile(".*?\[(.*?)\]")
  entity_terms = set([])
  linktext_in_sentence = re.findall(entity_tag, sentence)

  for x in linktext_in_sentence:    
    entity_terms.add(x)

  return entity_terms


def build_labeled_sentence(sentence,rel):
  # print sentence, rel
  import re

  return_sentences = set([])
  for match_e1 in re.finditer('(\\<e1\\>.*?\\<\\/e1\\>)', s_):
    for match_e2 in re.finditer('(\\<e2\\>.*?\\<\\/e2\\>)', s_):

      if match_e1.start() < match_e2.start():
        false_sentence = s_[:match_e1.start()] + '__e1__' + s_[match_e1.start()+len(match_e1.group(1)):match_e2.start()] + \
                             '__e2__' + s_[match_e2.start()+len(match_e2.group(1)):]
        false_sentence = false_sentence.replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
        false_sentence = false_sentence.replace('__e2__',match_e2.group(1))
        false_sentence = false_sentence.replace('__e1__',match_e1.group(1))
      else:
        false_sentence = s_[:match_e2.start()] + '__e2__' + s_[match_e2.start()+len(match_e2.group(1)):match_e1.start()] + \
                             '__e1__' + s_[match_e1.start()+len(match_e1.group(1)):]
        false_sentence = false_sentence.replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
        false_sentence = false_sentence.replace('__e2__',match_e2.group(1))
        false_sentence = false_sentence.replace('__e1__',match_e1.group(1))
          

      return_sentences.add((false_sentence,rel))
  return return_sentences



def synthesize_sentence(sentence, entity):
  if entity not in sentence:
    return_sen =  '[False]' + entity + ' ' + sentence
    return ' '.join(return_sen.split())
  else:
    return ' '.join(sentence.split())


if isSemeval_flag == 'FALSE' and isLive_flag == 'TRUE':

  print '\tReady...', datetime.datetime.now()

  print '\tAnalyzing input text...', datetime.datetime.now()
  test_sh = []
  test_st = []
  test_oh = []
  test_ot = []
  test_part = []


  import csv
  with open(b2k_input_doc, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\t')
    for row in csvreader:
        s = row[0]
        temp_doc_id = row[1]
        temp_par_id = row[2]
        temp_sen_id = row[3]

        entity_set = set([])
        for entity in re.findall('(\\[.*?\\])', s):
          entity_set.add(entity)

        import itertools
        for (e1,e2) in set(itertools.combinations(entity_set,2)):
          a = e1.replace('[','').replace(']','')
          b = e2.replace('[','').replace(']','')
          x = s.replace(e1,'<e1>'+a+'</e1>').replace(e2,'<e2>'+b+'</e2>').replace('[','').replace(']','')
          sbj_head = x.split('<e1>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + a
          sbj_tail = x.split('<e1>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
          obj_head = x.split('<e2>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + b
          obj_tail = x.split('<e2>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')

          test_sh.append((sbj_head.split(' '), 'sbj_head')) 
          test_st.append((sbj_tail.split(' '), 'sbj_tail')) 
          test_oh.append((obj_head.split(' '), 'obj_head')) 
          test_ot.append((obj_tail.split(' '), 'obj_tail')) 
          test_part.append((x, 'unknown', temp_doc_id, temp_par_id, temp_sen_id))

          x = s.replace(e1,'<e2>'+a+'</e2>').replace(e2,'<e1>'+b+'</e1>').replace('[','').replace(']','')
          sbj_head = x.split('<e1>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + a
          sbj_tail = x.split('<e1>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
          obj_head = x.split('<e2>')[0].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')+ ' ' + b
          obj_tail = x.split('<e2>')[1].replace('<e2>','').replace('</e2>','').replace('<e1>','').replace('</e1>','')
       
          test_sh.append((sbj_head.split(' '), 'sbj_head')) 
          test_st.append((sbj_tail.split(' '), 'sbj_tail')) 
          test_oh.append((obj_head.split(' '), 'obj_head')) 
          test_ot.append((obj_tail.split(' '), 'obj_tail')) 
          test_part.append((x, 'unknown', temp_doc_id, temp_par_id, temp_sen_id))


  print '\tLoading vector model...'
  d2v_model_sh      = Doc2Vec.load(d2v_model_sh_file)
  d2v_model_st      = Doc2Vec.load(d2v_model_st_file)
  d2v_model_oh      = Doc2Vec.load(d2v_model_oh_file)
  d2v_model_ot      = Doc2Vec.load(d2v_model_ot_file)

  from collections import namedtuple
  DistantlySupervisedSentence = namedtuple('DistantlySupervisedSentence', 'words tags')

  labeled_sen_test_sh = [DistantlySupervisedSentence(d, [c]) for d, c in test_sh]
  labeled_sen_test_st = [DistantlySupervisedSentence(d, [c]) for d, c in test_st]
  labeled_sen_test_oh = [DistantlySupervisedSentence(d, [c]) for d, c in test_oh]
  labeled_sen_test_ot = [DistantlySupervisedSentence(d, [c]) for d, c in test_ot]

  test_model_sh_array   = [d2v_model_sh.infer_vector(doc.words) for doc in labeled_sen_test_sh]
  test_model_st_array   = [d2v_model_st.infer_vector(doc.words) for doc in labeled_sen_test_st]
  test_model_oh_array   = [d2v_model_oh.infer_vector(doc.words) for doc in labeled_sen_test_oh]
  test_model_ot_array   = [d2v_model_ot.infer_vector(doc.words) for doc in labeled_sen_test_ot]

  import numpy as np
  test_sen_arrays = np.column_stack((test_model_sh_array, test_model_st_array, test_model_oh_array, test_model_ot_array))

  clf   = pickle.load(open(b2k_classifier))
  prob1 = clf.predict_proba(test_sen_arrays)
  print '\tPredicting...', datetime.datetime.now()

  system_result_data = []
  system_result_data_all = []
  i=0
  for x in test_part:
    tmp_dict = {}

    given_sentence  = x[0]
    given_temp_doc_id  = x[2]
    given_temp_par_id  = x[3]
    given_temp_sen_id  = x[4]
    target_sbj = given_sentence.split('<e1>')[1].split('</e1>')[0]
    target_obj = given_sentence.split('<e2>')[1].split('</e2>')[0]


    for v1, v2 in zip (clf.classes_, prob1[i]):
      tmp_dict[v1] = v2

    # top 1 storing...
    top1_relation = max(tmp_dict, key=tmp_dict.get)
    system_result_data.append([target_sbj, top1_relation, target_obj, '.', tmp_dict[top1_relation],given_sentence, given_temp_doc_id, given_temp_par_id, given_temp_sen_id])

    # all storing...  
    for k, v  in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True):
      system_result_data_all.append([target_sbj, k, target_obj, '.', v,given_sentence, given_temp_doc_id, given_temp_par_id, given_temp_sen_id])
    i += 1


  result_csv_file = open(predict_output_filename, 'w')
  with result_csv_file:
    writer = csv.writer(result_csv_file, delimiter='\t')
    writer.writerows(system_result_data)

  result_csv_file = open(predict_output_all_filename, 'w')
  with result_csv_file:
    writer = csv.writer(result_csv_file, delimiter='\t')
    writer.writerows(system_result_data_all)


else:
  print "[Error] Please check the configuration file."


