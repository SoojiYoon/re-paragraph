#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import csv
import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read('b2k.conf')

# -------------------------------------------------------------------------
sep = Config.get('Ingredient', 'sep')

iterno = Config.get('Ingredient', 'iterno')
subject_restoring_flag = Config.get('Ingredient', 'subject_restoring_flag')

# kb_file	= Config.get('Ingredient', 'kb_file_prefix') + str(iterno) + Config.get('Ingredient', 'kb_file_suffix') 
kb_file = Config.get('Ingredient', 'kb_file')
corpus_file = Config.get('Ingredient', 'corpus_file') 

result_output_file = Config.get('Ingredient', 'b2k_ds_labeled_senetences_prefix') + str(iterno)

# -------------------------------------------------------------------------

def buildListWithSet(givenlist, key, value):
	try:
		givenlist[key] = givenlist[key] | set([value])
	except KeyError:
		givenlist[key] = set([value])


class multival_dict(dict):
	def __setitem__(self, key, value):
		self.setdefault(key, []).append(value)


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
	
	import re
	entity_set = set([])
	for x in givenlist:
		x = x.replace('<e1>','[').replace('<e2>','[').replace('</e1>',']').replace('</e2>',']')
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
	centraility_out 	= nx.out_degree_centrality(G)

	out_ 	= ensure_listvalue(centraility_out)
	
	return out_



kb_memory = {}
with open(kb_file, 'r') as f:
	reader = csv.reader(f, delimiter='\t')
  	for row in reader:
  		s = row[0]
  		p = row[1]
  		o = row[2]
		buildListWithSet(kb_memory, s+sep+o, p)

fo = open(result_output_file, 'wb')
mdict = multival_dict()
with open(corpus_file, 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		sentence = row[0]
		relation = row[1]
		doc_id	= row[2]
		para_id	= row[3]
		sen_id	= row[4]

		fo.write('%s\t%s(e1,e2)\n' % (sentence.replace('[','').replace(']',''), relation))
		mdict[doc_id+sep+para_id] = sentence


'''subject restoring '''
if subject_restoring_flag == '1':
	print "restoring subject"
	for k, sens in mdict.items():
		try:
			out_ = get_centrality_in_list(sens)
		except ZeroDivisionError:
			continue
		
		if out_ is not None :
			for s in sens:
				s = s.replace('<e1>','[').replace('<e2>','[').replace('</e1>',']').replace('</e2>',']')
				if synthesize_sentence(s, out_) is not None :
					sx = synthesize_sentence(s, out_).replace('[False]','')
					f = re.findall('(\\[.*?\\])', sx)[0]
					sx = sx.replace(f,'')
					sx = ' '.join(sx.split())

					import re
					entity_set = set([])
					for entity in re.findall('(\\[.*?\\])', sx):
						entity_set.add(entity)

					for e in entity_set:
						a = f.replace('[','').replace(']','')
						b = e.replace('[','').replace(']','')
						try:
							for r in kb_memory[a+sep+b]:
								s_ = '<e1>'+a+'</e1> ' + sx.replace(e,'<e2>'+b+'</e2>').replace('[','').replace(']','')
								labeled_data = build_labeled_sentence(s_, r)
								for (labeled_sen, labeled_rel) in labeled_data:
									# print labeled_sen, labeled_rel
									fo.write('%s\t%s(e1,e2)\n' % (labeled_sen, labeled_rel))	

						except KeyError:
							continue



fo.close()
