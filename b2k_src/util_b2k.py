#!/usr/bin/python
# -*- coding: utf-8 -*-



def buildListWithSet(givenlist, key, value):
	try:
		givenlist[key] = givenlist[key] | set([value])
	except KeyError:
		givenlist[key] = set([value])

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data


def read_csv(filename):
	import csv
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			sentence = row[0]
			relation = row[1]
			doc_id	= row[2]
			para_id	= row[3]
			sen_id	= row[4]

def find_linked_term(sentence):
	import re
	link_tag = re.compile(".*?\<link>(.*?)\</link>")
	linked_term = set([])
	linktext_in_sentence = re.findall(link_tag, sentence)

	for x in linktext_in_sentence:    
		sentence = sentence.replace(x,x.split('|')[1].replace(' ','_'))
		linked_term.add(x.split('|')[1].replace(' ','_'))

	sentence = sentence.replace('<link>','').replace('</link>','')    
	return sentence, linked_term


import itertools
def findsubset(S,m):
  return set(itertools.combinations(S,m))

