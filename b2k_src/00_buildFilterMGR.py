#!/usr/bin/python
# -*- coding: utf-8 -*-
# korean encoding
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read('b2k.conf')

iterno = Config.get('Ingredient', 'iterno')
# kb_file	= Config.get('Ingredient', 'kb_file_prefix') + str(iterno) + Config.get('Ingredient', 'kb_file_suffix') 
kb_file = Config.get('Ingredient', 'kb_file')
domain_info = Config.get("Ingredient", "b2k_domain_info")
range_info = Config.get("Ingredient", "b2k_range_info")
import pickle


domainable_entities = {}
rangeable_entities = {}
for row in open(kb_file, 'r'):
	s, p, o = row.strip().split('\t')

	try:
		domainable_entities[p] = domainable_entities[p] | set([s])
	except KeyError:
		domainable_entities[p] = set([s])


	try:
		rangeable_entities[p] = rangeable_entities[p] | set([o])
	except KeyError:
		rangeable_entities[p] = set([o])


print '\t\t Dumping domain'
output = open(domain_info, 'wb')
pickle.dump(domainable_entities, output)
output.close()



print '\t\t Dumping range'
output = open(range_info, 'wb')
pickle.dump(rangeable_entities, output)
output.close()
