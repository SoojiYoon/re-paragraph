ds_test_filename = '/Users/kekeeo/Desktop/2018_summer_task/tta/tta_init.csv'


import csv
with open(ds_test_filename, 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		try:
			print row[0]
		except IndexError:
			print row
		