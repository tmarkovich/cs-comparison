
# lolz
import numpy as np
import imp
import matplotlib as plt
units = imp.load_source('units',"/Users/tmarkovich/Dropbox/Projects/csfree/units.py")
sr = imp.load_source('sr',"/Users/tmarkovich/Dropbox/Projects/cslibrary/cs.py")

def signal_load():
	import csv
	f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal001/signal001.csv', 'rb')
	reader = csv.reader(f)
	signal = []
	for row in reader:
	    signal.append(row)
	f.close()
	signal = np.array(signal).astype(np.float)
	time = signal[:,0]
	signal = np.squeeze(signal[:,1])
	return time, signal

def errors(signal, reproduced):
	errors = np.zeros((3,4096/64))
	for i in range(64):
	    errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
	    errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
	    errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)