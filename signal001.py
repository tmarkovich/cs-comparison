import numpy as np
import imp
import matplotlib as plt
units = imp.load_source('units',"/Users/tmarkovich/Dropbox/Projects/csfree/units.py")
sr = imp.load_source('sr',"/Users/tmarkovich/Dropbox/Projects/cslibrary/cs.py")

def super_resolution():
	import csv
	f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal001.csv', 'rb')
	reader = csv.reader(f)
	signal = []
	for row in reader:
	    signal.append(row)
	f.close()
	signal = np.array(signal).astype(np.float)
	time = signal[:,0]
	signal = np.squeeze(signal[:,1])

	sr.method = 'sine'
	dt = np.diff(time)[0]
	print "dt =", dt
	dw = 0.1
	sr.dw = dw
	print "dw =", dw
	sr.minFreq = 0
	sr.maxFreq = 4096
	sr.Verbosity = False
	stepRange = np.linspace(0, 1, 11)

	# Main loop that adds successively more time
	tIncluded = 64
	j = 0
	reproduced = np.zeros((len(time), 4096/64))
	while tIncluded <= 4096:
	    print "j =", j, " and tIncluded =", tIncluded
	    timeSlice = time[0:tIncluded]
	    matrixGeneration(timeSlice, stepRange)
	    lambdas = sr.l1min(signal[0:tIncluded])
	    DOS, reproduced[:,j] = sr.constructSeriesSine(time, sr.freq, lambdas)
	    for i in range(0, len(stepRange)):
	        reproduced[:,j] += lambdas[len(sr.freq)+i]*sr.step(time, stepRange[i])
	    for i in range(0, len(stepRange)):
	        reproduced[:,j] += lambdas[len(sr.freq)+len(stepRange)+i]*sr.invStep(time, stepRange[i])
	    tIncluded += 64
	    j = j + 1
	errors = np.zeros((3,4096/64))
	for i in range(64):
	    errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
	    errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
	    errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)


def harmonic_inversion():
	import csv
	f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal001.csv', 'rb')
	reader = csv.reader(f)
	signal = []
	for row in reader:
	    signal.append(row)
	f.close()
	signal = np.array(signal).astype(np.float)
	time = signal[:,0]
	signal = np.squeeze(signal[:,1])

	# Harmonic Inversion part of the comparison
	harminv = imp.load_source('harminv',"/Users/tmarkovich/Dropbox/Projects/cslibrary/harminv.py")

	dt = 1.0/4096
	n = len(signal[:,1])
	nf = 500
	harminv.dens = 1.4
	harminv.NF_MAX = 300
	harminv.dataCreate(n, signal[:,1], 0.0*dt, 0.1, nf)
	harminv.solve_once()
	harminv.compute_amps()
	reproduced = harminv.reproduce(time)
	plt.plot(time, np.real(reproduced) , color='b', linewidth=3)
	plt.plot(time, np.imag(reproduced) , color='r', linewidth=3)
	plt.plot(time, signal[:,1], color='g', linewidth=3)


def compressed_sensing():
	sr.method = 'sine'

	import csv
	f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal001.csv', 'rb')
	reader = csv.reader(f)
	signal = []
	for row in reader:
	    signal.append(row)
	f.close()
	signal = np.array(signal).astype(np.float)
	time = signal[:,0]
	signal = np.squeeze(signal[:,1])


	dt = np.diff(time)[0]
	print "dt =", dt
	dw = 0.1
	sr.dw = dw
	print "dw =", dw
	sr.minFreq = 0
	sr.maxFreq = 4096
	sr.Verbosity = False
	stepRange = np.linspace(0, 1, 11)

	# Main loop that adds successively more time
	tIncluded = 64
	j = 0
	reproduced = np.zeros((len(time), 4096/64))
	errors = np.zeros((3,4096/64))
	indices = np.arange(time.size)
	np.random.shuffle(indices)

	while tIncluded <= 4096:
	    print "j =", j, " and tIncluded =", tIncluded
	    idx = np.sort(indices[0:tIncluded])
	    timeSlice = np.sort(time[idx])
	    
	    matrixGeneration(timeSlice, stepRange)
	    lambdas = sr.l1min(signal[idx])
	    DOS, reproduced[:,j] = sr.constructSeriesSine(time, sr.freq, lambdas)
	    for i in range(0, len(stepRange)):
	        reproduced[:,j] += lambdas[len(sr.freq)+i]*sr.step(time, stepRange[i])
	    for i in range(0, len(stepRange)):
	        reproduced[:,j] += lambdas[len(sr.freq)+len(stepRange)+i]*sr.invStep(time, stepRange[i])
	    errors[0, j] = np.linalg.norm(reproduced[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
	    errors[1, j] = np.linalg.norm(reproduced[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
	    errors[2, j] = np.linalg.norm(reproduced[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
	    j += 1
	    tIncluded += 64


def matrixGeneration(time, stepRange):
    Astep = np.zeros((len(time), len(stepRange)))
    AstepInv = np.zeros((len(time), len(stepRange)))
    for i in range(0, len(stepRange)):
        Astep[:,i] = sr.step(time, stepRange[i])
    for i in range(0, len(stepRange)):
        AstepInv[:,i] = sr.invStep(time, stepRange[i])
    Astep = np.hstack((Astep, AstepInv))
    sr.allocateFreqs()
    sr.matrixBuilderSine(time)
    sr.A = np.hstack((sr.A,Astep))


if __name__ == '__main__':
	compressed_sensing()