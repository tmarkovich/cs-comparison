import numpy as np
import imp
units = imp.load_source('units',"/Users/tmarkovich/Dropbox/Projects/csfree/units.py")
sr = imp.load_source('sr',"/Users/tmarkovich/Dropbox/Projects/cslibrary/cs.py")

def super_resolution():
    time, signal = signal_load()

    sr.method = 'sine'
    dt = np.diff(time)[0]
    print "dt =", dt
    dw = 0.1
    sr.dw = dw
    print "dw =", dw
    sr.minFreq = 0
    sr.maxFreq = 4096
    sr.Verbosity = False
    spacing = 1

    # Main loop that adds successively more time
    tIncluded = 64
    j = 0
    reproduced = np.zeros((len(time), 4096/64))
    errors = np.zeros((3, 4096/64))
    while tIncluded <= 4096:
        print "j =", j, " and tIncluded =", tIncluded
        timeSlice = time[0:tIncluded]
        matrixGeneration(timeSlice, spacing)
        lambdas = sr.l1min(signal[0:tIncluded])

        DOS, reproduced[:,j] = sr.constructSeriesCosine(time, sr.freq, lambdas)
        
        for i in range(0, len(timeSlice)):
            reproduced[i,j] += lambdas[len(sr.freq)+i]*1.0

        tIncluded += 64
        j = j + 1

    for i in range(64):
        errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

    np.save('reproduced_SR.npy', reproduced)
    np.save('errors_SR.npy', errors)


def harmonic_inversion():
    time, signal = signal_load()

    # Harmonic Inversion part of the comparison
    harminv = imp.load_source('harminv',"/Users/tmarkovich/Dropbox/Projects/cslibrary/harminv.py")

    dt = 1.0/4096
    n = len(signal)
    nf = 500
    harminv.dens = 1.4
    harminv.NF_MAX = 300
    reproduced = np.zeros((len(time), 4096/64), dtype=np.complex128)
    errors = np.zeros((3,4096/64))
    j = 0
    tIncluded = 64

    while tIncluded <= 4096:
        print "j =", j, " and tIncluded =", tIncluded
        n = len(signal[0:tIncluded])
        harminv.dataCreate(n, signal[0:tIncluded], 0.0*dt, 0.1, 500)
        harminv.solve_once()
        harminv.compute_amps()
        reproduced[:,j] = harminv.reproduce(time)

        errors[0, j] = np.linalg.norm(reproduced[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, j] = np.linalg.norm(reproduced[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, j] = np.linalg.norm(reproduced[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

        filename = 'parameters'+str(j)+'.txt'
        np.save(filename, harminv.data.u)

        tIncluded += 64
        j += 1

    np.save('reproduced_HI.npy', reproduced)
    np.save('errors_HI.npy', errors)


def signal_load():
    import csv
    f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal003/signal003.csv', 'rb')
    reader = csv.reader(f)
    signal = []
    for row in reader:
        signal.append(row)
    f.close()
    signal = np.array(signal).astype(np.float)
    time = signal[:,0]
    signal = np.squeeze(signal[:,1])
    return time, signal


def compressed_sensing():
    sr.method = 'sine'
    time, signal = signal_load()

    dt = np.diff(time)[0]
    print "dt =", dt
    dw = 0.1
    sr.dw = dw
    print "dw =", dw
    sr.minFreq = 0
    sr.maxFreq = 4096
    sr.Verbosity = False

    # Main loop that adds successively more time
    tIncluded = 64
    j = 0
    reproduced = np.zeros((len(time), 4096/64))
    errors = np.zeros((3, 4096/64))
    indices = np.arange(time.size)
    np.random.shuffle(indices)
    spacing = 1.0

    while tIncluded <= 4096:
        print "j =", j, " and tIncluded =", tIncluded
        idx = np.sort(indices[0:tIncluded])
        timeSlice = np.sort(time[idx])
        
        matrixGeneration(timeSlice, spacing)
        lambdas = sr.l1min(signal[idx])

        DOS, reproduced[:,j] = sr.constructSeriesCosine(time, sr.freq, lambdas)
        
        for i in range(0, len(timeSlice)):
            reproduced[i,j] += lambdas[len(sr.freq)+i]*1.0

        errors[0, j] = np.linalg.norm(reproduced[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, j] = np.linalg.norm(reproduced[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, j] = np.linalg.norm(reproduced[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
        j += 1
        tIncluded += 64

    np.save('reproduced_CS.npy', reproduced)
    np.save('errors_CS.npy', errors)


def matrixGeneration(time, spacing):
    Adirac = np.zeros((len(time), len(time)))
    for i in range(0, len(time)):
        Adirac[:,i] = np.zeros_like(time)
        Adirac[i,i] = 1.0
    sr.allocateFreqs()
    sr.matrixBuilderCosine(time)
    sr.A = np.hstack((sr.A,Adirac))


if __name__ == '__main__':
    print "Compressed Sensing"
    compressed_sensing()
    print "Super Resolution"
    super_resolution()
    print "Harmonic Inversion"
    harmonic_inversion()