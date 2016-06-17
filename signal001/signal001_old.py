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

def super_resolution():
    time, signal = signal_load()

    sr.method = 'sine'
    dt = np.diff(time)[0]
    print "dt =", dt
    dw = 0.1*np.pi
    sr.dw = dw
    print "dw =", dw
    sr.minFreq = 0
    sr.maxFreq = 200*np.pi
    sr.Verbosity = False
    stepRange = np.arange(0.2, 0.8, 0.01)

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
            reproduced[:,j] += lambdas[len(sr.freq)+i]*np.sign(time - stepRange[i])
        for i in range(0, len(stepRange)):
            reproduced[:,j] += lambdas[len(sr.freq)+len(stepRange)+i]*np.sign(stepRange[i] - time)

        tIncluded += 64
        j = j + 1
    errors = np.zeros((3,4096/64))
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


def compressed_sensing():
    sr.method = 'sine'
    time, signal = signal_load()

    dt = np.diff(time)[0]
    print "dt =", dt
    dw = 0.1*np.pi
    sr.dw = dw
    print "dw =", dw
    sr.minFreq = 0
    sr.maxFreq = 200*np.pi
    sr.Verbosity = False

    # Main loop that adds successively more time
    tIncluded = 64
    j = 0
    reproduced = np.zeros((len(time), 4096/64))
    errors = np.zeros((3,4096/64))
    indices = np.arange(time.size)
    np.random.shuffle(indices)
    stepRange = np.arange(0.2, 0.8, 0.01)

    while tIncluded <= 4096:
        print "j =", j, " and tIncluded =", tIncluded
        idx = np.sort(indices[0:tIncluded])
        timeSlice = np.sort(time[idx])
        
        matrixGeneration(timeSlice, stepRange)
        lambdas = sr.l1min(signal[idx])

        DOS, reproduced[:,j] = sr.constructSeriesSine(time, sr.freq, lambdas)
        for i in range(0, len(stepRange)):
            reproduced[:,j] += lambdas[len(sr.freq)+i]*np.sign(time - stepRange[i])
        for i in range(0, len(stepRange)):
            reproduced[:,j] += lambdas[len(sr.freq)+len(stepRange)+i]*np.sign(stepRange[i] - time)

        errors[0, j] = np.linalg.norm(reproduced[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, j] = np.linalg.norm(reproduced[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, j] = np.linalg.norm(reproduced[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
        j += 1
        tIncluded += 64

    np.save('reproduced_CS.npy', reproduced)
    np.save('errors_CS.npy', errors)


def matrixGeneration(time, stepRange):
    tmax = max(time)
    Astep = np.zeros((len(time),len(stepRange)))
    for i in range(0, len(stepRange)):
        if stepRange[i] < tmax:
            Astep[:,i] = np.sign(time - stepRange[i])
        else:
            break
    AstepInv = np.zeros((len(time),len(stepRange)))
    for i in range(0, len(stepRange)):
        if stepRange[i] < tmax:
            AstepInv[:,i] = np.sign(stepRange[i] - time)
        else:
            break
    Astep = np.hstack((Astep, AstepInv))
    sr.allocateFreqs()
    sr.matrixBuilderSine(time)
    sr.A = np.hstack((sr.A,Astep))


if __name__ == '__main__':
    compressed_sensing()
    super_resolution()
    # harmonic_inversion()
