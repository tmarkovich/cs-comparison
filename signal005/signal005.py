import numpy as np
import imp
import csv

cs_utils = imp.load_source('cs_utils',"/Users/tmarkovich/Dropbox/Projects/cslib2/cs_utils.py")
Astruct = imp.load_source('Astruct', '/Users/tmarkovich/Dropbox/Projects/cslib2/Astruct.py')
twist = imp.load_source('twist',"/Users/tmarkovich/Dropbox/Projects/cslib2/TwIST_Solver.py")
time_range = np.array([2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8])

def signal_load():
    import csv
    f = open('/Users/tmarkovich/Dropbox/Projects/CSComparisonPaper/signals/signal005/signal005.csv', 'rb')
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

    # Main loop that adds successively more time
    reproduced = np.zeros((len(time), len(time_range)))
    errors = np.zeros(shape=(3, len(time_range)))

    for i in range(len(time_range)):
        print "j =", i
        timeSlice = time[::time_range[i]]
        signalSlice = signal[::time_range[i]]

        A_diracs, A_cos, A = matrixGeneration(timeSlice)
        twist.tolA = 1e-7
        twist.tolD = 1e-9
        twist.verbose = False
        lambdas, lambdas_debias, objective, times, debias_start, max_svd = twist.solve(signalSlice, A)
        reproduced[:,i] = reproduction(time, signal, lambdas_debias, A_diracs, A_cos, A, plotting = False)

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

    reproduced = np.zeros((len(time), len(time_range)), dtype=np.complex128)
    errors = np.zeros(shape=(3, len(time_range)))

    for i in range(len(time_range)):
        print "j =", i
        timeSlice = time[::time_range[i]]
        signalSlice = signal[::time_range[i]]

        n = len(signalSlice)
        harminv.dataCreate(n, signalSlice, 0.0*dt, 0.1, 500)
        harminv.solve_once()
        harminv.compute_amps()
        reproduced[:,i] = harminv.reproduce(time)

        errors[0, i] = np.linalg.norm(reproduced[:, i]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, i] = np.linalg.norm(reproduced[:, i]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, i] = np.linalg.norm(reproduced[:, i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

        filename = 'parameters'+str(i)+'.txt'
        np.save(filename, harminv.data.u)

    np.save('reproduced_HI.npy', reproduced)
    np.save('errors_HI.npy', errors)


def compressed_sensing():
    time, signal = signal_load()

    # Main loop that adds successively more time
    reproduced = np.zeros((len(time), len(time_range)))
    errors = np.zeros(shape=(3, len(time_range)))

    indices = np.arange(time.size)
    np.random.shuffle(indices)

    for i in range(len(time_range)):
        print "j =", i
        idx = np.sort(indices[0:len(time[::time_range[i]])])
        timeSlice = time[idx]
        signalSlice = signal[idx]

        A_diracs, A_cos, A = matrixGeneration(timeSlice, debug = False)
        twist.tolA = 1e-7
        twist.tolD = 1e-9
        twist.verbose = False
        lambdas, lambdas_debias, objective, times, debias_start, max_svd = twist.solve(signalSlice, A)
        reproduced[:,i] = reproduction(time, signal, lambdas_debias, A_diracs, A_cos, A, plotting = False)

        errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
        errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
        errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

    np.save('reproduced_CS.npy', reproduced)
    np.save('errors_CS.npy', errors)

def dirac(time, location):
    diracs = np.zeros((len(time), ))
    diracs[location] = 1.0
    return diracs


def cosine(x, freq):
    return np.cos(x*freq)


def matrixGeneration(x, minfreq = 0, maxfreq = 200, dw = 0.1, debug = False):
    spike_locations = np.ones_like(x)
    # Creates the diracs function structure
    A_diracs = Astruct.Astruct(dirac, 
                  x, 
                  spike_locations)
    A_diracs.matrix = cs_utils.matrix_maker(A_diracs)
    

    # Creates the sine function structure
    freq = np.arange(minfreq, maxfreq+dw, dw)
    A_cos = Astruct.Astruct(cosine, 
                  x, 
                  freq)
    A_cos.matrix = cs_utils.matrix_maker(A_cos)
    A = np.hstack((A_cos.matrix, A_diracs.matrix))

    return A_diracs, A_cos, A


def reproduction(time, signal, lambdas, A_diracs, A_cos, A, plotting = False):
    reproduced = np.zeros(shape=np.shape(time))
    reproduced += cs_utils.reproduce_series(A_cos, lambdas[0:len(A_cos.params)], time)

    offset = len(A_cos.params)
    reproduced += cs_utils.reproduce_series(A_diracs, lambdas[offset:offset+len(A_diracs.params)], time)

    if(plotting):
        cs_utils.thomas_ploting(time, [signal, reproduced], 
                                title='Original Signal vs Reproduction',
                                legend_label=['Original Signal', 'Reproduced'],
                                xlabel='Time [s]',
                                ylabel='Signal [Arb]')
    return reproduced

if __name__ == '__main__':
    print "Compressed Sensing"
    compressed_sensing()
    print "Super Resolution"
    super_resolution()
    print "Harmonic Inversion"
    harmonic_inversion()