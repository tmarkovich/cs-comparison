import numpy as np
import imp
import csv

cs_utils = imp.load_source('cs_utils',"/Users/tmarkovich/Dropbox (Aspuru-Guzik Lab)/Projects/cslib2/cs_utils.py")
drude_lorentz = imp.load_source('cs_utils',"/Users/tmarkovich/Dropbox (Aspuru-Guzik Lab)/Projects/cslib2/drude_lorentz.py")
Astruct = imp.load_source('Astruct', '/Users/tmarkovich/Dropbox (Aspuru-Guzik Lab)/Projects/cslib2/Astruct.py')
twist = imp.load_source('twist',"/Users/tmarkovich/Dropbox (Aspuru-Guzik Lab)/Projects/cslib2/TwIST_Solver.py")
time_range = np.array([2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8])

# import pint for some unit special sauce
from pint import UnitRegistry
ureg = UnitRegistry()
ureg.enable_contexts('spectroscopy')
ureg.enable_contexts('boltzmann')
Q_ = ureg.Quantity


def signal_load():
    asdf = np.loadtxt('signal.csv', delimiter=",")
    time = asdf[0,:]
    signal = asdf[1,:]
    return time*ureg.second, signal


def super_resolution():
    time, signal = signal_load()

    # Main loop that adds successively more time
    reproduced = np.zeros((len(time), len(time_range)))
    errors = np.zeros(shape=(3, len(time_range)))

    for i in range(len(time_range)):
        print "j =", i
        timeSlice = time[::time_range[i]]
        signalSlice = signal[::time_range[i]]

        A = matrixGeneration(timeSlice)
        twist.tolA = 1e-7
        twist.tolD = 1e-9
        twist.verbose = False
        lambdas, lambdas_debias, objective, times, debias_start, max_svd = twist.solve(signalSlice, A.matrix)
        reproduced[:,i] = reproduction(time.to('second').magnitude, signal, lambdas_debias, A, plotting = False)

        errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1)
        errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2)
        errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

    np.save('reproduced_SR.npy', reproduced)
    np.save('errors_SR.npy', errors)


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
        print np.shape(idx)
        timeSlice = time[idx]
        signalSlice = signal[idx]

        A = matrixGeneration(timeSlice)
        twist.tolA = 1e-7
        twist.tolD = 1e-9
        twist.verbose = False
        lambdas, lambdas_debias, objective, times, debias_start, max_svd = twist.solve(signalSlice, A.matrix)
        reproduced[:,i] = reproduction(time.to('second').magnitude, signal, lambdas_debias, A, plotting = False)

        errors[0, i] = np.linalg.norm(reproduced[:,i]-signal , ord=1)/np.linalg.norm(signal, ord=1)
        errors[1, i] = np.linalg.norm(reproduced[:,i]-signal , ord=2)/np.linalg.norm(signal, ord=2)
        errors[2, i] = np.linalg.norm(reproduced[:,i]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)

    np.save('reproduced_CS.npy', reproduced)
    np.save('errors_CS.npy', errors)


def harmonic_inversion():
    time, signal = signal_load()
    time = time.to('second').magnitude

    # Harmonic Inversion part of the comparison
    harminv = imp.load_source('harminv',"/Users/tmarkovich/Dropbox (Aspuru-Guzik Lab)/Projects/cslibrary/harminv.py")

    NF_MAX = 1400
    NF_MIN = 2200
    harminv.NF_MAX = NF_MAX
    harminv.NF_MIN = NF_MIN
    dt = 1.0
    n = 1.0
    dens = 1.0
    harminv.dens = dens
    fmin = -1.0
    fmax = 1.0
    nf = (fmax - fmin)*dt*n*dens

    DEBUG = True

    if nf > NF_MAX:
        nf = NF_MAX
    elif nf < NF_MIN:
        nf = NF_MIN
    else:
        nf = nf
    n = len(signal)

    reproduced = np.zeros((len(time), len(time_range)), dtype=np.complex128)
    errors = np.zeros(shape=(3, len(time_range)))

    for i in range(len(time_range)):
        print "j =", i
        timeSlice = time[::time_range[i]]
        signalSlice = signal[::time_range[i]]

        n = len(signalSlice)
        harminv.dataCreate(n, signal, fmin*dt, fmax*dt, nf)
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


def reproduction(time, signal, lambdas, A, plotting = False):
    reproduced = np.zeros(shape=np.shape(time))
    reproduced += cs_utils.reproduce_series(A, lambdas, time)

    if(plotting):
        cs_utils.thomas_ploting(time, [signal, reproduced],
                                title='Original Signal vs Reproduction',
                                legend_label=['Original Signal', 'Reproduced'],
                                xlabel='Time [s]',
                                ylabel='Signal [Arb]')
    return reproduced


def matrixGeneration(x, omegaMin = 0.01, omegaMax = 300, numOmega = 250, gammaMin = 1.5, gammaMax = 5, numGamma = 50):
    # Param Array Generation
    gamma_grid = np.linspace(gammaMin, gammaMax, num=numGamma)*ureg.Hz
    omega_grid = np.linspace(omegaMin, omegaMax, num=numOmega)*ureg.Hz
    length = len(gamma_grid)*len(omega_grid)
    gammaList = np.zeros(shape=(length, 1))
    omegaList = np.zeros(shape=(length, 1))

    ind = 0
    for i in xrange(np.shape(gamma_grid)[0]):
        for j in xrange(np.shape(omega_grid)[0]):
            gammaList[ind] = gamma_grid[i].magnitude
            omegaList[ind] = omega_grid[j].magnitude
            ind += 1
    param_array = np.hstack((gammaList, omegaList))*ureg.Hz

    A = Astruct.Astruct(drude_lorentz.time,
                  x.to('second').magnitude,
                  param_array.to('Hz').magnitude)
    A.matrix = cs_utils.matrix_maker(A)
    return A

if __name__ == '__main__':
    print "Compressed Sensing"
    compressed_sensing()
    print "Super Resolution"
    super_resolution()
    print "Harmonic Inversion"
    harmonic_inversion()
