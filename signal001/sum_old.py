
# coding: utf-8

## Signal Generation

# This simply generates the signal couplet that Vlad wanted to generate in the paper. Note, I've taken on the real part (this shouldn't be an issue) and fixed a sign error that he had.

# In[31]:

import numpy as np
import imp

asdf = np.loadtxt('signal.csv', delimiter=",")

time = asdf[0,:]
signal = asdf[1,:]
del asdf

figure(figsize=(14,10))
plot(time, signal, linewidth=5)


## Super Resolution

# In[39]:

units = imp.load_source('units',"/Users/tmarkovich/Dropbox/Projects/csfree/units.py")
sr = imp.load_source('sr',"/Users/tmarkovich/Dropbox/Projects/cslibrary/cs.py")

def matrixGeneration(time):
    sr.numOmega = 2500
    sr.numGamma = 50
    sr.gammaMin = 1.5*units.eVtocmm1*units.hbar
    sr.gammaMax = 5*units.eVtocmm1*units.hbar
    
    sr.omegaMin = 0.01*units.eVtocmm1*units.hbar
    sr.omegaMax = 300*units.eVtocmm1*units.hbar
    
    sr.allocateFreqs()
    sr.matrixBuilderDL(time)

sr.method = 'DL'
sr.maxiterm = 4000
dt = np.diff(time)[0]
sr.Verbosity = False

j = 0
tIncluded = 64
reproducedSR = np.zeros((len(time), 4096/64))
errorsSR = np.zeros((3,4096/64))

while tIncluded <= 4096:
    print "j =", j, " and tIncluded =", tIncluded
    timeSlice = time[0:tIncluded]
    matrixGeneration(timeSlice)
    lambdas = sr.l1min(signal[0:tIncluded])
    DOS, reproducedSR[:,j] = sr.constructSeriesDL(time, sr.freq, lambdas)
    
    errorsSR[0,j] = np.linalg.norm(reproducedSR[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
    errorsSR[1,j] = np.linalg.norm(reproducedSR[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
    errorsSR[2,j] = np.linalg.norm(reproducedSR[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
    
    tIncluded += 64
    j = j + 1
    
np.save('reproduced_SR.npy', reproducedSR)
np.save('errors_SR.npy', errorsSR)


## Compressed Sensing

# In[43]:

units = imp.load_source('units',"/Users/tmarkovich/Dropbox/Projects/csfree/units.py")
sr = imp.load_source('sr',"/Users/tmarkovich/Dropbox/Projects/cslibrary/cs.py")

def matrixGeneration(time):
    sr.numOmega = 2500
    sr.numGamma = 50
    sr.gammaMin = 1.5*units.eVtocmm1*units.hbar
    sr.gammaMax = 5*units.eVtocmm1*units.hbar
    
    sr.omegaMin = 0.01*units.eVtocmm1*units.hbar
    sr.omegaMax = 300*units.eVtocmm1*units.hbar
    
    sr.allocateFreqs()
    sr.matrixBuilderDL(time)

sr.method = 'DL'
sr.maxiterm = 4000
dt = np.diff(time)[0]
sr.Verbosity = False

j = 0
tIncluded = 64
reproducedCS = np.zeros((len(time), 4096/64))
errorsCS = np.zeros((3,4096/64))

indices = np.arange(time.size)
np.random.shuffle(indices)

while tIncluded <= 4096:
    print "j =", j, " and tIncluded =", tIncluded

    idx = np.sort(indices[0:tIncluded])
    timeSlice = np.sort(time[idx])
    matrixGeneration(timeSlice)
    lambdas = sr.l1min(signal[idx])
    DOS, reproducedCS[:,j] = sr.constructSeriesDL(time, sr.freq, lambdas)
    
    errorsCS[0,j] = np.linalg.norm(reproducedCS[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
    errorsCS[1,j] = np.linalg.norm(reproducedCS[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
    errorsCS[2,j] = np.linalg.norm(reproducedCS[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
    
    tIncluded += 64
    j = j + 1
    
np.save('reproduced_CS.npy', reproducedCS)
np.save('errors_CS.npy', errorsCS)


## Filter Diagonalization

# In[46]:

harminv = imp.load_source('harminv',"/Users/tmarkovich/Dropbox/Projects/cslibrary/harminv.py")

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


reproducedHI = np.zeros((len(time), 4096/64), dtype=np.complex128)
errorsHI = np.zeros((3,4096/64))
j = 0
tIncluded = 64

while tIncluded <= 4096:
    print "j =", j, " and tIncluded =", tIncluded
    n = len(signal[0:tIncluded])
    
    harminv.dataCreate(n, signal, fmin*dt, fmax*dt, nf)
    harminv.solve_once()
    harminv.compute_amps()
    
    reproducedHI[:,j] = harminv.reproduce(time)
    errorsHI[0, j] = np.linalg.norm(reproducedHI[:,j]-signal , ord=1)/np.linalg.norm(signal, ord=1) 
    errorsHI[1, j] = np.linalg.norm(reproducedHI[:,j]-signal , ord=2)/np.linalg.norm(signal, ord=2) 
    errorsHI[2, j] = np.linalg.norm(reproducedHI[:,j]-signal , ord=np.inf)/np.linalg.norm(signal, ord=np.inf)
    filename = 'parameters'+str(j)+'.txt'
    np.save(filename, harminv.data.u)
    tIncluded += 64
    j += 1

np.save('reproduced_HI.npy', reproducedHI)
np.save('errors_HI.npy', errorsHI)


## Error Plotting

# In[35]:

figure(figsize=(14, 10))
plot(np.log(errorsCS[1,:]), linewidth='5')
plot(np.log(errorsSR[1,:]), linewidth='5')
plot(np.log(errorsHI[1,:]), linewidth='5')
legend(('CS','SR','FDM'))


## Error Plotting

# In[65]:

errors_CS = np.load('errors_CS.npy')
errors_SR = np.load('errors_SR.npy')
errors_HI = np.load('errors_HI.npy')

print errors_CS

figure(figsize=(14, 10))
plot(np.log10(errors_CS[1,:]), linewidth='5')
plot(np.log10(errors_SR[1,:]), linewidth='5')
plot(np.log10(errors_HI[1,:]), linewidth='5')

