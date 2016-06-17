import numpy as np

errors_CS = np.load('errors_CS.npy')
errors_HI = np.load('errors_HI.npy')
errors_SR = np.load('errors_SR.npy')
np.savetxt('errors_CS.csv',errors_CS,delimiter=",")
np.savetxt('errors_HI.csv',errors_HI,delimiter=",")
np.savetxt('errors_SR.csv',errors_SR,delimiter=",")
