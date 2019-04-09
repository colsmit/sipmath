
import numpy as np
import scipy as sp
import matplotlib as plt

import sys


import sipmath as sm
import sipmath.pymetalog as pm

import matplotlib.pyplot as plt


# metalog demonstration
fish_data = np.loadtxt('sipmath/pymetalog/fishout.csv', delimiter=',', skiprows=1, dtype='str')[:,1].astype(np.float)

fish_metalog = pm.metalog(x=fish_data, bounds=[0,60], boundedness='b', step_len = 0.01, term_limit=9)

fish_metalog_samples = pm.rmetalog(fish_metalog, term=6, n=10000)

# sipmath demonstration
model = sm.sipmodel(10000)
fish_sipinput = model.sipinput(distribution='metalog', metalog=fish_metalog, term=9)

model.sample()

plt.hist(fish_sipinput, 100)
#plt.show()

print(pm.qmetalog(fish_metalog, y =[0.25,0.5,0.75], term = 9))


