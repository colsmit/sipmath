import numpy as np
import pandas as pd
import pymetalog as m
import pymetalog.class_method as cm
import matplotlib.pyplot as plt
import scipy


fish = np.loadtxt('fishout.csv', delimiter=',', skiprows=1, dtype='str')[:,1]
fish = fish.astype(np.float)

#a = m.metalog(x=fish, bounds=[0,10000], boundedness='u', step_len = 0.01, term_limit=9)
a = m.metalog(x=fish, bounds=[100], boundedness='su', step_len = 0.01, term_limit=9)
#1000000000000

print(a.output_list['A'])
print(a.output_list['Validation'])
print(type(a.output_list['A'].iloc[0,0]))

print(m.qmetalog(a,[.1,.73],term=5))
#cm.plot(a)
#plt.show()
cm.summary(a)