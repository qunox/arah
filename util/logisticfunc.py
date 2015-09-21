from __future__ import  division
__author__ = 'mmdali'

from math import exp
import numpy as np
from matplotlib import pyplot as plt

type = 'dampedsin'
L = 1
irr = 1000
x0 = 0.5*irr
irr_list = range(irr)
k_list = [10/irr,5/irr, 2.5/irr, 100/irr]
k_label = ['10/T', '5/T', '2.5/T', '100/T']
collist = ['r','b','g','k']

if type == 'derlogis':
    logisticfunc = lambda x : L * exp(-k*(x - x0)) / (1 + exp(-k*(x - x0))**2)
elif type == 'logis':
    logisticfunc = lambda x: L / (1 + exp(-k*(x - x0)))
elif type == 'reverselogis':
    logisticfunc = lambda x: L - (L / (1 + exp(-k*(x - x0))))
elif type =='deriviatetanh':
    logisticfunc = lambda x: L / (np.cosh((-k*(x - x0))) * np.cosh((-k*(x - x0))))
elif type == 'dampedsin':
    logisticfunc = lambda  x: abs(L * exp(-k*x) * np.cos(0.07 *x))

k_index = 0
for k ,c in zip(k_list ,collist):
    print 'calculating for k=:' , k
    result = map(logisticfunc, irr_list)
    plt.plot(irr_list , result , color= c , label = k_label[k_index])
    k_index +=1

plt.xlabel('Training Iteration')
plt.ylabel('Smoothing Value')
#plt.title('Learning Constant vs Training Iteration using the logistic function')
#plt.legend(title = 'Gradient - k ')
plt.ylim(0, 1)
plt.grid()
plt.show()