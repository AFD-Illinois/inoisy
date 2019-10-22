import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os

if len(sys.argv) != 4:
  print('ERROR Format is python plot.py [file] [nstart] [nend]')
  sys.exit()

filename = sys.argv[1]
nstart   = int(sys.argv[2])
nend     = int(sys.argv[3])

#filename = 'vis/256_256_1024_2019_10_10_025729.h5'
#nstart   = 1
#nend     = 2

if nstart > nend:
    print('ERROR nstart must be less than or equal to nend')
    sys.exit()

hf   = h5py.File(filename, 'r')
data = hf['data/data']
data = np.array(data)
Min  = hf['params/min']
Min  = np.array(Min)
Max  = hf['params/max']
Max  = np.array(Max)

if nend > data.shape[0]:
    print('ERROR nend is longer than the file')
    sys.exit()

output = os.path.splitext(filename)[0]+'/'
os.makedirs(output,exist_ok=True)

ni = data.shape[2]
nj = data.shape[1]

x0  = 0.0704486
dx1 = 0.0400164 * 96/ni

r  = np.exp(np.linspace(x0, x0 + ni*dx1, ni))
th = np.linspace(0, 2*np.pi, nj)
R, Theta = np.meshgrid(r, th)
x = R * np.cos(Theta)
y = R * np.sin(Theta)
for i in range(nstart,nend):
    var = data[i,:,:]
    plt.pcolormesh(x,y,var, cmap='jet', vmin=Min, vmax=Max/3)
    plt.axis('equal')
    plt.savefig(output+str(i)+'.png')
    plt.clf()