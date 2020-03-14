import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='Plots file.')
parser.add_argument('filename', metavar='FILE', type=str, 
                    help='HDF5 data file.')
parser.add_argument('nstart', metavar='NSTART', type=int, 
                    help='Starting timeslice.')
parser.add_argument('nend', metavar='NEND', type=int,
                    help='Plots from NSTART to NEND - 1.')
parser.add_argument('-g', '--grid', type=str, choices=['xy', 'logr'], default='xy', 
                    help='Coordinate system for plot (default: xy).')
parser.add_argument('-d', '--dtype', type=str, choices=['raw', 'env'], default='env', 
                    help='Plot raw data or data with envelope function (default: env).')
parser.add_argument('-p', '--plot', type=str, choices=['linear', 'log'], default='linear', 
                    help='Plot data in logscale or linear (default: linear).')
group = parser.add_mutually_exclusive_group()
group.add_argument('-m', '--minmax', action="store_true",
                   help='Sets plotting range to be the minimum and maximum of data (default).')
group.add_argument('-sd', '--stdev', nargs=2, metavar=('SDLOW', 'SDHIGH'), type=float,
                   help='Sets plotting range to be between AVG - SDLOW * SD and AVG + SDHIGH * SD')
args = parser.parse_args()

#Set default plot range
if not args.minmax and not args.stdev:
    args.minmax = True

#args.filename = 'vis/128_128_512_2020_02_05_184118.h5'
#args.nstart   = 0
#args.nend     = 1

if args.nstart > args.nend:
    print("ERROR NSTART must be less than or equal to NEND\n")
    sys.exit()
if args.stdev and -args.stdev[0] > args.stdev[1]:
    print("ERROR AVG - SDLOW * SD must be lower than AVG + SDHIGH * SD\n")
    sys.exit()
    
hf   = h5py.File(args.filename, 'r')
data = np.array(hf['data/data_'+args.dtype])
var  = np.array(hf['stats/var_'+args.dtype])
avg  = np.array(hf['stats/avg_'+args.dtype])

if args.nend > data.shape[0]:
    print("ERROR NEND is longer than the file")
    sys.exit()

x1start = np.array(hf['params/x1start'])
x2start = np.array(hf['params/x2start'])
x1end = np.array(hf['params/x1end'])
x2end = np.array(hf['params/x2end'])

Min = np.array(hf['stats/min_'+args.dtype])
Max = np.array(hf['stats/max_'+args.dtype]) 

if args.plot == 'linear' and args.stdev:
    Min = avg - args.stdev[0] * np.sqrt(var)
    Max = avg + args.stdev[1] * np.sqrt(var)
if args.plot == 'log':
    if Min < 0:
        print("ERROR minimum of data is negative, can't plot log\n")
        sys.exit()
    nonzerodata = np.log10(data[np.nonzero(data)])
    data = np.log10(data)
    if args.minmax:
        Min = np.min(nonzerodata)
        Max = np.log(Max)
    if args.stdev:
        avg = np.mean(nonzerodata)
        sd  = np.std(nonzerodata)
        Min = avg - args.stdev[0] * sd
        Max = avg + args.stdev[1] * sd        

output = os.path.splitext(args.filename)[0]+'/'
os.makedirs(output,exist_ok=True)

ni = data.shape[1]
nj = data.shape[2]

dx1 = np.array(hf['params/dx1'])
dy2 = np.array(hf['params/dx2'])

x1 = np.linspace(x1start, x1end, ni+1)
x2 = np.linspace(x2start, x2end, nj+1)

if args.grid == 'xy':
    x, y = np.meshgrid(x1, x2)
if args.grid == 'logr':
    r  = np.exp(x1)
    th = x2
    Theta, R = np.meshgrid(th, r)
    x = R * np.cos(Theta)
    y = R * np.sin(Theta)

cmap = plt.get_cmap('afmhot')
cmap.set_bad(color = 'black')

for i in range(args.nstart,args.nend):
    var = data[i,:,:]
    plt.pcolormesh(x,y,var, cmap=cmap, vmin=Min, vmax=Max)
    plt.axis('scaled')  
    plt.savefig(output+str(i)+'_'+args.dtype+'.png')
    plt.clf()

h5py.File.close(hf)