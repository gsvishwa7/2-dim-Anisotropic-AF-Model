import numpy as np
from scipy import stats
import random
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
import os
#path = 'C:\\Users\\Chester\OneDrive - Imperial College London\\Y3\\UROP'
#os.chdir(path)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class Lattice:
    
    def __init__(self, L, draw = False):
        self.__Lx = L
        self.__Ly = L
        self.__bonds = [[(i,j),(i+1,j)] for i in range(self.__Lx-1) for j in range(self.__Ly)] + \
                       [[(i,j),(i,j+1)] for i in range(self.__Lx) for j in range(self.__Ly-1)]
        self.__ptr = {}
        self.__draw = draw
        self.__cluster = 0

        if self.__draw:
            self.__fig = plt.figure()
            self.__axes = plt.gca()
            for b in self.__bonds:
                x1, y1 = b[0]
                x2, y2 = b[1]
                line = plt.Line2D((x1,x2), (y1,y2), lw=0.1, color='k')
                self.__axes.add_line(line)
            self.__axes.set_xlim(-1,self.__Lx)
            self.__axes.set_ylim(-1,self.__Ly)
            plt.axis('equal')
            self.__fig.canvas.draw()

    def filter_bonds(self, ratio, seed=None):
        px, py = ratio
        M = len(self.__bonds)
        m = int(0.5*M)
        if py < 1:
            n = int(0.5*py*M)
            n_list = list(range(m,2*m,1))
            n_list = np.random.choice(n_list, size=n, replace=False)
            n_list = np.flip(np.sort(n_list))
            for n in n_list: del self.__bonds[n]
        if px < 1:
            n = int(0.5*px*M)
            n_list = list(range(0,m,1))
            n_list = np.random.choice(n_list, size=n, replace=False)
            n_list = np.flip(np.sort(n_list))
            for n in n_list: del self.__bonds[n]

    def get_numberofbonds(self):
        return len(self.__bonds)

    def randomise_bonds(self, seed):
        random.seed(seed)
        M = len(self.__bonds)
        for i in range(M):
            j = random.randint(i,M-1)
            self.__bonds[i], self.__bonds[j] = self.__bonds[j], self.__bonds[i]
        return self.__bonds

    def findroot(self, site):
            r = s = site
            while isinstance(self.__ptr[r],tuple):
                self.__ptr[s] = self.__ptr[r]
                s = r
                r = self.__ptr[r]
            return r

    def add_bond(self, index):
        s1 = self.__bonds[index][0]
        s2 = self.__bonds[index][1]
        if s1 in self.__ptr:
            if s2 in self.__ptr:
                r1 = self.findroot(s1)
                r2 = self.findroot(s2)
                if r1 != r2:
                    c1, x1min, x1max, y1min, y1max = tuple(self.__ptr[r1])
                    c2, x2min, x2max, y2min, y2max = tuple(self.__ptr[r2])
                    if x1min < x2min:
                        xmin = x1min
                    else:
                        xmin = x2min
                    if x1max > x2max:
                        xmax = x1max
                    else:
                        xmax = x2max
                    if y1min < y2min:
                        ymin = y1min
                    else:
                        ymin = y2min
                    if y1max > y2max:
                        ymax = y1max
                    else:
                        ymax = y2max
                    c = 2*(xmax-xmin+ymax-ymin)
                    c = [c,xmin,xmax,ymin,ymax]
                    if c2 > c1:
                        self.__ptr[r2] = c
                        self.__ptr[r1] = r2
                    else:
                        self.__ptr[r1] = c
                        self.__ptr[r2] = r1
                else:
                    c, xmin, xmax, ymin, ymax = tuple(self.__ptr[r1])
                    c = [c,xmin,xmax,ymin,ymax]
            else:
                r1 = self.findroot(s1)
                self.__ptr.update({s2:r1})
                c, xmin, xmax, ymin, ymax = tuple(self.__ptr[r1])
                x, y = s2
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
                c = 2*(xmax-xmin+ymax-ymin)
                c = [c,xmin,xmax,ymin,ymax]
                self.__ptr[r1] = c
        else:
            if s2 in self.__ptr:
                r2 = self.findroot(s2)
                self.__ptr.update({s1:r2})
                c, xmin, xmax, ymin, ymax = tuple(self.__ptr[r2])
                x, y = s1
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
                c = 2*(xmax-xmin+ymax-ymin)
                c = [c,xmin,xmax,ymin,ymax]
                self.__ptr[r2] = c
            else:
                x1, y1 = s1
                x2, y2 = s2
                if x1 < x2:
                    xmin = x1
                    xmax = x2
                else:
                    xmin = x2
                    xmax = x1
                if y1 < y2:
                    ymin = y1
                    ymax = y2
                else:
                    ymin = y2
                    ymax = y1
                c = 2*(xmax-xmin+ymax-ymin)
                c = [c,xmin,xmax,ymin,ymax]                
                self.__ptr.update({s1:c})
                self.__ptr.update({s2:s1})
        # draw bond
        if self.__draw:
                x1, y1 = s1
                x2, y2 = s2
                line = plt.Line2D((x1,x2), (y1,y2), lw=1.0, color='k')
                self.__axes.add_line(line)
                self.__fig.canvas.draw()
        # check for percolation
        if xmin==0 and xmax==self.__Lx-1 or ymin==0 and ymax==self.__Ly-1:
            if self.__draw:
                rc = self.findroot(s1)
                for i in range(0,self.__Lx):
                    for j in range(0,self.__Ly):
                        if (i,j) in self.__ptr: r = self.findroot((i,j)) 
                cluster = [s for (s,r) in self.__ptr.items() if r == rc] + [rc]
                x1, y1 = s1
                x2, y2 = s2
                line = plt.Line2D((x1,x2), (y1,y2), lw=1.0, color='b')
                self.__axes.add_line(line)
                for s in cluster:
                    circle = plt.Circle(s, 0.25, fc='r')
                    self.__axes.add_patch(circle)
                self.__fig.canvas.draw()
                print('Percolated!')
            return True, c[0]
        # get size of largest cluster
        if c[0] > self.__cluster: self.__cluster = c[0]                           
        return False, self.__cluster

visualise = False
if visualise:
    lattice = Lattice(L=10, draw=True)
    lattice.filter_bonds(1)
    lattice.randomise_bonds(None)
    i = 0
    percolate = False
    for i in range(60):
        percolate, cluster = lattice.add_bond(i)
        if percolate: break
        i += 1
        print(i)
    print(cluster)
    plt.show()

L = 100
runs_per_L = 1000
p_list = list(range(1,2*L*(L-1)+1,1))
ratios = [(1.0,1.0), \
          (1.0,0.9),(1.0,0.8),(1.0,0.7),(1.0,0.6),(1.0,0.5),(1.0,0.4),(1.0,0.3),(1.0,0.2),(1.0,0.1),(1.0,0.0), \
          (0.9,1.0),(0.8,1.0),(0.7,1.0),(0.6,1.0),(0.5,1.0),(0.4,1.0),(0.3,1.0),(0.2,1.0),(0.1,1.0),(0.0,1.0)]

generate = False
if generate:
    total_time = time.time()
    for ratio in ratios:
        px, py = ratio
        start_time = time.time()
        p = {i:[] for i in p_list}
        for run in range(runs_per_L):
            start = time.time()
            lattice = Lattice(L=L, draw=False)
            lattice.filter_bonds(ratio)
            lattice.randomise_bonds(run)
            i = 0
            percolate = False
            numberofbonds = lattice.get_numberofbonds()
            for i in range(numberofbonds):
                percolate, cluster = lattice.add_bond(i)
                if percolate: break
                else: p[i+1] += [cluster]
            del lattice
            end = time.time()
            print('ratio = (%2.2f,%2.2f), run = %d, time = %5.5fs'%(px, py, run, end-start))
        save_obj(p, 'BP2D, ratio=(%2.2f,%2.2f)'%(px,py))
        end_time = time.time()
        print('ratio = (%2.2f,%2.2f), time = %5.5fs'%(px, py, end_time-start_time))
    print('total time = %5.5fs'%(time.time()-total_time))

def find_nearest(array, value):
    array = np.asarray(array)
    i = (np.abs(array-value)).argmin()
    return i

p_list = list(range(1,2*L*(L-1)+1,100))

analyse = True
if analyse:
    total_time = time.time()
    numberofbonds = 2*L*(L-1)
    clusters = {}
    for ratio in ratios:
        start_time = time.time()
        px, py = ratio
        p = load_obj('BP2D, ratio=(%2.2f,%2.2f)'%(px,py))
        P = p
        cluster = {i:0 for i in p_list}
        plot = []
        for i in p_list:
            if len(p[i])==0: continue
            bins = list(range(min(p[i]),max(p[i])+1,2))
            P[i], bins = np.histogram(p[i], bins=bins)
            bins = bins[:-1]
            if len(P[i])==0:
                cluster[i] = cluster[p_list[p_list.index(i)-1]]
                continue            
            cluster[i] = bins[np.where(P[i]==max(P[i]))[0][0]]
            if i == plot:
                plt.figure()
                plt.plot([bins[0]], P[i][0]/max(P[i]), 'w.', label='$p$ = %2.2f'%(float(i)/float(numberofbonds)))
                plt.plot(bins, P[i]/max(P[i]), '.', label=r'$\frac{dP}{dL}$')
            P[i] = P[i] / float(runs_per_L)
            for j in range(P[i].size):
                if j != 0:
                    P[i][-j-1] += P[i][-j]
            if i == plot:
                plt.plot(bins, P[i], '.', label='$P$')
                plt.plot([cluster[i],cluster[i]],[0,1],'r:', label=r'$\xi$ = %2.2f'%(cluster[i]))
                plt.xlabel(r'$L$')
                plt.ylabel(r'$P$')
                plt.legend()
                plt.show()

        if isinstance(plot,int):
            plt.figure()
            x, y = [], []
            for i in p_list:
                x += [float(i)/float(numberofbonds)]
                y += [cluster[i]]
            plt.plot(x, y, '.')
            plt.xlabel(r'$p$')
            plt.ylabel(r'$\xi$')
            plt.show()

        clusters.update({ratio:cluster})
        end_time = time.time()
        print('ratio = (%2.2f,%2.2f), time = %5.5fs'%(px, py, end_time-start_time))
    
    x = []
    y = []
    z = []
    start_time = time.time()
    for ratio in ratios:
        start_time = time.time()
        for i in p_list: 
            if clusters[ratio][i] > 100: 
                z += [100]
            else: 
                z += [clusters[ratio][i]]
            p = float(i)/float(numberofbonds)
            px, py = ratio
            if px > 0: 
                r = py/px
                x += [(1.-2.*p+r)/(r+1.)]
                y += [2.*(1.-p)-(1.-2.*p+r)/(r+1.)]
            else:
                # px = 0.
                x += [1.]
                y += [2.*(1.-p)-1.]
        end_time = time.time()        
        print('ratio = (%2.2f,%2.2f), time = %5.5fs'%(px, py, end_time-start_time))
    x, y, z = np.array(x), np.array(y), np.array(z)
    xi = np.linspace(0., 1., 100)
    yi = np.linspace(0., 1., 100)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    plt.figure()
    plt.plot(x, y, 'k.', ms=0.1)
    contour = plt.tricontourf(x, y, z, cmap="Oranges", levels=50, vmax=100, alpha=0.4)
    plt.colorbar(contour)
    plt.plot([0,1], [1,0], 'r-', lw=1)
    plt.xlim((0.0,1.0))
    plt.ylim((0.0,1.0))
    plt.xlabel(r'$v_{\parallel}$')
    plt.ylabel(r'$v_{\perp}$')
    # plt.gca().set_facecolor('lightgreen')
    phase_space = load_obj('phase_space')
    mat = plt.imshow(phase_space, cmap='Blues', vmax=50, interpolation='none', origin='Lower', extent=[0,1,0,1], alpha=0.79)
    plt.colorbar(mat)
    plt.tight_layout()
    print('total time = %5.5fs'%(time.time()-total_time)) 
    plt.show()