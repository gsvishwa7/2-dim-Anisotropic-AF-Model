import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import pickle
import os
path = 'C:\\Users\\f18ho\\Google Drive\\UROP_2019'
os.chdir(path)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

view_from = (1.0,1.5,2.0)
def project(point, view_point, t_lat, t_long):
    ix, iy, iz = point
    px, py, pz = view_point
    t_lat, t_long = math.radians(t_lat), math.radians(t_long)
    longitude = math.atan2(px,py)
    angle_xy = longitude + t_long - 0.5 * math.pi

    xy = math.sqrt(px*px+py*py)
    
    latitude = math.atan2(xy,pz)
    angle_xyz = latitude + t_lat - 0.5 * math.pi

    if ix == 0 and iy == 0: xi = 0
    else: xi = math.sqrt(ix*ix+iy*iy) * math.cos(math.atan2(ix,iy)-longitude)

    angle_e = longitude + t_long
    l_xy = (px-(py-px*math.tan(angle_xy)-iy+ix*math.tan(angle_e))/(math.tan(angle_e)-math.tan(angle_xy)))/math.cos(angle_xy)

    angle_e = latitude + t_lat
    l_xyz = (xy-(pz-xy*math.tan(angle_xyz)-iz+xi*math.tan(angle_e))/(math.tan(angle_e)-math.tan(angle_xyz)))/math.cos(angle_xyz)
    
    return (l_xy,l_xyz)        

class Lattice:
    def __init__(self, L, draw = False):
        Lx, Ly, Lz = L
        self.__Lx = Lx
        self.__Ly = Ly
        self.__Lz = Lz
        self.__bonds = [[(i,j,k),(i+1,j,k)] for i in range(self.__Lx-1) for j in range(self.__Ly) for k in range(self.__Lz)] + \
                       [[(i,j,k),(i,j+1,k)] for i in range(self.__Lx) for j in range(self.__Ly-1) for k in range(self.__Lz)] + \
                       [[(i,j,k),(i,j,k+1)] for i in range(self.__Lx) for j in range(self.__Ly) for k in range(self.__Lz-1)]
        self.__ptr = {}
        self.__draw = draw
        self.__cluster = 0
        
        if self.__draw:
            self.__fig = plt.figure()
            self.__axes = plt.gca()
            for b in self.__bonds:
                x1, y1, z1 = b[0]
                x2, y2, z2 = b[1]
                x1, y1 = project((x1,y1,z1),view_from,0,0)
                x2, y2 = project((x2,y2,z2),view_from,0,0)
                line = plt.Line2D((x1,x2), (y1,y2), lw=0.1, color='k')
                self.__axes.add_line(line)
            plt.axis('equal')
            self.__fig.canvas.draw()

    def filter_bonds(self, ratio, seed=None):
        px, py = ratio
        pz = py
        np.random.seed(seed)
        mx = (self.__Lx-1)*self.__Ly*self.__Lz
        my = self.__Lx*(self.__Ly-1)*self.__Lz
        mz = self.__Lx*self.__Ly*(self.__Lz-1)
        if pz < 1:
            n = mz*pz
            n_list = list(range(mx+my,mx+my+mz,1))
            n_list = np.random.choice(n_list, size=n, replace=False)
            n_list = np.flip(np.sort(n_list))
            for n in n_list: del self.__bonds[n]
        if py < 1:
            n = my*py
            n_list = list(range(mx,mx+my,1))
            n_list = np.random.choice(n_list, size=n, replace=False)
            n_list = np.flip(np.sort(n_list))
            for n in n_list: del self.__bonds[n]
        if px < 1:
            n = mx*px
            n_list = list(range(0,mx,1))
            n_list = np.random.choice(n_list, size=n, replace=False)
            n_list = np.flip(np.sort(n_list))
            for n in n_list: del self.__bonds[n]

    def get_numberofbonds(self):
        return len(self.__bonds)

    def randomise_bonds(self, seed=None):
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
                    c1, x1min, x1max, y1min, y1max, z1min, z1max = tuple(self.__ptr[r1])
                    c2, x2min, x2max, y2min, y2max, z2min, z2max = tuple(self.__ptr[r2])
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
                    if z1min < z2min:
                        zmin = z1min
                    else:
                        zmin = z2min
                    if z1max > z2max:
                        zmax = z1max
                    else:
                        zmax = z2max
                    c = xmax-xmin+ymax-ymin+zmax-zmin
                    c = [c,xmin,xmax,ymin,ymax,zmin,zmax]
                    if c2 > c1:
                        self.__ptr[r2] = c
                        self.__ptr[r1] = r2
                    else:
                        self.__ptr[r1] = c
                        self.__ptr[r2] = r1
                else:
                    c, xmin, xmax, ymin, ymax, zmin, zmax = tuple(self.__ptr[r1])
                    c = [c,xmin,xmax,ymin,ymax,zmin,zmax]
            else:
                r1 = self.findroot(s1)
                self.__ptr.update({s2:r1})
                c, xmin, xmax, ymin, ymax, zmin, zmax = tuple(self.__ptr[r1])
                x, y, z = s2
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
                if z < zmin: zmin = z
                if z > zmax: zmax = z
                c = xmax-xmin+ymax-ymin+zmax-zmin
                c = [c,xmin,xmax,ymin,ymax,zmin,zmax]
                self.__ptr[r1] = c
        else:
            if s2 in self.__ptr:
                r2 = self.findroot(s2)
                self.__ptr.update({s1:r2})
                c, xmin, xmax, ymin, ymax, zmin, zmax = tuple(self.__ptr[r2])
                x, y, z = s1
                if x < xmin: xmin = x
                if x > xmax: xmax = x
                if y < ymin: ymin = y
                if y > ymax: ymax = y
                if z < zmin: zmin = z
                if z > zmax: zmax = z
                c = xmax-xmin+ymax-ymin+zmax-zmin
                c = [c,xmin,xmax,ymin,ymax,zmin,zmax]
                self.__ptr[r2] = c
            else:
                x1, y1, z1 = s1
                x2, y2, z2 = s2
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
                if z1 < z2:
                    zmin = z1
                    zmax = z2
                else:
                    zmin = z2
                    zmax = z1
                c = xmax-xmin+ymax-ymin+zmax-zmin
                c = [c,xmin,xmax,ymin,ymax,zmin,zmax]
                self.__ptr.update({s1:c})
                self.__ptr.update({s2:s1})
        # draw bond
        if self.__draw:
            x1, y1, z1 = s1
            x2, y2, z2 = s2
            x1, y1 = project((x1,y1,z1),view_from,0,0)
            x2, y2 = project((x2,y2,z2),view_from,0,0)
            line = plt.Line2D((x1,x2), (y1,y2), lw=1.0, color='k')
            self.__axes.add_line(line)
            self.__fig.canvas.draw()                
        # check for percolation
        if xmin==0 and xmax==self.__Lx-1 or ymin==0 and ymax==self.__Ly-1 or zmin==0 and zmax==self.__Lz-1:
            percolated = True
        else:
            percolated = False
        # get size of largest cluster
        if c[0] > self.__cluster: self.__cluster = c[0]          
        return percolated, self.__cluster

visualise = False
if visualise:
    lattice = Lattice(L=20, draw=False)
    lattice.filter_bonds((1,1), seed=2)
    lattice.randomise_bonds(seed=2)
    numberofbonds = lattice.get_numberofbonds()
    i = 0
    percolate = False
    hulls = []
    for i in range(numberofbonds):
        percolated, hull = lattice.add_bond(i)
        if hull > 0: hulls += [hull]
        i += 1
        print('%d/%d'%(i,numberofbonds))
    print(hulls)
    plt.show()

L = (200,200,25)
Lx, Ly, Lz = L
runs_per_L = 1000
p_list = list(range(1,(Lx-1)*Ly*Lz+Lx*(Ly-1)*Lz+Lx*Ly*(Lz-1),1))
ratios = [(1.0,1.0), \
          (1.0,0.9),(1.0,0.8),(1.0,0.7),(1.0,0.6),(1.0,0.5),(1.0,0.4),(1.0,0.3),(1.0,0.2),(1.0,0.1),(1.0,0.0), \
          (0.9,1.0),(0.8,1.0),(0.7,1.0),(0.6,1.0),(0.5,1.0),(0.4,1.0),(0.3,1.0),(0.2,1.0),(0.1,1.0),(0.0,1.0)]

generate = True
if generate:
    total_time = time.time()
    for ratio in ratios:
        px, py = ratio
        start_time = time.time()
        p = {i:[] for i in p_list}
        for run in range(runs_per_L):
            start = time.time()
            lattice = Lattice(L, draw=False)
            lattice.filter_bonds(ratio, seed=run)
            lattice.randomise_bonds(seed=run)
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
        save_obj(p, 'BP3D, ratio=(%2.2f,%2.2f)'%(px,py))
        end_time = time.time()
        print('ratio = (%2.2f,%2.2f), time = %5.5fs'%(px, py, end_time-start_time))
    print('total time = %5.5fs'%(time.time()-total_time))

def find_nearest(array, value):
    array = np.asarray(array)
    i = (np.abs(array-value)).argmin()
    return i

p_list = list(range(1,(Lx-1)*Ly*Lz+Lx*(Ly-1)*Lz+Lx*Ly*(Lz-1),1))

analyse = False
if analyse:
    total_time = time.time()
    numberofbonds = (Lx-1)*Ly*Lz+Lx*(Ly-1)*Lz+Lx*Ly*(Lz-1)
    clusters = {}
    for ratio in ratios:
        start_time = time.time()
        px, py = ratio
        p = load_obj('BP3D, ratio=(%2.2f,%2.2f)'%(px,py))
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
            z += [clusters[ratio][i]]
            p = float(i)/float(numberofbonds)
            px, py = ratio
            if px > 0: 
                r = py/px
                y += [(1.-3.*p+2*r)/(2*r+1.)]
                x += [0.5*(3.*(1.-p)-(1.-3.*p+2*r)/(2*r+1.))]
            else:
                # px = 0.
                y += [1.]
                x += [0.5*(3.*(1.-p)-1.)]
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
    contour = plt.tricontourf(x, y, z, cmap="seismic", levels=50, vmax=100)
    plt.colorbar(contour)
    plt.plot([0,0.5], [1,0], 'r-', lw=1)
    plt.xlim((0.0,1.0))
    plt.ylim((0.0,1.0))
    plt.ylabel(r'$v_{\parallel}$')
    plt.xlabel(r'$v_{\perp}$')
    plt.gca().set_facecolor('lightgreen')
    plt.tight_layout()
    print('total time = %5.5fs'%(time.time()-total_time)) 
    plt.show()    