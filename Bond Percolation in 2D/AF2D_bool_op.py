# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:52:45 2019

@author: GIRISH S VISHWA
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time as time
import itertools as it
import matplotlib.animation as animation
import seaborn as sns
#import matplotlib
#matplotlib.use("Agg")
#import os


class Lattice:

    def __init__(self, L=5, tau=3, nux=0.5, nuy=0.45, delta=0.1, epsilon=1, speed=0, draw=False):
        self.L = L
        self.__nux = nux
        self.__nuy = nuy
        self.__delta = delta
        self.__epsilon = epsilon
        self.__speed = speed
        self.__tau = tau
        self.__draw = draw
 #       self.cell_grid = np.zeros((self.__L, self.__L))
        self.cell_grid = np.zeros((self.L, self.L)).astype(int)

        rand_matrix = np.random.rand(L,L)
                
        #Define all the UP bonds for every cell in the lattice
        self.up_bonds = (rand_matrix < self.__nuy)
        self.up_bonds[0,:] = 0 # Set boundary condition
        
        self.down_bonds = np.roll(self.up_bonds,-1,0)
        
        #Define all the RIGHT bonds for every cell in the lattice
        self.right_bonds = (rand_matrix < self.__nux)
        self.right_bonds[:,-1] = 0 #Set boundary condition
        
        self.left_bonds = np.roll(self.right_bonds,1,1)
        #Define a random fraction of dysfunctional cells in the lattice

        self.dysfunctional_cells = (rand_matrix < self.__delta)
        
        self.no_of_times_cell_excited = (np.zeros((self.L, self.L))).astype(int)
        
        self.pulsed_cells=np.zeros(self.L)
        self.prev_time_of_exc=np.zeros((self.L,self.L))
        self.new_time_of_exc=np.zeros((self.L,self.L))

        self.avg_list = np.array([])
        self.t = np.array([])
        self.AF = False
#        print(f'{self.up_bonds} \n') 
#        print(f'{self.down_bonds} \n')
#        print(f'{self.right_bonds} \n')
#        print(f'{self.left_bonds} \n')
#        print(self.dysfunctional_cells)


    
    def pulse(self):

#######TELL US WHICH CELLS WERE JUST PULSED. THESE CELLS ARE NEWLY EXCITED AND NOT EXCITABLE############
        self.pulsed_cells = np.logical_and(np.logical_not(self.dysfunctional_cells[:,0] * \
                      np.random.choice(2, size=self.L, p=[1-self.__epsilon,self.__epsilon])), (self.cell_grid[:,0]==0))# + self.cell_grid[:,0]
                

            

    def propagate(self, t, T):
        
        #This line will update the grid of cells each time to see which ones can be fired
        excitable_cells = (np.logical_not(self.dysfunctional_cells*np.random.choice(2,size=(self.L, self.L), p=[1-self.__epsilon,self.__epsilon]))) * (self.cell_grid==0) 
        excitable_cells[:,0] = np.logical_and(excitable_cells[:,0], np.logical_not(self.pulsed_cells))
        
        #This line will pick out all the cells on the grid that are currently excited
        #And thus ready to excite neighbouring cells. Note how cells that are just pulsed do not count here!
        excited_cells = (self.cell_grid==self.__tau)#.astype(int)
  
        new_excited_cells_right = np.logical_and(np.roll(np.logical_and(excited_cells, self.right_bonds), 1, 1), excitable_cells)
        new_excited_cells_left = np.logical_and(np.roll(np.logical_and(excited_cells, self.left_bonds), -1, 1), excitable_cells)
        new_excited_cells_up = np.logical_and(np.roll(np.logical_and(excited_cells, self.up_bonds), -1, 0), excitable_cells)
        new_excited_cells_down = np.logical_and(np.roll(np.logical_and(excited_cells, self.down_bonds), 1, 0), excitable_cells)
      
        new_excited_cells = np.logical_or.reduce((new_excited_cells_right,new_excited_cells_left,new_excited_cells_up,new_excited_cells_down))
        new_excited_cells[:,0] = np.logical_or(new_excited_cells[:,0], self.pulsed_cells)
        
        
        #Update the cell grid with the new excited cells
        self.cell_grid = (self.cell_grid - 1) * ((self.cell_grid - 1) > 0).astype(int) + self.__tau * new_excited_cells.astype(int)
        
        self.no_of_times_cell_excited += (new_excited_cells).astype(int)
        self.prev_time_of_exc[np.nonzero(new_excited_cells)] = self.new_time_of_exc[(new_excited_cells)]
        self.new_time_of_exc[np.nonzero(new_excited_cells)] = t
        
        time_diff=(self.new_time_of_exc-self.prev_time_of_exc)*(self.no_of_times_cell_excited > 1)

        if (time_diff[np.nonzero(time_diff)]).size==0:
            average_time_diff=0
        else:
            average_time_diff=np.average(time_diff[np.nonzero(time_diff)])
            
        if average_time_diff != 0:
            self.avg_list = np.append(self.avg_list, average_time_diff)
            self.t = np.append(self.t, t)
#            print(f't={t} \n')

#        print(f'Current cell grid [frame={t}]: \n {self.cell_grid}')
#        print(f'Time of last excitation: \n {self.new_time_of_exc}')
#        print(f'Time of previous excitation:\n {self.prev_time_of_exc} \n')
#        print(f'Time between excitations: \n {time_diff}')
#        print(f'Average time difference: \n {average_time_diff}')
#        print(f'No. of times each cell has been excited: \n {self.no_of_times_cell_excited} \n')            
            
            
        self.pulsed_cells=np.zeros(self.L)
        
        if self.__draw == False:
            if len(self.avg_list>=T):
                if len(np.argwhere(self.avg_list[-T:]<0.7*T))>=T:
                    print('WE ARE IN AF!!')
                    self.AF = True
                else:
                    self.AF = False
                return self.AF
        else:
            return self.cell_grid

         
        
    def send_wave(self, t, T):
        start_time = time.time()
        for i in it.islice(it.count(0), t):
            if i%T==0:
                self.pulse()                
            c=self.propagate(i, T)
            if c==True:
                break
        end_time = time.time()
        print(f'time taken: {end_time - start_time}')



L=200
tau=50
nux=1
nuy=0.12    
delta=0.05
epsilon=0.05
animate=True
T=220       

test=Lattice(L=L, tau=tau, nux=nux, nuy=nuy, delta=delta, epsilon=epsilon, draw=animate) 

if not animate:       
#    for x in range(7):
#        test=Lattice(L=L, tau=tau, nux=nux, nuy=nuy, delta=delta, epsilon=epsilon, draw=animate)        
#        test.send_wave(5000,T)
    test=Lattice(L=L, tau=tau, nux=nux, nuy=nuy, delta=delta, epsilon=epsilon, draw=animate) 
    test.send_wave(5000, T)
    plt.plot(test.t, test.avg_list, 'g')
    #plt.ylim(105,115)
    
    
if animate:
    fig, ax = plt.subplots()
    mat = ax.imshow(np.zeros((test.L, test.L)), cmap='bone', vmin=0, vmax=tau, interpolation='nearest')
    plt.title(f'nux={nux}, nuy={nuy}')
    #fig.colorbar(mat)
    
    def init():
        test.pulse()
        mat.set_data(test.cell_grid)
        return mat,
        
    def animate(i):
        #print(f'i={i}\n')
        if i%T==0:
            test.pulse()
            a=test.propagate(i, T)
            #print(f'This is cell grid pulse: {a} \n')
            mat.set_data(a)
        else:        
            a=test.propagate(i, T)
            #print(f'This is cell grid no pulse: {a} \n')
            mat.set_data(a)
        return mat, 
        
    ani=animation.FuncAnimation(fig, animate, frames=np.arange(0,5000,1), init_func=init, interval=0, repeat=False, blit=0)
    
    plt.show()   
    
save = False
if save:
#    Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Girish S Vishwa'), bitrate=1800) 
    
    ani.save('AF2D_8.mp4', writer=writer)

   

