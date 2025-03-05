"""
pyTransmon

Copyright (c) 2025 James Saslow
All rights reserved.

This software is proprietary but may be used for personal and educational purposes. 
Commercial use, modification, or distribution without prior written permission is prohibited.

For licensing inquiries, contact: jamessaslow@gmail.com or james.saslow@sjsu.edu
"""


# Importing Packages

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

from matplotlib.animation import FuncAnimation
import matplotlib
from ipywidgets import interact, widgets # type: ignore
from scipy.linalg import expm
from IPython.display import display, clear_output



class Transmon:
    
    # ===== Pauli Spin Matrices =====
    
    sigma_y = np.array([[0,-1j],
                        [1j,0]])
    
    sigma_z = np.array([[1,0],
                        [0,-1]])
    # ===============================   
    
    def __init__(self,t,psi,wq,W,mu,stdev,V0,delta,wd,frame):
        self.t     = t
        self.psi   = psi
        self.wq    = wq
        self.W     = W
        self.mu    = mu
        self.stdev = stdev
        self.V0    = V0
        self.delta = delta
        self.wd    = wd
        self.frame = frame
        
    def get(self, Frame):
        psi = self.psi

        if Frame == 'Lab':
            return psi

        # Check if 'RotatingFrame' is passed and is True
        if Frame == 'Qubit':  
            sigma_z = self.sigma_z
            wq      = self.wq
            t       = self.t
            transposed_psi_RF = np.transpose(psi)
            new_psi_transpose = np.zeros((len(t),2), dtype = complex)

            def U_RF(t):
                # return np.array([[np.exp(-1j*wq*t/2) , 0 ],
                #                  [0, np.exp(1j*wq*t/2)]])
                H0 = -wq*sigma_z/2
                return expm(1j*H0*t)

            
            for i in range(len(t)):
                new_psi_transpose[i] = np.matmul(U_RF(t[i]) , transposed_psi_RF[i])

            psi = np.transpose(new_psi_transpose)
            return psi

    # ============ Defining Function For AWG Voltage Pulse ===========    

    # Envelope Function
    def envelope(self,t):
        mu = self.mu
        stdev = self.stdev
        norm = 1/(stdev*np.sqrt(2*np.pi))
        arg1 = (t-mu)/stdev
        arg2 = -0.5*arg1**2
        return norm*np.exp(arg2)

    def voltage(self,t):
        envelope = self.envelope
        V0 = self.V0
        wd = self.wd
        delta = self.delta
    

        return V0 * envelope(t) * np.sin(wd*t + delta)

    # ================================================================
    
    
    def voltage_plot(self):
        t = self.t
        V = self.voltage(t)
        s = self.envelope(t)
        V0 = self.V0
        
        plt.figure(figsize = (6,4))
        plt.title('Voltage vs Time')
        plt.plot(t,V)
        plt.plot(t,V0*s, linestyle = '--',color = 'red', alpha = 0.4)
        plt.plot(t,-V0*s, linestyle = '--',color = 'red', alpha = 0.4)
        # plt.xlim(min(t),max(t))
        plt.ylim(-3.5,3.5)
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.show()
        
    
    def run(self):
        '''
        Running simulation of time evolution of the qubit 
        '''
        t     = self.t
        psi   = self.psi
        wq    = self.wq
        W     = self.W
        mu    = self.mu
        stdev = self.stdev
        V0    = self.V0
        delta = self.delta
        wd    = self.wd
        
        sigma_y = self.sigma_y
        sigma_z = self.sigma_z
        
        
        # ============ Defining Functions for the Hamiltonian ============
        
        # Qubit Hamiltonian
        def H0(t):
            return -wq/2 * sigma_z
        
        # Drive Hamiltonian
        def Hd(t):
            Vd = self.voltage(t)
            return W * Vd * sigma_y
        
        # Total Hamiltonian
        def H(t):
            return H0(t) + Hd(t)
        # =========================================================================
        
        
        # ====== Differential Equation Solver (Runge-Kutta 4th Order Method) ======
        def f(r,t):    
            M = -1j*H(t)
            v = M@r # Array of Velocities
            return v


        def RK4(f,t,init):
            N = len(t)
            dt = t[1] - t[0]
            r = [init]
            for i in range(N-1):
                k1 = dt * f(r[i],t[i])
                k2 = dt * f(r[i] + 0.5*k1, t[i] + 0.5*dt)
                k3 = dt * f(r[i] + 0.5*k2, t[i] + 0.5*dt)
                k4 = dt * f(r[i] + k3, t[i] + dt)
                r.append(r[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
            return np.transpose(r)
        # =========================================================================
        
        
        # =========== Solving the TDSE ============
        init = psi
        sol  = RK4(f,t,init)
        self.psi = sol
        #==========================================
        
    def prob_plot(self):
        '''
        Plots probabilities of |0> and |1> with respect to time
        '''
        
        t   = self.t
        # psi = self.psi

        frame = self.frame
        psi = self.get(frame)        

        a,b = psi
        prob_a = np.abs(a)**2
        prob_b = np.abs(b)**2
            
        plt.figure(figsize = (6,4))
        plt.title('Probability vs Time')
        plt.plot(t,prob_a, label = '$P_{|0>}$')
        plt.plot(t,prob_b, label = '$P_{|1>}$')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.show()
        
    
    def bloch_sphere_angles(self):
        # psi = self.psi
        frame = self.frame
        psi = self.get(frame)

        a,b = psi
        theta = 2*np.arctan( np.abs(b) / np.abs(a) )
        phi = np.angle(b) - np.angle(a)
        return theta, phi
    
    
    def get_states(self):
        theta,phi = self.bloch_sphere_angles()
        theta_i = theta[0]
        phi_i   = phi[0]
        
        a_i = np.round(np.cos(theta_i/2),3)
        b_i = np.round(np.sin(theta_i/2)*np.exp(1j*phi_i),3)
        
        psi_i = str(a_i) + '|0> + ' + str(b_i) + '|1>'
        
        theta_f = theta[-1]
        phi_f   = phi[-1]
    
        a_f = np.round(np.cos(theta_f/2),3)
        b_f = np.round(np.sin(theta_f/2)*np.exp(1j*phi_f),3)
    
        psi_f = str(a_f) + ' |0> + ' + str(b_f) + '|1>'
        
        return psi_i,psi_f
        
    def theta_plot(self):
        t   = self.t
        # psi = self.psi
        frame = self.frame
        psi = self.get(frame)

        theta,phi = self.bloch_sphere_angles()
        
        plt.figure(figsize = (6,4))
        plt.plot(t,theta)
        plt.title('$\u03B8$' + ' vs Time')
        plt.xlabel('Time')
        plt.ylabel('$\u03B8$')
        plt.show()
        
        
    def phi_plot(self):
        t = self.t
        # psi = self.psi
        theta,phi = self.bloch_sphere_angles()
        
        plt.figure(figsize = (6,4))
        plt.plot(t,phi)
        plt.title('\u03A6' + ' vs Time')
        plt.xlabel('Time')
        plt.ylabel('\u03A6')
        plt.show()
        
    
    def bloch_sphere_plot(self, **kwargs):
        
        # psi = self.psi
        theta, phi = self.bloch_sphere_angles()
        
        size = [16,12]
        if 'size' in kwargs:
            size = kwargs['size']
        size1,size2 = size
        
        fig = plt.figure(figsize=(size1, size2)) 
        ax = fig.add_subplot(111, projection='3d')

        # Data for a three-dimensional line
        xline = np.sin(theta)*np.cos(phi)
        yline = np.sin(theta)*np.sin(phi)
        zline = np.cos(theta)

        ax.plot3D(xline, yline, zline, 'black', linewidth = 4, alpha = 0.8)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Drawing the Bloch Sphere
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:40j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="r", alpha = 0.3)
    
        
        psi_i_label='$|'+'\u03C8'+'>'+'_i$'
        psi_f_label='$|'+'\u03C8'+'>'+'_f$'
        wavefunction_bool = kwargs.get('wavefunction',False)
        if wavefunction_bool == True:
            psi_i,psi_f = self.get_states()
            psi_i_label += ('= ' + psi_i)
            psi_f_label += ('= ' + psi_f)
        
        
        ax.scatter3D(xline[0], yline[0], zline[0], s = 155, c='yellow', label = psi_i_label)
        ax.scatter3D(xline[-1], yline[-1], zline[-1], s = 155, c='blue', label = psi_f_label)
        
        axes_bool = kwargs.get('axes',False)
        if axes_bool == True:
            ax.plot3D([-1, 1], [0, 0], [0, 0], linestyle = '--',color = 'black' ,alpha = 0.6 ,linewidth=2)  # X-axis
            ax.plot3D([0, 0], [-1, 1], [0, 0], linestyle = '--',color = 'black' ,alpha = 0.6,linewidth=2)  # Y-axis
            ax.plot3D([0, 0], [0, 0], [-1, 1], linestyle = '--',color = 'black' ,alpha = 0.6,linewidth=2)  # Z-axis
        
        labels_bool = kwargs.get('labels', False)
        if labels_bool == True:
            fontsize = 1.7*np.mean(size)
            ax.text(1.5, 0, 0, "$|+>$", color='green', fontsize=fontsize, weight = 'bold',fontfamily='DejaVu Sans')
            ax.text(0, 0, 1.4, "$|0>$", color='green', fontsize=fontsize, weight = 'bold',fontfamily='DejaVu Sans')
            ax.text(0, 1.4, 0, "$|+i>$", color='green', fontsize=fontsize, weight = 'bold',fontfamily='DejaVu Sans')
        
        legend_bool = kwargs.get('legend', False)
        if legend_bool == True:
            ax.legend()
            
        ax.view_init(elev=20, azim=45)
        plt.axis('off')
        plt.show()


