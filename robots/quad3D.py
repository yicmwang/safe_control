import numpy as np
import casadi as ca

"""
Created on July 14h, 2024
@author: Taekyung Kim, Yichen Wang

@description: 
3D Quad model for CBF-QP and MPC-CBF (casadi)
"""

def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

class Quad3D:
    
    def __init__(self, dt, robot_spec):
        '''
            X: [px, py, pz, vx, vy, vz, phi, theta, psi]
            U: [f, phi_dot, theta_dot, psi_dot]
            system parameters: m
            cbf: h(x) = ||x(:,0:3)-x_obs||^2 - beta*d_min^2 - sigma(s)
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        if 'phi_dot_max' not in self.robot_spec:
            self.robot_spec['phi_dot_max'] = 1.0
        if 'theta_dot_max' not in self.robot_spec:
            self.robot_spec['theta_dot_max'] = 1.0
        if 'psi_dot_max' not in self.robot_spec:
            self.robot_spec['psi_dot_max'] = 1.0
        if 'f_max' not in self.robot_spec:
            self.robot_spec['f_max'] = 1.0
        
        if 'mass' not in self.robot_spec:
            self.robot_spec['mass'] = 1.0
        self.df_dx = np.vstack([np.hstack([np.zeros([3,3]), np.eye(3),np.zeros([3,3])]),np.zeros([6,9])])
      
        # for exp
        self.m = self.robot_spec['mass']
        self.gravity = 9.8

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat([
                X[3],
                X[4],
                X[5],
                0,
                0,
                self.gravity,
                0,
                0,
                0,
            ])
        else:
            return np.array([
                X[3],
                X[4],
                X[5],
                0,
                0,
                self.gravity,
                0,
                0,
                0,
            ]).reshape(-1,1)
    
    def g(self, X, casadi=False):
        if casadi:
            g = ca.SX.zeros(9, 4)
            g[4, 1] = -ca.sin(X[7]) / self.m
            g[5, 1] = ca.cos(X[7]) * ca.sin(X[6]) / self.m
            g[6, 1] = -ca.cos(X[7]) * ca.cos(X[6]) / self.m
            g[6, 2] = 1
            g[7, 3] = 1
            g[8, 4] = 1
            return g
        else:
            g = np.zeros(9, 4)
            g[4, 1] = -np.sin(X[7]) / self.m
            g[5, 1] = np.cos(X[7]) * np.sin(6) / self.m
            g[6, 1] = -np.cos(X[7]) * np.cos(6) / self.m
            g[6, 2] = 1
            g[7, 3] = 1
            g[8, 4] = 1
            return g
    def step(self, X, U): 
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        X[2,0] = angle_normalize(X[2,0])
        return X

    def nominal_input(self, X, goal, d_min = 0.05, k_ang = 2.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1,1)) # goal state
        phi_dot_max = self.robot_spec['phi_dot_max']
        theta_dot_max = self.robot_spec['theta_dot_max']
        psi_dot_max = self.robot_spec['psi_dot_max']
        f_max = self.robot_spec['f_max']
        
        
        u_nom = np.zeros(4,1)

        x_err = X[0:3] - goal[0:3]
        F_des = x_err * k_v + np.array([0, 0, 9.8 * self.m]).reshape(-1,1) #proportional control & gravity compensation
        u_nom[0] = min(np.linalg.norm(F_des), f_max)
        a_des = F_des / u_nom[0]
        theta_des = np.asin(-1 * a_des[0])
        phi_des = np.asin(a_des[1] / np.sin(theta_des))
        u_nom[1] = min((phi_des - X[6]) * k_ang, phi_dot_max)
        u_nom[2] = min((theta_des - X[7]) * k_ang, theta_dot_max)
        u_nom[3] = min(-1 * X[8] * k_ang, psi_dot_max)
        return u_nom
    
    def stop(self, X, k_stop = 1):
        u_stop = np.zeros(4,1)

        v_curr = X[3:6]
        F_des = v_curr * k_stop + np.array([0, 0, 9.8 * self.m]).reshape(-1,1) #proportional control & gravity compensation
        u_stop[0] = np.linalg.norm(F_des)
        a_des = F_des / u_stop[0]
        theta_des = np.asin(-1 * a_des[0])
        phi_des = np.asin(a_des[1] / np.sin(theta_des))
        u_stop[1] = (phi_des - X[6]) * k_stop
        u_stop[2] = (theta_des - X[7]) * k_stop
        u_stop[3] = -1 * X[8] * k_stop
        return u_stop
    
    def has_stopped(self, X, tol = 0.05):
        return np.linalg.norm(X[3:6]) < tol

    def rotate_to(self, X, ang_des, k_omega = 2.0):
        u = np.zeros(4,1)
        u[1] = (ang_des[0] - X[6]) * k_omega
        u[2] = (ang_des[1] - X[7]) * k_omega
        u[3] = (ang_des[2] - X[8]) * k_omega
        return u
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''obs: [x, y, r]'''
        '''obstacles are infinite cylinders at x and y'''
        '''X : [x y z vx vy yz phi theta psi]'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        dh_dot_dx = np.hstack([(2 * self.f(X)[0:2]).T, 0,
                               2 * (X[0:2] - obsX[0:2]).T,
                               np.zeros([1,4])])
        return h, h_dot, dh_dot_dx
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, obs, robot_radius, beta = 1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        return h_k, d_h, dd_h