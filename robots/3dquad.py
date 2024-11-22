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
        self.robot_spec = robot_spec # not used in this model
        self.df_dx = np.vstack([np.hstack([np.zeros(3), np.eye(3), np.zeros(3)]), np.zeros([6,9])])
      
        # for exp (CBF for unicycle)
        self.m = 1
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

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1,1)) # goal state

        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.05)
        theta_d = np.arctan2(G[1,0]-X[1,0],G[0,0]-X[0,0])
        error_theta = angle_normalize( theta_d - X[2,0] )

        omega = k_omega * error_theta   
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = k_v*( distance )*np.cos( error_theta )

        return np.array([v, omega]).reshape(-1,1)
    
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
        obsX = obs[0:3]
        d_min = obs[3][0] + robot_radius # obs radius + robot radius

        phi = X[6,0]
        theta = X[7,0]
        psi = X[8,0]

        h = np.linalg.norm( X[0:3] - obsX )**2 - beta*d_min**2   
        s = ( X[0:2] - obsX[0:2]).T @ np.array( [np.cos(theta),np.sin(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        
        der_sigma = self.sigma_der(s)
        # [dh/dx, dh/dy, dh/dtheta]^T
        dh_dx = np.append( 
                    2*( X[0:2] - obsX[0:2] ).T - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta)] ]) ),
                    - der_sigma * ( -np.sin(theta)*( X[0,0]-obsX[0,0] ) + np.cos(theta)*( X[1,0] - obsX[1,0] ) ),
                     axis=1)
        # print(h)
        # print(dh_dx)
        return h, dh_dx
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, obs, robot_radius, beta = 1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            z_obs = obs[2]
            r_obs = obs[3]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 + (x[2, 0] - z_obs)**2 - beta*d_min**2
            return h

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        return h_k, d_h, dd_h