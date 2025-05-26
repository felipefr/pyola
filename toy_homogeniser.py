#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:16:37 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:53:45 2025

@author: felipe
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import copy
from toy_solver import * 
# from numba import njit

class TrussLocalCanonicalIntegrator:
    def __init__(self, A, material, V):
        self.A = A
        self.material = material
        self.element = V.element
        self.V = V
        self.k = 0
        self.l = 0
    
    def set_kl(self, k, l):
        self.k, self.l = k, l
        
    def compute(self, X, u, uold, e):
        A = self.A[e]
        a, b, lmbda, lmbda_old, L0, Bmat = self.element.get_truss_kinematics(X, u, uold)
        
        stress, dtang = self.material(lmbda, lmbda_old)
        
        V = L0*A  # Volume

        # Internal force vector (2D)
        D = dtang * np.outer(b,b) 
        D += stress*(np.eye(2) - np.outer(b,b))/lmbda
        

        # Tangent stiffness matrix (4x4)
        K = V * Bmat.T @ D @ Bmat
        f_int = - V * a[self.l] * Bmat.T @ D[:, self.k] 
        
        return K, f_int


class MicroModel:
    def __init__(self, mesh, param):
        self.mesh = mesh

        if(param['model'] == 'truss'):
            self.material = LinearEngStrainTrussMaterial(E=param['E'])
            self.material_can = LinearEngStrainTrussMaterial(E=param['E'])
            
        elif(param['model'] == 'cable'):
            self.material = LinearEngStrainCableMaterial(E=param['E'], eta = param['eta'])
            self.material_can = LinearEngStrainCableMaterial(E=param['E'], eta = 0.0)
        
        self.U = FunctionSpace(self.mesh, TrussElement())
        self.dh = DOFHandler(self.mesh)
        self.dh.add_space(self.U, name = 'displacement')
        self.form = TrussLocalIntegrator(param['A'], self.material, self.U)
        self.form_can = TrussLocalCanonicalIntegrator(param['A'], self.material, self.U)
        
        self.ngamma_nodes = len(mesh.bnd_nodes)
        uD = np.zeros((self.ngamma_nodes, 2))
        self.bcs = [DirichletBC(mesh.bnd_nodes, [0, 1], uD)]
        
        self.yG = np.array([0.5,0.5])
        self.vol = 1.0

    def get_ufixed(self, G):
        uD = np.zeros_like(self.bcs[0].value)
        for i, j in enumerate(self.mesh.bnd_nodes):
            uD[i,:] = G@(self.mesh.X[j,:] - self.yG)
            
        return uD
    
    
    def homogeniseP(self, G, u0 = None):        
        self.bcs[0].value = self.get_ufixed(G)
        u0 = Function(self.U) if type(u0) == type(None) else copy.deepcopy(u0)
        forces = np.zeros_like(u0)
        u =  solve_nonlinear(self.mesh, self.U, self.dh, self.form, forces, self.bcs, uold = u0, tol = 1e-8)
        P = self.homogeniseP_given_disp(u)
        
        return P, u
    
    
    def homogeniseP_given_disp(self, u):
        P = np.zeros((2,2))
        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = self.dh.get_cell_dofs(c)[0] # only for the first space (U)
            uL = u.array[cell_dofs]
        
            A = self.form.A[c]
            a, b, lmbda, lmbda_old, L0, Bmat = self.form.element.get_truss_kinematics(X.flatten(), uL, uL) # last argument is dummy
            stress, dtang = self.material(lmbda, lmbda_old)
            
            V = L0*A  # Volume            
            P += V*stress*np.outer(b,a)
        
        P = P/self.vol
        
        return P
    
    def homogenise_tang_ffd(self, G, tau = 1e-7):
        
        P_ref, u_ref = self.homogeniseP(G)
        P_ref = P_ref.flatten()
        Gref = G.flatten()
        n = len(Gref) 
        base_canonic = np.eye(n)
        Atang = np.zeros((n,n))
        
        for j in range(n):
            Gp = (Gref + tau*base_canonic[j,:]).reshape((int(n/2),int(n/2)))
            Pp  = self.homogeniseP(Gp, u0 = u_ref)[0].flatten()
            Atang[:,j] = (Pp - P_ref)/tau 
        
        return Atang
        
    def homogeniseC(self, G, u0 = None):
        self.bcs[0].value = self.get_ufixed(G)
        u0 = Function(self.U) if type(u0) == type(None) else copy.deepcopy(u0)
        forces = np.zeros_like(u0)
        u =  solve_nonlinear(self.mesh, self.U, self.dh, self.form, forces, self.bcs, uold = u0, tol = 1e-8)
        
        ass = Assembler(self.mesh, self.dh)
        
        self.bcs[0].value[:,:] = 0.0
        ukl_list = []
        for k in range(2):
            for l in range(2):
                self.form_can.set_kl(k, l)
                K, F_kl = ass.assemble(self.form_can, u, u)
                for bc in self.bcs:
                    bc.apply(K,F_kl)
                
                K.tocsr()
                ukl_list.append(spla.spsolve(K, F_kl))
    
        C = np.zeros((4,4))
        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = self.dh.get_cell_dofs(c)[0] # only for the first space (U)
            uL = u.array[cell_dofs]
            
            A = self.form.A[c]
            a, b, lmbda, lmbda_old, L0, Bmat = self.form.element.get_truss_kinematics(X.flatten(), uL, uL) # last argument is dummy
            stress, dtang = self.material(lmbda, lmbda_old)
            
            D = dtang * np.outer(b,b) 
            D += stress*(np.eye(2) - np.outer(b,b))/lmbda
            
            V = L0*A  # Volume        
            
            # Cbar (note that reshape respect the lexigraphic order)
            C += V*np.einsum("ik,j,l->ijkl", D, a, a).reshape((4,4)) 
            
            for kl in range(4):
                uklL = ukl_list[kl][cell_dofs]
                C[:,kl] += V* np.outer(D@Bmat@uklL,a).flatten() 
        
        return C
        
   
