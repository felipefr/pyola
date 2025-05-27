#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:19:13 2025

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:32:32 2025

@author: frocha
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import copy 
# from numba import njit

# Element Stiffness and Internal Force
# ----------------------------------------

# @njit



class Mesh:
    def __init__(self, X, cells, param=None):
        self.X = X
        self.cells = cells
        self.param = param
        self.n_cells = len(self.cells)
        self.n_nodes = len(self.X)
        self.ndim = self.X.shape[1]
        self.bnd_nodes = []
        
    def mark_boundary_nodes(self, tol = 1e-10):
        x_min, y_min = self.X.min(axis=0)
        x_max, y_max = self.X.max(axis=0)
        
        self.bnd_nodes = []
        
        for i, x in enumerate(self.X):
            if(np.abs(x[0]-x_min)<tol or 
               np.abs(x[1]-y_min)<tol or 
               np.abs(x[0]-x_max)<tol or
               np.abs(x[1]-y_max)<tol):
                  
                  self.bnd_nodes.append(i)

        self.bnd_nodes = np.array(self.bnd_nodes, dtype = 'int')
        return self.bnd_nodes



class TrussElement:
    def __init__(self):
        self.n_dofs = 2
        self.B_matrix = np.array([[-1,0,1,0], [0,-1,0,1]])
        
    def get_truss_kinematics(self, X, u, uold):
        a = self.B_matrix@X
        L0 = np.linalg.norm(a) # underformed length
        a = a/L0 # unitary underformed truss vector 
        Bmat = self.B_matrix/L0 # discrete gradient operator
        
        q = a + Bmat@u # deformed truss vector (stretch lenght)  
        lmbda = np.linalg.norm(q) # L/L0
        b = q/lmbda # unitary deformed truss vector  
        
        lmbda_old = np.linalg.norm(a + self.B_matrix@uold) 
        
        return a, b, lmbda, lmbda_old, L0, Bmat

class FunctionSpace:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.n_dofs = element.n_dofs * mesh.n_nodes

    def get_dofs_for_cell(self, c):
        n1, n2 = self.mesh.cells[c]
        dofs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])
        return dofs

class Function:
    def __init__(self, V):
        self.functionspace = V
        self.array = np.zeros(V.n_dofs)

class LinearEngStrainTrussMaterial:
    def __init__(self, E):
        self.E = E
        
    def __call__(self, lmbda, lmbda_old):
        strain = lmbda - 1
        stress = self.E * strain 
        dtang = self.E
        return stress, dtang


class LinearEngStrainCableMaterial:
    def __init__(self, E, eta = 0.0):
        self.E = E
        self.eta = eta
        
    def __call__(self, lmbda, lmbda_old):
        strain = lmbda - 1
        stress = self.E * strain if strain > 0.0 else 0.0 
        dtang = self.E if strain > 0.0 else 0.0
        
        stress = stress + self.eta*(lmbda - lmbda_old)
        dtang = dtang + lmbda_old
        return stress, dtang

    
class TrussLocalIntegrator:
    def __init__(self, A, material, V):
        self.A = A
        self.material = material
        self.element = V.element
        self.V = V
        
    def compute(self, X, u, uold, e):
        A = self.A[e]
        a, b, lmbda, lmbda_old, L0, Bmat = self.element.get_truss_kinematics(X, u, uold)
        
        stress, dtang = self.material(lmbda, lmbda_old)
        
        V = L0*A  # Volume

        # Internal force vector (2D)
        F_int = V * stress * (Bmat.T @ b)
        D_mat = dtang * np.outer(b,b) 
        D_geo = stress*(np.eye(2) - np.outer(b,b))/lmbda

        # Tangent stiffness matrix (4x4)
        K = V * Bmat.T@(D_mat + D_geo)@Bmat
        
        return K, F_int
    
    
class DOFHandler:
    def __init__(self, mesh):
        self.mesh = mesh
        self.function_spaces = []
        self.offsets = []
        self.total_dofs = 0

    def add_space(self, space, name=None):
        self.offsets.append(self.total_dofs)
        self.function_spaces.append((name, space))
        self.total_dofs += space.n_dofs

    def finalize(self):
        pass

    def get_cell_dofs(self, cell_id):
        dofs = []
        for name, fs in self.function_spaces:
            local = fs.get_dofs_for_cell(cell_id)
            global_dofs = self.offsets[self.function_spaces.index((name, fs))] + local
            dofs.append(global_dofs)
        return dofs
    
class Assembler:
    def __init__(self, mesh, dof_handler):
        self.mesh = mesh
        self.dh = dof_handler
        self.ndofs = self.dh.total_dofs

    def assemble(self, integrator, u, uold = None):
        rows, cols, data = [], [], []
        F = np.zeros(self.ndofs)
    
        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = self.dh.get_cell_dofs(c)[0] # only for the first space (U)
            Ke, fe = integrator.compute(X.flatten(), u.array[cell_dofs], uold.array[cell_dofs], c)
            F[cell_dofs]+= fe
            for i, gi in enumerate(cell_dofs):
                for j, gj in enumerate(cell_dofs):
                    rows.append(gi)
                    cols.append(gj)
                    data.append(Ke[i, j])
                    
        K = sp.coo_matrix((data, (rows, cols)), shape=(self.ndofs, self.ndofs)).tolil()
        return K, F

class DirichletBC:
    def __init__(self, nodes, ldofs, value):
        self.nodes = nodes 
        self.ldofs = ldofs
        self.value = value
        self.n_ldofs = len(ldofs)
        self.dofmap = np.array([ [i*self.n_ldofs + j for j in self.ldofs] for i in self.nodes]).flatten()

    def apply(self, A, b):
        A[self.dofmap,:] = 0.0
        A[self.dofmap,self.dofmap] = 1.0
        b[self.dofmap] = self.value.flatten()       

    def homogenise(self):
        return DirichletBC(self.node, self.ldof, np.zeros_like(self.value))


# ----------------------------------------
# Linear Solver
# ----------------------------------------
def solve_linear(mesh, U, dh, form, forces, bcs):

    ass = Assembler(mesh, dh)
    
    u0 = Function(U)
    K, F = ass.assemble(form, u0, u0) # u0 is dummy
    F+=forces
    for bc in bcs: 
        bc.apply(K,F)
    
    K = K.tocsr()
    u = spla.spsolve(K, F)
    
    return u


# ----------------------------------------
# Newton-Raphson Solver
# ----------------------------------------
def solve_nonlinear(mesh, U, dh, form, forces, bcs, uold, tol=1e-13, max_iter=50, 
                    omega = 1.0, log = True):
    u = Function(U)
    u.array[:] = uold.array[:]
    
    ass = Assembler(mesh, dh)
    
    for k in range(max_iter):
        K, F_int = ass.assemble(form, u, uold)
        b = omega*(forces - F_int) + K@u.array[:]
        
        for bc in bcs:
            bc.apply(K,b)
        
        K = K.tocsr()
        b = np.array(b).astype(np.float64)

        uold.array[:] = u.array[:] 
        u.array[:] = spla.spsolve(K, b)
        norm_du = np.linalg.norm(u.array - uold.array)/np.linalg.norm(u.array)
        # norm_res = np.linalg.norm(forces-F_int)        

        if(log): print(f"Iter {k:2d}: increment = {norm_du:.3e}")
        if norm_du < tol:
            break
        
    return u




# plotting
def plot_truss(mesh, u, scale=1.0, show_nodes=True):
    """
    Plot undeformed and deformed truss structure.

    Parameters:
        coords (ndarray): Original coordinates (n_nodes x 2)
        elements (ndarray): Element connectivity (n_elements x 2)
        u (ndarray): Global displacement vector (2*n_nodes)
        scale (float): Scale factor for displacements
        show_nodes (bool): Show node indices
    """
    n_nodes = mesh.X.shape[0]
    u_nodes = u.reshape((n_nodes, 2))
    coords_def = mesh.X + scale * u_nodes

    plt.figure(figsize=(8, 6))
    for e in mesh.cells:
        n1, n2 = e
        x_orig = mesh.X[[n1, n2]]
        x_def = coords_def[[n1, n2]]

        # Undeformed (dashed gray)
        plt.plot(x_orig[:, 0], x_orig[:, 1], 'k--', lw=1, alpha=0.5)

        # Deformed (solid red)
        plt.plot(x_def[:, 0], x_def[:, 1], 'r-', lw=2)

    if show_nodes:
        for i, (x, y) in enumerate(mesh.X):
            plt.text(x, y, f'{i}', color='blue', fontsize=10)

    plt.axis('equal')
    plt.grid(True)
    plt.title("Truss: Undeformed (black dashed) and Deformed (red)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    



        

if __name__ == "__main__":
    # Node coordinates
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ])
    # Elements (0-based indexing)
    elements = np.array([
        [0, 2],
        [1, 2],
        [0, 1],
    ])
    
    mesh = Mesh(coords, elements)

    mat = LinearEngStrainTrussMaterial(E=210e9)

    U = FunctionSpace(mesh, TrussElement())
    dh = DOFHandler(mesh)
    dh.add_space(U, name = 'displacement')
    
    A = np.array([1e-4, 1e-4, 1e-4])
    
    form = TrussLocalIntegrator(A, mat, U)
    
    bcs = [DirichletBC([0,1], [0,1], np.array([[0.0,0.0],[0.01,0.0]])) ]

    forces = np.zeros(U.n_dofs)
    forces[5] = -1e6  # Load at node 2, y-direction
   
    u = solve_linear(mesh, U, dh, form, forces, bcs)
    
    u0 = Function(U) # zero by default
    
    u = solve_nonlinear(mesh, U, dh, form, forces, bcs, uold = u0, tol = 1e-8)
    
    plot_truss(mesh, u.array, scale=10.0)
