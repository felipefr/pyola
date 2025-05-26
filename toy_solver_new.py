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
        
    def get_truss_kinematics(self, X, u):
        a = self.B_matrix@X
        L0 = np.linalg.norm(a) # underformed length
        a = a/L0 # unitary underformed truss vector 
        Bmat = self.B_matrix/L0 # discrete gradient operator
        
        q = a + Bmat@u # deformed truss vector (stretch lenght)  
        lmbda = np.linalg.norm(q) # L/L0
        b = q/lmbda # unitary deformed truss vector  
        
        return a, b, lmbda, L0, Bmat

class FunctionSpace:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.n_dofs = element.n_dofs * mesh.n_nodes

    def dof_map(self, cell_idx):
        pass


class LinearEngStrainMaterial:
    def __init__(self, E):
        self.E = E
    def __call__(self, lmbda):
        strain = lmbda - 1
        stress = self.E * strain 
        dtang = self.E
        return stress, dtang
    
class TrussLocalIntegrator:
    def __init__(self, A, material, V):
        self.A = A
        self.material = material
        self.element = V.element
        
    def compute(self, X, u, e):
        A = self.A[e]
        a, b, lmbda, L0, Bmat = self.element.get_truss_kinematics(X, u)
        
        stress, dtang = self.material(lmbda)
        
        V = L0*A  # Volume

        # Internal force vector (2D)
        f_int = V * stress * (Bmat.T @ b)
        D_mat = dtang * np.outer(b,b) 
        D_geo = stress*(np.eye(2) - np.outer(b,b))/lmbda

        # Tangent stiffness matrix (4x4)
        K = V * Bmat.T@(D_mat + D_geo)@Bmat
        
        return K, f_int
    
    
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

    def assemble(self, integrator, u):
        rows, cols, data = [], [], []
        F = np.zeros(self.ndofs)

        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            cell_dofs = integrator.V.dh.get_cell_dofs(c)
            Ke, fe = integrator.compute(X, u[cell_dofs], c)

            for i, gi in enumerate(cell_dofs):
                F[gi] += fe[i]
                for j, gj in enumerate(cell_dofs):
                    rows.append(gi)
                    cols.append(gj)
                    data.append(Ke[i, j])

        K = sp.csr_matrix((data, (rows, cols)), shape=(self.ndofs, self.ndofs))
        return K, F


# ----------------------------------------
# Global Assembly
# ----------------------------------------

# @njit
# def assemble_global(mesh, u, uold, component, param = [], func = truss_element_stiffness_force):
#     ndofs = len(u)
#     data = []
#     rows = []
#     cols = []
#     F_int = np.zeros(ndofs)

#     param = {}
#     for e in range(mesh.cells.shape[0]):
#         n1, n2 = mesh.cells[e]
#         dofs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])
        
#         XL = mesh.X.flatten()[dofs]
#         uL = u[dofs]
#         uoldL = uold[dofs]
        
#         param['A'] = mesh.param['A'][e] 
#         param['E'] = mesh.param['E'][e]
#         # param['eta'] =mesh.param['eta'][e]
        
#         Ke, fe = func(XL, uL, uoldL, param, component)

#         F_int[dofs] += fe

#         grid = np.meshgrid(dofs, dofs) # x first, y second, in row-wise order
#         rows += list(grid[1].flatten()) # i runs in y
#         cols += list(grid[0].flatten()) # j runs in x
#         data += list(Ke.flatten())
        
#     K_global = sp.coo_matrix((data, (rows, cols)), shape=(ndofs, ndofs)).tocsr()
#     return K_global, F_int

# ----------------------------------------
# Apply boundary conditions
# ----------------------------------------

def apply_boundary_conditions(K, F, fixed_dofs, u_fixed):
    all_dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    
    F_mod = F[free_dofs] - K[free_dofs][:,fixed_dofs].toarray()@u_fixed

    return K[free_dofs][:, free_dofs], F_mod, free_dofs

# ----------------------------------------
# Newton-Raphson Solver
# ----------------------------------------

def solve_nonlinear(mesh, forces, fixed_dofs, u_fixed, u0 = None, tol=1e-10, max_iter=50, component='truss'):
    nnodes = mesh.X.shape[0]
    ndofs = 2 * nnodes
    u = u0 if type(u0)==type(None) else np.zeros(ndofs)
    uold = copy.deepcopy(u0) if type(u0)==type(None) else np.zeros(ndofs)
    
    zero_u_fixed = np.zeros_like(u_fixed)
    du = np.zeros_like(u)
    
    
    for k in range(max_iter):
        K, F_int = assemble_global(mesh, u, uold, component)
        R = forces - F_int
        
        # check it
        du_fixed = (u_fixed - u[fixed_dofs]) if k==0 else zero_u_fixed  
        
        # du_fixed = u_fixed if k==0 else zero_u_fixed
        K_mod, R_mod, free_dofs = apply_boundary_conditions(K, R, fixed_dofs, du_fixed)
        
        du[free_dofs] = spla.spsolve(K_mod, R_mod)
        du[fixed_dofs] = du_fixed
        
        uold[:] = u[:]
        u += du

        norm_res = np.linalg.norm(R_mod)
        print(f"Iter {k:2d}: Residual = {norm_res:.3e}")
        if norm_res < tol:
            break
        
    return u

def solve_linear(mesh, forces, fixed_dofs, u_fixed, component='truss'):
    nnodes = mesh.X.shape[0]
    ndofs = 2 * nnodes
    u = np.zeros(ndofs)
    udummy = np.zeros(ndofs)
    
    K, F = assemble_global(mesh, u, udummy, component, func = truss_element_canonical_problem) # None pour u_old
    F += forces
    
    # du_fixed = u_fixed if k==0 else zero_u_fixed
    K_mod, F_mod, free_dofs = apply_boundary_conditions(K, F, fixed_dofs, u_fixed)
    
    u[free_dofs] = spla.spsolve(K_mod, F_mod)
    u[fixed_dofs] = u_fixed
    
    
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

    mat = LinearEngStrainMaterial(E=210e9)

    U = FunctionSpace(mesh, TrussElement())
    
    A = np.array([1e-4, 1e-4, 1e-4])
    
    form = TrussLocalIntegrator(A, mat, U)
    ass = Assembler(mesh)
    
    K, F = ass.assemble(form, u)
    
    # ndofs = 2 * coords.shape[0]
    # forces = np.zeros(ndofs)
    # forces[5] = -1e6  # Load at node 2, y-direction

    # fixed_dofs = np.array([0, 1, 2, 3])  # Node 0 and 1 fixed
    # u_fixed = np.array( [0.0, 0.0, 0.01, 0.0] )
    
    # u0 = np.zeros_like(forces)
    # u = solve_nonlinear(mesh, forces, fixed_dofs, u_fixed, u0 = u0)

    # print("\nDisplacements:")
    # print(u.reshape(-1, 2))
    
    # plot_truss(mesh, u, scale=10.0)
