#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:11:58 2025

@author: felipe
"""

import numpy as np

# Mesh and geometry
class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)

# Degrees of freedom and function space
class FunctionSpace:
    def __init__(self, mesh):
        self.mesh = mesh
        self.num_nodes = len(mesh.nodes)
        self.dofs = np.arange(2 * self.num_nodes)  # 2D: u,v per node

    def get_dof_indices(self, element):
        return np.ravel([[2*i, 2*i+1] for i in element])

# Truss element with Green-Lagrange strain
class TrussElement:
    def __init__(self, E_func, A_func):
        self.E_func = E_func  # Young's modulus function
        self.A_func = A_func  # Area function

    def compute(self, X, u, element):
        xi, xj = X[element[0]], X[element[1]]
        ui, uj = u[element[0]], u[element[1]]
        
        dX = xj - xi
        du = uj - ui

        L0 = np.linalg.norm(dX)
        dx = dX + du
        L = np.linalg.norm(dx)

        # Green-Lagrange strain
        eps = 0.5 * ((L/L0)**2 - 1)

        E = self.E_func((xi + xj)/2)
        A = self.A_func((xi + xj)/2)

        # Stress and force
        sigma = E * eps
        P = A * sigma  # First Piola

        # Tangent stiffness
        k_geo = A * E * eps / L**2
        k_mat = A * E / L0**2
        k_total = (k_mat + 2 * k_geo)

        # Direction cosines
        n = dx / L
        B = np.outer(n, n)

        # Residual
        f_int = P * np.concatenate([-n, n])

        # Stiffness matrix
        k_local = k_total * np.block([
            [ B, -B],
            [-B,  B]
        ])

        return f_int, k_local

# Assembler
class Assembler:
    def __init__(self, mesh, function_space, element_model):
        self.mesh = mesh
        self.fs = function_space
        self.element_model = element_model

    def assemble(self, u):
        ndofs = 2 * self.fs.num_nodes
        K = np.zeros((ndofs, ndofs))
        R = np.zeros(ndofs)
        X = self.mesh.nodes
        U = u.reshape((-1, 2))

        for elem in self.mesh.elements:
            dofs = self.fs.get_dof_indices(elem)
            f_int, k_local = self.element_model.compute(X, U, elem)
            R[dofs] += f_int
            K[np.ix_(dofs, dofs)] += k_local

        return K, R

# Dirichlet boundary condition
class BoundaryCondition:
    def __init__(self, node, value):
        self.node = node
        self.value = value  # 2D

    def apply(self, A, b):
        for i in range(2):
            idx = 2*self.node + i
            A[idx, :] = 0
            A[:, idx] = 0
            A[idx, idx] = 1.0
            b[idx] = self.value[i]

# Nonlinear solver
class NonlinearSolver:
    def __init__(self, assembler, bcs):
        self.assembler = assembler
        self.bcs = bcs

    def solve(self, u0, tol=1e-8, maxiter=30):
        u = u0.copy()
        for iter in range(maxiter):
            K, R = self.assembler.assemble(u)
            R *= -1  # residual: R = f_int - f_ext = 0

            for bc in self.bcs:
                bc.apply(K, R)

            du = np.linalg.solve(K, R)
            u += du

            if np.linalg.norm(du) < tol:
                print(f'Converged in {iter} iterations.')
                return u
        raise RuntimeError('Did not converge')

# Example usage
if __name__ == "__main__":
    # Define nodes and elements
    nodes = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    elements = [
        [0, 1],
        [1, 2],
        [0, 2],
    ]

    mesh = Mesh(nodes, elements)
    fs = FunctionSpace(mesh)

    E_func = lambda x: 200e9  # Young's modulus
    A_func = lambda x: 0.01   # Cross-sectional area

    elem_model = TrussElement(E_func, A_func)
    assembler = Assembler(mesh, fs, elem_model)

    u0 = np.zeros(2 * len(nodes))  # Initial guess

    # Boundary conditions: node 0 fixed, node 2 moves down
    bcs = [
        BoundaryCondition(0, [0.0, 0.0]),
        BoundaryCondition(2, [0.0, -0.1])
    ]

    solver = NonlinearSolver(assembler, bcs)
    u = solver.solve(u0)

    # Print final displacements
    print("Displacements (x, y) per node:")
    print(u.reshape((-1, 2)))
