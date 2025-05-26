#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:59:59 2025

@author: felipe
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Mesh class
class Mesh:
    def __init__(self, X, cells):
        self.X = np.array(X)
        self.cells = np.array(cells)
        self.n_nodes = self.X.shape[0]
        self.n_elements = self.cells.shape[0]

# Element class (P1 Triangle)
class P1Element:
    def reference_coords(self):
        return np.array([[0, 0], [1, 0], [0, 1]])

    def basis_gradients(self):
        # Constant gradients for linear triangle
        return np.array([[-1, -1], [1, 0], [0, 1]])

    def quadrature(self):
        # 1-point quadrature rule
        qp = np.array([[1/3, 1/3]])
        w = np.array([0.5])
        return qp, w

# DOF handler
class DOFHandler:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.dofs_per_cell = 3
        self.n_dofs = mesh.n_nodes
        self.cell_dofs = self.build_cell_dofs()

    def build_cell_dofs(self):
        return self.mesh.cells

# Function class
class Function:
    def __init__(self, dofhandler, values=None):
        self.dofhandler = dofhandler
        self.values = np.zeros(dofhandler.n_dofs) if values is None else values

# Poisson integrator
def local_stiffness(mesh, dofhandler, element, cell):
    x = mesh.X[cell]
    grad_ref = element.basis_gradients()
    J = np.array([x[1] - x[0], x[2] - x[0]]).T
    detJ = np.abs(np.linalg.det(J))
    invJT = np.linalg.inv(J).T
    grad = grad_ref @ invJT
    Ke = detJ * (grad @ grad.T)
    return Ke

def local_rhs(mesh, element, cell, f):
    x = mesh.X[cell]
    qp, w = element.quadrature()
    # Linear shape functions evaluated at single quad point are [1/3, 1/3, 1/3]
    phi = np.array([1/3, 1/3, 1/3])
    x_qp = phi @ x
    fe = w[0] * f(x_qp) * phi
    return fe

# Assembly routine
def assemble(mesh, dofhandler, element, f):
    A = sp.lil_matrix((dofhandler.n_dofs, dofhandler.n_dofs))
    b = np.zeros(dofhandler.n_dofs)

    for c in range(mesh.n_elements):
        cell = mesh.cells[c]
        Ke = local_stiffness(mesh, dofhandler, element, cell)
        fe = local_rhs(mesh, element, cell, f)
        for i in range(3):
            for j in range(3):
                A[cell[i], cell[j]] += Ke[i, j]
            b[cell[i]] += fe[i]

    return A.tocsr(), b

# Boundary condition class
class DirichletBC:
    def __init__(self, dofhandler, boundary_fun, g):
        self.dofs = []
        self.values = []
        for i, x in enumerate(dofhandler.mesh.X):
            if boundary_fun(x):
                self.dofs.append(i)
                self.values.append(g(x))
        self.dofs = np.array(self.dofs, dtype=int)
        self.values = np.array(self.values)

def apply_bc(A, b, bc):
    free_dofs = np.setdiff1d(np.arange(A.shape[0]), bc.dofs)
    b -= A[:, bc.dofs] @ bc.values
    A[bc.dofs, :] = 0.0
    A[:, bc.dofs] = 0.0
    A[bc.dofs, bc.dofs] = 1.0
    b[bc.dofs] = bc.values
    return A, b

# Solve system
def solve(A, b):
    return spla.spsolve(A, b)

# Plotting
def plot(mesh, u):
    from matplotlib.tri import Triangulation
    triang = Triangulation(mesh.X[:, 0], mesh.X[:, 1], mesh.cells)
    plt.tricontourf(triang, u, levels=50)
    plt.colorbar()
    plt.title("Solution u(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()

# Example run

X = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])
cells = np.array([
    [0, 1, 2],
    [0, 2, 3]
])
mesh = Mesh(X, cells)
element = P1Element()
dofhandler = DOFHandler(mesh, element)

f = lambda x: 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
g = lambda x: 0.0
boundary_fun = lambda x: np.any(np.isclose(x, 0)) or np.any(np.isclose(x, 1))

A, b = assemble(mesh, dofhandler, element, f)
bc = DirichletBC(dofhandler, boundary_fun, g)
A, b = apply_bc(A, b, bc)
u = solve(A, b)

plot(mesh, u)

