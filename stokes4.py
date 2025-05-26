#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:29:51 2025

@author: felipe
"""

# Re-run the full implementation after the code execution environment was reset

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------

class Mesh:
    def __init__(self, X, cells):
        self.X = X  # (n_nodes, 2)
        self.cells = cells  # (n_cells, 3)
        self.edges, self.edge_map = self.build_edges()
        self.n_nodes = len(X)
        self.n_edges = len(self.edges)
        self.n_cells = len(cells)

    def build_edges(self):
        edge_dict = {}
        edge_list = []

        for c in self.cells:
            for i in range(3):
                a, b = sorted((c[i], c[(i+1)%3]))
                if (a, b) not in edge_dict:
                    edge_dict[(a, b)] = len(edge_list)
                    edge_list.append((a, b))

        return np.array(edge_list), edge_dict

# -----------------------------------------------------------------------------
# FunctionSpace
# -----------------------------------------------------------------------------

class FunctionSpace:
    def __init__(self, mesh, entity, dofs_per_entity=1, name="unnamed"):
        self.mesh = mesh
        self.entity = entity
        self.dofs_per_entity = dofs_per_entity
        self.name = name
        if entity == "node":
            self.n_dofs = mesh.n_nodes * dofs_per_entity
        elif entity == "edge":
            self.n_dofs = mesh.n_edges * dofs_per_entity
        else:
            raise NotImplementedError(f"Entity '{entity}' not supported")

    def get_dofs_for_cell(self, cell_id):
        cell = self.mesh.cells[cell_id]
        if self.entity == "node":
            return cell
        elif self.entity == "edge":
            edges = []
            for i in range(3):
                a, b = sorted((cell[i], cell[(i+1)%3]))
                edge_idx = self.mesh.edge_map[(a, b)]
                edges.append(edge_idx)
            return np.array(edges)
        else:
            raise NotImplementedError()

# -----------------------------------------------------------------------------
# DOFHandler
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Basis and B-matrix helper for P2 and P1
# -----------------------------------------------------------------------------

def local_stokes_matrix(X):
    area = 0.5 * np.linalg.det(np.array([[1, *X[0]], [1, *X[1]], [1, *X[2]]]))
    grads = np.array([
        [X[1,1] - X[2,1], X[2,0] - X[1,0]],
        [X[2,1] - X[0,1], X[0,0] - X[2,0]],
        [X[0,1] - X[1,1], X[1,0] - X[0,0]]
    ]) / (2 * area)

    B = np.zeros((3, 6))
    for i in range(3):
        B[0, i]   = grads[i, 0]
        B[1, i+3] = grads[i, 1]
        B[2, i]   = grads[i, 1]
        B[2, i+3] = grads[i, 0]

    D = np.eye(3)
    Ke = area * B.T @ D @ B

    G = np.zeros((3, 6))
    for i in range(3):
        G[i, i]   = grads[i, 0]
        G[i, i+3] = grads[i, 1]
    Be = -area * G

    K_local = np.block([
        [Ke, Be.T],
        [Be, np.zeros((3, 3))]
    ])

    f_local = np.zeros(9)
    return K_local, f_local

# -----------------------------------------------------------------------------
# Assembler
# -----------------------------------------------------------------------------

class Assembler:
    def __init__(self, mesh, dof_handler):
        self.mesh = mesh
        self.dh = dof_handler
        self.ndofs = self.dh.total_dofs

    def assemble(self, local_stiffness_func):
        rows, cols, data = [], [], []
        F = np.zeros(self.ndofs)

        for c in range(self.mesh.n_cells):
            X = self.mesh.X[self.mesh.cells[c]]
            local_maps = self.dh.get_cell_dofs(c)
            cell_dofs = np.concatenate(local_maps)
            Ke, fe = local_stiffness_func(X)

            for i, gi in enumerate(cell_dofs):
                F[gi] += fe[i]
                for j, gj in enumerate(cell_dofs):
                    rows.append(gi)
                    cols.append(gj)
                    data.append(Ke[i, j])

        K = sp.csr_matrix((data, (rows, cols)), shape=(self.ndofs, self.ndofs))
        return K, F

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

def example_stokes():
    X = np.array([[0, 0], [1, 0], [0, 1]])
    cells = np.array([[0, 1, 2]])

    mesh = Mesh(X, cells)

    V = FunctionSpace(mesh, entity="edge", dofs_per_entity=1, name="velocity")
    Q = FunctionSpace(mesh, entity="node", dofs_per_entity=1, name="pressure")

    dh = DOFHandler(mesh)
    dh.add_space(V, name="velocity")
    dh.add_space(Q, name="pressure")
    dh.finalize()

    assembler = Assembler(mesh, dh)
    K, F = assembler.assemble(local_stokes_matrix)

    print("Stiffness matrix shape:", K.shape)
    print("RHS vector:", F)

example_stokes()
