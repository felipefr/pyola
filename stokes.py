#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:00:16 2025

@author: felipe
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# --- Mesh same as before ---

class Mesh:
    def __init__(self, vertices, cells):
        self.vertices = np.array(vertices)
        self.cells = np.array(cells)
        self.n_vertices = len(vertices)
        self.n_cells = len(cells)

    def cell_vertices(self, cell_idx):
        return self.vertices[self.cells[cell_idx]]

# --- FunctionSpace same as before ---

class FunctionSpace:
    def __init__(self, mesh, entity_dim, element):
        self.mesh = mesh
        self.entity_dim = entity_dim  # 0=vertex, 1=edge, 2=cell
        self.element = element
        if entity_dim == 2:
            self.n_dofs = mesh.n_cells * element.n_dofs
        elif entity_dim == 0:
            self.n_dofs = mesh.n_vertices * element.n_dofs
        else:
            raise NotImplementedError

    def dof_map(self, entity_idx):
        base = entity_idx * self.element.n_dofs
        return [base + i for i in range(self.element.n_dofs)]

# --- Element: pure basis info, no PDE specifics ---

class P1Element:
    def __init__(self):
        self.n_dofs = 3  # P1 triangle

    def shape_functions(self, xi_eta):
        xi, eta = xi_eta
        return np.array([1 - xi - eta, xi, eta])

    def grad_shape_functions(self):
        return np.array([[-1, -1],
                         [1, 0],
                         [0, 1]])

# --- Quadrature ---

class QuadratureRule:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

def triangle_quadrature():
    # simple 1-point quadrature for demo
    return QuadratureRule(points=[(1/3, 1/3)], weights=[0.5])

# --- LocalOperator: assembles local matrices for PDEs ---

class LocalOperator:
    def __init__(self, velocity_element, pressure_element, quadrature):
        self.vel_el = velocity_element
        self.pre_el = pressure_element
        self.quadrature = quadrature

    def evaluate(self, vertices, properties):
        # properties can include e.g. viscosity, density, source terms, domain markers

        grads = self.vel_el.grad_shape_functions()
        J = np.array([vertices[1] - vertices[0], vertices[2] - vertices[0]]).T
        detJ = np.abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        grads_phys = grads @ invJ

        n_u = self.vel_el.n_dofs
        n_p = self.pre_el.n_dofs
        local_mat = np.zeros((n_u + n_p, n_u + n_p))

        # Assemble velocity Laplacian (viscous term)
        for i in range(n_u):
            for j in range(n_u):
                # simple constant viscosity from properties
                mu = properties.get("viscosity", 1.0)
                val = mu * detJ/2 * np.dot(grads_phys[i], grads_phys[j])
                local_mat[i,j] += val

        # Assemble divergence matrix (velocity-pressure coupling)
        # Here we approximate B and B^T blocks
        # For demo, zero blocks
        # TODO: implement divergence matrix based on grads_phys

        return local_mat

# --- Assembly ---

class Assembly:
    def __init__(self, function_spaces):
        self.spaces = function_spaces
        self.n_dofs = sum(V.n_dofs for V in function_spaces)
        self.global_matrix = lil_matrix((self.n_dofs, self.n_dofs))
        self.global_rhs = np.zeros(self.n_dofs)
        self.offsets = np.cumsum([0]+[V.n_dofs for V in function_spaces[:-1]])

    def assemble_local(self, local_matrix, space_idx, dof_indices):
        offset = self.offsets[space_idx]
        dofs = [d + offset for d in dof_indices]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                self.global_matrix[di,dj] += local_matrix[i,j]

    def finalize(self):
        self.global_matrix = self.global_matrix.tocsr()

# --- Solver ---

class Solver:
    def __init__(self, assembly):
        self.A = assembly.global_matrix
        self.b = assembly.global_rhs

    def solve(self):
        return spsolve(self.A, self.b)

# --- BoundaryMarker same as before ---

class BoundaryMarker:
    def __init__(self, mesh):
        from scipy.spatial import ConvexHull
        self.mesh = mesh
        hull = ConvexHull(mesh.vertices)
        self.boundary_nodes = set(hull.vertices)

    def is_boundary_node(self, node):
        return node in self.boundary_nodes

# --- Assembler (generic) ---

class Assembler:
    def __init__(self, mesh, function_spaces, local_operator, domain_marker=None):
        self.mesh = mesh
        self.spaces = function_spaces
        self.local_operator = local_operator
        self.domain_marker = domain_marker if domain_marker else lambda ci: True
        self.assembly = Assembly(function_spaces)

    def assemble(self, properties_per_cell):
        for ci in range(self.mesh.n_cells):
            if not self.domain_marker(ci):
                continue
            vertices = self.mesh.cell_vertices(ci)
            props = properties_per_cell.get(ci, {})
            local_mat = self.local_operator.evaluate(vertices, props)

            # For demo: map dofs for velocity and pressure on cell
            dofs_u = self.spaces[0].dof_map(ci)
            dofs_p = self.spaces[1].dof_map(ci)
            dofs = dofs_u + dofs_p

            # Assemble local into global matrix
            offset_u = self.assembly.offsets[0]
            offset_p = self.assembly.offsets[1]
            global_dofs = [d + offset_u for d in dofs_u] + [d + offset_p for d in dofs_p]

            for i_local, i_global in enumerate(global_dofs):
                for j_local, j_global in enumerate(global_dofs):
                    self.assembly.global_matrix[i_global,j_global] += local_mat[i_local,j_local]

        self.assembly.finalize()
        return self.assembly

# --- StokesProblem ---

class StokesProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.vel_el = P1Element()
        self.pre_el = P1Element()

        self.V = FunctionSpace(mesh, 0, self.vel_el)
        self.Q = FunctionSpace(mesh, 0, self.pre_el)

        self.u = np.zeros(self.V.n_dofs)
        self.p = np.zeros(self.Q.n_dofs)

        self.local_operator = LocalOperator(self.vel_el, self.pre_el, triangle_quadrature())
        self.boundary = BoundaryMarker(mesh)

    def solve(self):
        def domain_marker(cell_idx):
            # use all cells
            return True

        # example properties per cell: viscosity constant
        properties = {ci: {"viscosity":1.0} for ci in range(self.mesh.n_cells)}

        assembler = Assembler(self.mesh, [self.V, self.Q], self.local_operator, domain_marker)
        assembly = assembler.assemble(properties)

        # Apply Dirichlet BC: zero velocity on boundary nodes
        for node in self.boundary.boundary_nodes:
            for i in range(self.vel_el.n_dofs):
                dof = self.V.dof_map(node)[i]
                global_dof = assembly.offsets[0] + dof
                assembly.global_matrix[global_dof,:] = 0
                assembly.global_matrix[global_dof, global_dof] = 1
                assembly.global_rhs[global_dof] = 0

        solver = Solver(assembly)
        sol = solver.solve()

        self.u = sol[assembly.offsets[0]:assembly.offsets[0]+self.V.n_dofs]
        self.p = sol[assembly.offsets[1]:assembly.offsets[1]+self.Q.n_dofs]

# --- Example usage ---

if __name__ == "__main__":
    vertices = [(0,0), (1,0), (0,1)]
    cells = [(0,1,2)]

    mesh = Mesh(vertices, cells)
    problem = StokesProblem(mesh)
    problem.solve()

    print("Velocity:", problem.u)
    print("Pressure:", problem.p)
