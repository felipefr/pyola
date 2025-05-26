#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:13:44 2025

@author: felipe
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# --- Mesh ---

class Mesh:
    def __init__(self, vertices, cells):
        self.vertices = np.array(vertices)
        self.cells = np.array(cells)
        self.n_vertices = len(vertices)
        self.n_cells = len(cells)

    def cell_vertices(self, cell_idx):
        return self.vertices[self.cells[cell_idx]]

# --- Function Spaces ---

class FunctionSpace:
    def __init__(self, mesh, entity_dim, element):
        self.mesh = mesh
        self.entity_dim = entity_dim
        self.element = element
        if entity_dim == 0:
            self.n_entities = mesh.n_vertices
        elif entity_dim == 2:
            self.n_entities = mesh.n_cells
        else:
            raise NotImplementedError("Only vertex or cell spaces supported")

        # Total DOFs = entities * dofs per entity
        self.n_dofs = self.n_entities * element.n_dofs

    def dof_map(self, entity_idx):
        base = entity_idx * self.element.n_dofs
        return [base + i for i in range(self.element.n_dofs)]

# --- Elements ---

class P1Element:
    # Linear triangle element with 3 DOFs
    def __init__(self):
        self.n_dofs = 3

    def shape_functions(self, xi_eta):
        xi, eta = xi_eta
        return np.array([1 - xi - eta, xi, eta])

    def grad_shape_functions(self):
        # reference gradients for P1 on triangle
        return np.array([[-1, -1],
                         [1, 0],
                         [0, 1]])

class VectorElement:
    def __init__(self, scalar_element, dim):
        self.scalar_element = scalar_element
        self.dim = dim
        self.n_dofs = scalar_element.n_dofs * dim

    def shape_functions(self, xi_eta):
        # vector shape functions = scalar shape functions times unit vectors
        scalar_sf = self.scalar_element.shape_functions(xi_eta)
        # returns (n_dofs, ) vector (block structure)
        # Just replicate scalar SF for each component for simplicity
        return np.tile(scalar_sf, self.dim)

    def grad_shape_functions(self):
        # Return block diagonal of scalar grad_shape_functions
        gs = self.scalar_element.grad_shape_functions()
        n = self.scalar_element.n_dofs
        d = self.dim
        grad_vec = np.zeros((n*d, d))
        for i in range(d):
            grad_vec[i*n:(i+1)*n, :] = gs
        return grad_vec

# --- Quadrature ---

class QuadratureRule:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

def triangle_quadrature():
    # 1-point quadrature for demo
    return QuadratureRule(points=[(1/3,1/3)], weights=[0.5])

# --- Local Operator for Stokes ---

class StokesLocalOperator:
    def __init__(self, velocity_element, pressure_element, quadrature):
        self.vel_el = velocity_element  # vector element
        self.pre_el = pressure_element  # scalar element
        self.quad = quadrature

    def evaluate(self, vertices, properties):
        mu = properties.get("viscosity", 1.0)
        # Geometric quantities
        grads_ref = self.vel_el.scalar_element.grad_shape_functions()  # 3x2
        J = np.array([vertices[1] - vertices[0], vertices[2] - vertices[0]]).T  # 2x2
        detJ = np.abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        grads_phys = grads_ref @ invJ  # 3x2

        # Number DOFs
        n_u = self.vel_el.n_dofs  # 6 for P1 2D vector
        n_p = self.pre_el.n_dofs  # 3 for P1 scalar

        # Initialize local matrix
        local_mat = np.zeros((n_u + n_p, n_u + n_p))

        # Velocity-velocity block (viscous term)
        # Loop over components (u_x and u_y)
        for i_comp in range(self.vel_el.dim):
            for j_comp in range(self.vel_el.dim):
                for i in range(self.vel_el.scalar_element.n_dofs):
                    for j in range(self.vel_el.scalar_element.n_dofs):
                        idx_i = i_comp * self.vel_el.scalar_element.n_dofs + i
                        idx_j = j_comp * self.vel_el.scalar_element.n_dofs + j
                        val = mu * detJ/2 * np.dot(grads_phys[i], grads_phys[j]) * (1.0 if i_comp == j_comp else 0.0)
                        local_mat[idx_i, idx_j] += val

        # Velocity-pressure coupling (divergence and gradient terms)
        # B matrix blocks for continuity equation
        # velocity-pressure block (upper right)
        for i in range(self.vel_el.scalar_element.n_dofs):
            for j in range(self.pre_el.n_dofs):
                # Div u test * p trial (divergence of u times p)
                for comp in range(self.vel_el.dim):
                    idx_u = comp * self.vel_el.scalar_element.n_dofs + i
                    idx_p = n_u + j
                    local_mat[idx_u, idx_p] += -detJ/2 * grads_phys[i, comp] * (1.0 if i == j else 0.0)  # simplified

        # pressure-velocity block (lower left) transpose of above (grad p * v test)
        for i in range(self.pre_el.n_dofs):
            for j in range(self.vel_el.scalar_element.n_dofs):
                for comp in range(self.vel_el.dim):
                    idx_p = n_u + i
                    idx_u = comp * self.vel_el.scalar_element.n_dofs + j
                    local_mat[idx_p, idx_u] += -detJ/2 * grads_phys[j, comp] * (1.0 if i == j else 0.0)

        return local_mat

# --- Assembly (generic) ---

class Assembly:
    def __init__(self, function_spaces):
        self.spaces = function_spaces
        self.n_dofs = sum(fs.n_dofs for fs in function_spaces)
        self.global_matrix = lil_matrix((self.n_dofs, self.n_dofs))
        self.global_rhs = np.zeros(self.n_dofs)
        self.offsets = np.cumsum([0] + [fs.n_dofs for fs in function_spaces[:-1]])

    def add_local_matrix(self, local_mat, global_dofs):
        for i_local, i_global in enumerate(global_dofs):
            for j_local, j_global in enumerate(global_dofs):
                self.global_matrix[i_global, j_global] += local_mat[i_local, j_local]

    def finalize(self):
        self.global_matrix = self.global_matrix.tocsr()

# --- Boundary Marker ---

class BoundaryMarker:
    def __init__(self, mesh):
        from scipy.spatial import ConvexHull
        self.mesh = mesh
        hull = ConvexHull(mesh.vertices)
        self.boundary_nodes = set(hull.vertices)

    def is_boundary_node(self, node):
        return node in self.boundary_nodes

# --- Mesh Function (subset marker) ---

class MeshFunction:
    def __init__(self, mesh, dim, marker=None):
        self.mesh = mesh
        self.dim = dim
        self.marker = marker if marker else (lambda idx: True)

    def entities(self):
        if self.dim == 0:
            return [i for i in range(self.mesh.n_vertices) if self.marker(i)]
        elif self.dim == 2:
            return [i for i in range(self.mesh.n_cells) if self.marker(i)]
        else:
            raise NotImplementedError

# --- Boundary Conditions ---

class BoundaryCondition:
    def apply(self, assembly, problem):
        raise NotImplementedError

class DirichletBC(BoundaryCondition):
    def __init__(self, function_space, boundary_entities, value=0.0):
        self.fs = function_space
        self.entities = boundary_entities
        self.value = value

    def apply(self, assembly, problem):
        offset = assembly.offsets[problem.spaces.index(self.fs)]
        for e in self.entities:
            dofs = self.fs.dof_map(e)
            for dof in dofs:
                gdof = offset + dof
                assembly.global_matrix[gdof, :] = 0
                assembly.global_matrix[gdof, gdof] = 1
                assembly.global_rhs[gdof] = self.value

class LagrangeMultiplierBC(BoundaryCondition):
    def __init__(self, function_space, constraint_func):
        self.fs = function_space
        self.constraint_func = constraint_func

    def apply(self, assembly, problem):
        assembly.global_matrix = assembly.global_matrix.tolil()
        n = assembly.global_matrix.shape[0]
        assembly.global_matrix.resize((n+1, n+1))
        assembly.global_rhs = np.resize(assembly.global_rhs, n+1)

        cvec = self.constraint_func(self.fs, assembly)

        assembly.global_matrix[n, :n] = cvec
        assembly.global_matrix[:n, n] = cvec.T
        assembly.global_rhs[n] = 0

        assembly.global_matrix = assembly.global_matrix.tocsr()

# --- Assembler ---

class Assembler:
    def __init__(self, mesh, function_spaces, local_operator, domain_marker=None, bcs=None):
        self.mesh = mesh
        self.spaces = function_spaces
        self.local_operator = local_operator
        self.domain_marker = domain_marker if domain_marker else lambda ci: True
        self.bcs = bcs if bcs else []
        self.assembly = Assembly(function_spaces)

    def assemble(self, properties_per_cell):
        for ci in range(self.mesh.n_cells):
            if not self.domain_marker(ci):
                continue
            vertices = self.mesh.cell_vertices(ci)
            props = properties_per_cell.get(ci, {})
            local_mat = self.local_operator.evaluate(vertices, props)

            global_dofs = []
            for sidx, space in enumerate(self.spaces):
                dofs = space.dof_map(ci)
                offset = self.assembly.offsets[sidx]
                global_dofs += [d + offset for d in dofs]

            self.assembly.add_local_matrix(local_mat, global_dofs)

        for bc in self.bcs:
            bc.apply(self.assembly, self)

        self.assembly.finalize()
        return self.assembly

# --- Solver ---

class Solver:
    def __init__(self, assembly):
        self.A = assembly.global_matrix
        self.b = assembly.global_rhs

    def solve(self):
        return spsolve(self.A, self.b)

# --- Stokes Problem ---

class StokesProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.scalar_el = P1Element()
        self.vel_el = VectorElement(self.scalar_el, 2)  # 2D velocity
        self.pre_el = self.scalar_el  # scalar pressure

        self.V = FunctionSpace(mesh, 0, self.vel_el)  # velocity on vertices
        self.Q = FunctionSpace(mesh, 0, self.pre_el)  # pressure on vertices

        self.u = np.zeros(self.V.n_dofs)
        self.p = np.zeros(self.Q.n_dofs)

        self.local_operator = StokesLocalOperator(self.vel_el, self.pre_el, triangle_quadrature())
        self.boundary = BoundaryMarker(mesh)

    def solve(self):
        # Marker for all cells (for now)
        domain_marker = lambda ci: True
        properties = {ci: {"viscosity": 1.0} for ci in range(self.mesh.n_cells)}

        # Boundary nodes for velocity BC
        boundary_nodes = [i for i in range(self.mesh.n_vertices) if self.boundary.is_boundary_node(i)]

        # Dirichlet BC zero velocity on boundary
        bc_vel = DirichletBC(self.V, boundary_nodes, 0.0)

        # Lagrange multiplier zero average pressure
        def zero_average_pressure_constraint(fs, assembly):
            offset = assembly.offsets[problem.spaces.index(fs)]
            vec = np.zeros(assembly.global_matrix.shape[0])
            for i in range(fs.n_dofs):
                vec[offset + i] = 1.0 / fs.n_dofs
            return vec

        bc_avg_p = LagrangeMultiplierBC(self.Q, zero_average_pressure_constraint)

        assembler = Assembler(self.mesh, [self.V, self.Q], self.local_operator, domain_marker, bcs=[bc_vel, bc_avg_p])

        assembly = assembler.assemble(properties)
        solver = Solver(assembly)
        sol = solver.solve()

        self.u = sol[assembly.offsets[0]:assembly.offsets[0]+self.V.n_dofs]
        self.p = sol[assembly.offsets[1]:assembly.offsets[1]+self.Q.n_dofs]

        return self.u, self.p

# --- Example run ---

if __name__ == "__main__":
    vertices = [(0,0), (1,0), (0,1)]
    cells = [(0,1,2)]
    mesh = Mesh(vertices, cells)
    problem = StokesProblem(mesh)
    u, p = problem.solve()

    print("Velocity DOFs:", u)
    print("Pressure DOFs:", p)
