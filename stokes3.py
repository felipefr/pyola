#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:32:51 2025

@author: felipe
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# --- Mesh class ---

class Mesh:
    def __init__(self, vertices, cells):
        self.vertices = np.array(vertices)  # shape (n_vertices, 2)
        self.cells = np.array(cells)        # shape (n_cells, 3)

    @property
    def n_cells(self):
        return len(self.cells)

    def cell_vertices(self, ci):
        return self.vertices[self.cells[ci]]

# --- Finite Elements ---

class P1Element:
    def __init__(self):
        self.n_dofs = 3

    def shape_functions(self, xi_eta):
        xi, eta = xi_eta
        return np.array([1 - xi - eta, xi, eta])

    def grad_shape_functions(self):
        # Constant gradients for P1 on reference triangle
        return np.array([[-1, -1], [1, 0], [0, 1]])

# Vector element for velocity = dim * scalar P1
class VectorP1Element:
    def __init__(self, dim=2):
        self.scalar_element = P1Element()
        self.dim = dim
        self.n_dofs = dim * self.scalar_element.n_dofs

    def B_matrix(self, vertices):
        # Construct B matrix for viscous term
        # Returns a (dim*ndofs, dim*dim) matrix applying gradients and mapping
        grads_ref = self.scalar_element.grad_shape_functions()
        J = np.array([vertices[1] - vertices[0], vertices[2] - vertices[0]]).T
        detJ = np.abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        grads_phys = grads_ref @ invJ  # shape (3, 2)

        ndofs = self.scalar_element.n_dofs
        dim = self.dim

        # B matrix shape: (dim*ndofs, dim*dim)
        # Arrange gradients per dof per component for symmetric gradient
        B = np.zeros((dim*ndofs, dim*dim))
        for i in range(ndofs):
            # Rows for dof i in all velocity components
            for d in range(dim):
                row = d*ndofs + i
                # Symmetric gradient components for viscous term
                # Following 2D vector calculus: strain components e_11, e_22, e_12
                # We'll arrange B columns as [d/dx u1, d/dy u1, d/dx u2, d/dy u2]
                # But Stokes viscous term requires grad(u) : grad(v) = sum over components of grad shape functions
                # We'll store gradient components per component per dof as:
                # For 2D, gradient components per component: (du/dx, du/dy)
                # So for velocity component d:
                # put grads_phys[i,0] at col d*dim + 0
                # put grads_phys[i,1] at col d*dim + 1

                # But to simplify viscous term mu*∇u : ∇v,
                # we sum grad shape * grad shape for each component

                B[row, d*dim + 0] = grads_phys[i,0]  # d/dx u_d
                B[row, d*dim + 1] = grads_phys[i,1]  # d/dy u_d
        return B, detJ

# --- Quadrature ---

class QuadratureRule:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

def triangle_quadrature_3pt():
    pts = [(1/6,1/6), (2/3,1/6), (1/6,2/3)]
    wts = [1/6, 1/6, 1/6]
    return QuadratureRule(pts, wts)

# --- FunctionSpace ---

class FunctionSpace:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        self.n_dofs = element.n_dofs * mesh.vertices.shape[0] // element.scalar_element.n_dofs

    def dof_map(self, cell_idx):
        # Return global dofs for this cell
        cell = self.mesh.cells[cell_idx]
        # For vector, dofs per component in order: [u0_x, u1_x, u2_x, u0_y, u1_y, u2_y]
        # or for scalar, just nodal indices
        if isinstance(self.element, VectorP1Element):
            base = []
            for comp in range(self.element.dim):
                base += [comp * self.element.scalar_element.n_dofs + i for i in range(self.element.scalar_element.n_dofs)]
            # global dofs = for each node index + component offset
            global_dofs = []
            for comp in range(self.element.dim):
                for local_dof in range(self.element.scalar_element.n_dofs):
                    global_dofs.append(cell[local_dof] + comp * len(self.mesh.vertices))
            return global_dofs
        else:
            # scalar element
            return list(cell)

# --- Assembly and Local Operator ---

class Assembly:
    def __init__(self, function_spaces):
        self.spaces = function_spaces
        self.n_dofs = sum(fs.n_dofs for fs in function_spaces)
        self.global_matrix = lil_matrix((self.n_dofs, self.n_dofs))
        self.global_rhs = np.zeros(self.n_dofs)
        # offsets to locate each space block in global matrix/vector
        self.offsets = []
        offset = 0
        for fs in function_spaces:
            self.offsets.append(offset)
            offset += fs.n_dofs

    def add_local_matrix(self, local_mat, global_dofs):
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                self.global_matrix[gi, gj] += local_mat[i, j]

    def add_local_rhs(self, local_rhs, global_dofs):
        for i, gi in enumerate(global_dofs):
            self.global_rhs[gi] += local_rhs[i]

    def finalize(self):
        self.global_matrix = self.global_matrix.tocsr()

class StokesLocalOperator:
    def __init__(self, velocity_element, pressure_element, quadrature):
        self.vel_el = velocity_element
        self.pre_el = pressure_element
        self.quad = quadrature

    def evaluate(self, vertices, properties_per_qp):
        mu_vals = properties_per_qp.get("viscosity", [1.0]*len(self.quad.points))
        ndofs_u = self.vel_el.n_dofs
        ndofs_p = self.pre_el.n_dofs
        local_mat = np.zeros((ndofs_u + ndofs_p, ndofs_u + ndofs_p))

        for qp, w in enumerate(self.quad.weights):
            mu = mu_vals[qp]
            B, detJ = self.vel_el.B_matrix(vertices)  # B shape (ndofs_u, dim*dim)

            # Viscous term: A = mu * B * B^T * weight * detJ
            # Here B*B^T yields shape (ndofs_u, ndofs_u)
            A = mu * w * detJ * (B @ B.T)
            local_mat[:ndofs_u, :ndofs_u] += A

            # Gradients of velocity shape functions (for divergence terms)
            grads_ref = self.vel_el.scalar_element.grad_shape_functions()
            J = np.array([vertices[1] - vertices[0], vertices[2] - vertices[0]]).T
            detJ = np.abs(np.linalg.det(J))
            invJ = np.linalg.inv(J)
            grads_phys = grads_ref @ invJ  # (3, 2)

            # Pressure-velocity coupling terms
            # Bp = divergence of velocity shape functions, i.e. sum of grad components per velocity component
            # Assemble blocks:
            # - velocity-pressure block (divergence) shape (ndofs_u, ndofs_p)
            # - pressure-velocity block (transpose)
            ndofs = self.vel_el.scalar_element.n_dofs
            for i in range(ndofs):
                div_phi_i = grads_phys[i].sum()  # div phi_i approximation (simple sum grad_x + grad_y)
                for j in range(ndofs_p):
                    # velocity dof index in block (all components)
                    for comp in range(self.vel_el.dim):
                        row = comp * ndofs + i
                        col = ndofs_u + j
                        # Weak form coupling: - ∫ q div u ≈ - q_j * div phi_i
                        val = -w * detJ * grads_phys[i, comp] if i == j else 0.0
                        local_mat[row, col] += -w * detJ * grads_phys[i, comp]
                        local_mat[col, row] += -w * detJ * grads_phys[i, comp]

        return local_mat

# --- Boundary conditions ---

class BoundaryCondition:
    def __init__(self, function_space, dofs, values):
        self.fs = function_space
        self.dofs = dofs
        self.values = values

    def apply(self, assembly):
        # Strong imposition: zero out rows and cols, put 1 on diagonal, RHS=val
        for dof, val in zip(self.dofs, self.values):
            assembly.global_matrix[dof, :] = 0
            assembly.global_matrix[:, dof] = 0
            assembly.global_matrix[dof, dof] = 1
            assembly.global_rhs[dof] = val

# --- Multipoint constraint (Lagrange multiplier) ---

class MultipointConstraint:
    def __init__(self, fs, dof_indices, weights, value=0.0):
        self.fs = fs
        self.dof_indices = dof_indices
        self.weights = weights
        self.value = value

    def apply(self, assembly):
        assembly.global_matrix = assembly.global_matrix.tolil()
        n = assembly.global_matrix.shape[0]
        assembly.global_matrix.resize((n+1, n+1))
        assembly.global_rhs = np.resize(assembly.global_rhs, n+1)

        cvec = np.zeros(n)
        offset = assembly.offsets[assembly.spaces.index(self.fs)]
        for dof, w in zip(self.dof_indices, self.weights):
            cvec[offset + dof] = w

        assembly.global_matrix[n, :n] = cvec
        assembly.global_matrix[:n, n] = cvec.T
        assembly.global_rhs[n] = self.value
        assembly.global_matrix = assembly.global_matrix.tocsr()

# --- Assembler ---

class Assembler:
    def __init__(self, mesh, function_spaces, local_operator, domain_marker=None, bcs=None, mpcs=None):
        self.mesh = mesh
        self.spaces = function_spaces
        self.local_operator = local_operator
        self.domain_marker = domain_marker or (lambda ci: True)
        self.bcs = bcs or []
        self.mpcs = mpcs or []

        self.assembly = Assembly(function_spaces)

    def assemble_matrix(self, properties_per_cell):
        for ci in range(self.mesh.n_cells):
            if not self.domain_marker(ci):
                continue
            vertices = self.mesh.cell_vertices(ci)
            props = properties_per_cell.get(ci, {})
            local_mat = self.local_operator.evaluate(vertices, props)

            # Build global dofs combining all spaces per cell
            global_dofs = []
            for sidx, space in enumerate(self.spaces):
                global_dofs += space.dof_map(ci)

            self.assembly.add_local_matrix(local_mat, global_dofs)
        self.assembly.finalize()

        # Apply multipoint constraints before BCs
        for mpc in self.mpcs:
            mpc.apply(self.assembly)

        # Apply boundary conditions strongly
        for bc in self.bcs:
            bc.apply(self.assembly)

        return self.assembly.global_matrix

    def assemble_rhs(self, load_func):
        # Reset rhs
        self.assembly.global_rhs.fill(0)

        # User provides a function returning forcing values per global DOF
        self.assembly.global_rhs = load_func(self.assembly.n_dofs)

        # Reapply BCs on RHS after forcing
        for bc in self.bcs:
            bc.apply(self.assembly)

        # Reapply MPC RHS entries
        for mpc in self.mpcs:
            mpc.apply(self.assembly)

        return self.assembly.global_rhs

# --- Solver ---

class Solver:
    def __init__(self, assembly):
        self.A = assembly.global_matrix
        self.b = assembly.global_rhs

    def solve(self):
        return spsolve(self.A, self.b)

# --- Stokes problem ---

class StokesProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = FunctionSpace(mesh, VectorP1Element())
        self.Q = FunctionSpace(mesh, P1Element())
        self.quad = triangle_quadrature_3pt()
        self.local_operator = StokesLocalOperator(self.V.element, self.Q.element, self.quad)

    def zero_average_pressure_constraint(self, assembly):
        dofs = list(range(self.Q.n_dofs))
        weights = [1.0/self.Q.n_dofs] * self.Q.n_dofs
        return MultipointConstraint(self.Q, dofs, weights, 0.0)

    def solve(self):
        # Example viscosity per cell quadrature points (constant here)
        properties = {ci: {"viscosity": [1.0]*len(self.quad.points)} for ci in range(self.mesh.n_cells)}

        # Dirichlet BC on velocity (just zero velocity at all nodes for demo)
        bc_dofs = list(range(self.V.n_dofs))  # All velocity DOFs
        bc_vals = [0.0]*len(bc_dofs)
        bc_vel = BoundaryCondition(self.V, bc_dofs, bc_vals)

        # MPC zero-average pressure
        # Will be applied in assembly
        # (We can pass it to Assembler)

        assembler = Assembler(self.mesh, [self.V, self.Q], self.local_operator, bcs=[bc_vel])
        assembly = assembler.assemble_matrix(properties)

        mpc = self.zero_average_pressure_constraint(assembler.assembly)
        assembler.mpcs.append(mpc)
        mpc.apply(assembler.assembly)

        # RHS = zero forcing for demo
        rhs = assembler.assemble_rhs(lambda n: np.zeros(n))

        solver = Solver(assembler.assembly)
        sol = solver.solve()

        u = sol[:self.V.n_dofs]
        p = sol[self.V.n_dofs:self.V.n_dofs + self.Q.n_dofs]

        return u, p

# --- Demo ---

if __name__ == "__main__":
    vertices = [(0,0), (1,0), (0,1)]
    cells = [(0,1,2)]
    mesh = Mesh(vertices, cells)
    problem = StokesProblem(mesh)
    u, p = problem.solve()

    print("Velocity DOFs:", u)
    print("Pressure DOFs:", p)
