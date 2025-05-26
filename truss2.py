#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:17:53 2025

@author: felipe
"""

class TrussElement:
    def __init__(self, E, A, eta=0.0, component='truss'):
        self.E = E
        self.A = A
        self.eta = eta
        self.component = component

    def kinematics(self, X, u):
        Bmat = np.array([[-1,0,1,0], [0,-1,0,1]])
        a = Bmat @ X
        L0 = np.linalg.norm(a)
        a /= L0
        Bmat /= L0

        q = a + Bmat @ u
        lmbda = np.linalg.norm(q)
        b = q / lmbda
        return a, b, lmbda, L0, Bmat

    def stiffness_and_force(self, X, u, u_old):
        a, b, lmbda, L0, B = self.kinematics(X, u)
        lmbda_old = np.linalg.norm(a + B @ u_old)

        # Material model
        if self.component == 'truss':
            strain = lmbda - 1
            stress = self.E * strain
            dtang = self.E
        elif self.component == 'cable':
            strain = lmbda - 1
            stress = self.E * strain if lmbda > 1.0 else 0.0
            dtang = self.E if lmbda > 1.0 else 0.0
            stress += self.eta * (lmbda - lmbda_old)
            dtang += self.eta
        else:
            raise ValueError("Unknown component type")

        V = self.A * L0

        f_int = V * stress * (B.T @ b)
        D_mat = dtang * np.outer(b, b)
        D_geo = stress * (np.eye(2) - np.outer(b, b)) / lmbda
        K_local = V * B.T @ (D_mat + D_geo) @ B
        return K_local, f_int

class Assembler:
    def __init__(self, mesh, elements):
        self.mesh = mesh
        self.elements = elements  # List of TrussElement instances

    def assemble(self, u, u_old):
        ndofs = 2 * self.mesh.X.shape[0]
        K = sp.lil_matrix((ndofs, ndofs))
        F_int = np.zeros(ndofs)

        for e, (n1, n2) in enumerate(self.mesh.cells):
            dofs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])
            XL = self.mesh.X[[n1, n2]].flatten()
            uL = u[dofs]
            uoldL = u_old[dofs]
            Ke, fe = self.elements[e].stiffness_and_force(XL, uL, uoldL)
            for i in range(4):
                F_int[dofs[i]] += fe[i]
                for j in range(4):
                    K[dofs[i], dofs[j]] += Ke[i, j]

        return K.tocsr(), F_int
    
    # Example usage
coords = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0],
])
elements = np.array([
    [0, 2],
    [1, 2],
    [0, 1],
])
mesh = Mesh(coords, elements)

E = 210e9
A = 1e-4
truss_elements = [TrussElement(E, A) for _ in range(len(elements))]
assembler = Assembler(mesh, truss_elements)

forces = np.zeros(mesh.n_dofs)
forces[5] = -1e6

fixed_dofs = np.array([0, 1, 2, 3])
u_fixed = np.array([0.0, 0.0, 0.01, 0.0])
bc = BoundaryCondition(fixed_dofs, u_fixed)

solver = NonlinearSolver(assembler, bc, forces)
u = solver.solve()

plot_truss(mesh, u, scale=10.0)