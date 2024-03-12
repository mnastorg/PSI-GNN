##########################################################################
################################ PACKAGES ################################
##########################################################################
import os 

import numpy as np
from scipy.sparse import csr_matrix, find

from fenics import *
from mshr import *
from dolfin import *
##########################################################################
##########################################################################
##########################################################################

def solve_poisson(mesh_path, radius = 1):

    ### Initial information
    param_f = np.random.uniform(-10, 10, 3)
    param_g = np.random.uniform(-10, 10, 6)

    ### Define expressions of function f and g
    f = Expression( 'A*((x[0]/R)-1)*((x[0]/R)-1) + B*(x[1]/R)*(x[1]/R) + C',
                    A = param_f[0], B = param_f[1], C = param_f[2], 
                    R = radius, degree = 2 
                    )
    
    g = Expression( 'A*(x[0]/R)*(x[0]/R) + B*(x[0]/R)*(x[1]/R) + C*(x[1]/R)*(x[1]/R) + D*(x[0]/R) + E*(x[1]/R) + F',
                    A = param_g[0], B = param_g[1], C = param_g[2], 
                    D = param_g[3], E = param_g[4], F = param_g[5],
                    R = radius, degree = 2 
                    )

    ### Read mesh file
    comm = MPI.comm_world
    mesh = Mesh()
    loc = os.path.join(mesh_path, "mesh.h5")
    try :
        with HDF5File(comm, loc, "r") as h5file:
            h5file.read(mesh, "mesh", False)
            facet = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
            h5file.read(facet, "facet")
    except FileNotFoundError as fnf_error:
        print(fnf_error)

    ### Define function space + coordinates
    V = FunctionSpace(mesh, "Lagrange", 1)
    d2v = dof_to_vertex_map(V)
    vertex_coordinates = mesh.coordinates()
    dof_coordinates = vertex_coordinates[d2v]

    ### Variational Formulation and resolution of the poisson problem
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, g, facet, 101)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    A = assemble(a)
    b = assemble(L)

    bc.apply(A, b)

    u = Function(V)
    solve(A, u.vector(), b)

    ### Extract Relevant information
    A_array = A.copy().array()
    A_sparse_matrix = csr_matrix(A_array)
    b_matrix = b.get_local().reshape(-1,1)
    sol = u.compute_vertex_values(mesh)[d2v].reshape(-1,1)

    f_interior = f.compute_vertex_values(mesh)[d2v].reshape(-1,1)
    tags = np.zeros_like(sol)
    prb_data = np.hstack((f_interior, np.zeros(f_interior.shape)))
    g_boundary = bc.get_boundary_values()
    g_boundary = list(g_boundary.items())
    
    for items in g_boundary :
        prb_data[items[0],1] = items[1]
        prb_data[items[0],0] = 0
        tags[items[0],:] = 1

    coefficients = np.asarray(find(A_sparse_matrix))
    edge_index = coefficients[:2,:].T.astype('int')
    distance = compute_position(edge_index, dof_coordinates)

    return A_sparse_matrix, b_matrix, dof_coordinates, sol, prb_data, tags, distance

def compute_position(edge_index, coordinates):

    distance = np.zeros((np.shape(edge_index)[0], 3))

    for e in range(np.shape(edge_index)[0]):
        edge = edge_index[e,:]
        u_ij = coordinates[edge[0],:] - coordinates[edge[1],:]
        distance[e,:2] = u_ij
        distance[e,2] = np.sqrt(u_ij[0]**2 + u_ij[1]**2)

    return distance
