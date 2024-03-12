##################### PACKAGES #################################################
################################################################################

import os
import subprocess
import argparse 

import sys 
sys.path.append("..")

import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix, find
from tabulate import tabulate 

import torch
from torch_geometric.data import Data
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from fenics import *
from mshr import *
from dolfin import *

import vis 

import model_test as model

################################################################################
################################################################################

def solve_poisson(mesh_path):

    ### Initial information
    param_f = np.random.uniform(-10, 10, 3)
    param_g = np.random.uniform(-10, 10, 6)

    ### Define expressions of function f and g
    f = Expression( 'A*(x[0]-1)*(x[0]-1) + B*x[1]*x[1] + C',
                    A = param_f[0], B = param_f[1], C = param_f[2],
                    degree = 2 
                    )
    
    g = Expression( 'A*x[0]*x[0] + B*x[0]*x[1] + C*x[1]*x[1] + D*x[0] + E*x[1] + F',
                    A = param_g[0], B = param_g[1], C = param_g[2], D = param_g[3], E = param_g[4], F = param_g[5],
                    degree = 2 
                    )

    ### Read mesh file
    comm = MPI.comm_world
    mesh = Mesh()
    try :
        with HDF5File(comm, os.path.join(mesh_path,"mesh.h5"), "r") as h5file:
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

def build_data(path_mesh) : 
    
    ## Extract data and create Data object
    path_mesh = os.path.join(path_mesh)
    A_sparse_matrix, b_matrix, dof_coordinates, sol, prb_data, tags, distance = solve_poisson(path_mesh)

    precision = torch.float 

    prb_data_mean = torch.tensor([0.0464, -0.0006], dtype = precision)
    prb_data_std = torch.tensor([9.6267, 3.2935], dtype = precision)

    distance_mean = torch.tensor([0.0, 0.0, 0.0655], dtype = precision)
    distance_std = torch.tensor([0.0507, 0.0507, 0.0293], dtype = precision)

    # Build edge_index and a_ij
    coefficients = np.asarray(find(A_sparse_matrix))
    edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)
    a_ij = torch.tensor(coefficients[2,:].reshape(-1,1), dtype=precision)

    # Build b tensor
    b =  torch.tensor(b_matrix, dtype = precision)            

    # Extract exact solution
    sol = torch.tensor(sol, dtype = precision)

    # Extract prb_data
    prb_data = torch.tensor(prb_data, dtype = precision)
    prb_data = (prb_data - prb_data_mean) / prb_data_std

    # Extract prb_data
    edge_attr = torch.tensor(distance, dtype = precision)
    edge_attr = (edge_attr - distance_mean) / distance_std

    # Extract tags to differentiate nodes 
    tags = torch.tensor(tags, dtype=precision)

    # Extract coordinates
    pos = torch.tensor(dof_coordinates, dtype = precision)

    # Extract initial condition
    x = torch.zeros_like(sol)
    index_boundary = torch.where(tags==1)[0]
    x[index_boundary,:] = b[index_boundary,:]

    data = Data(    x = x, edge_index = edge_index, 
                    edge_attr = edge_attr, a_ij = a_ij, y = b, 
                    sol = sol, prb_data = prb_data, tags = tags,  
                    pos = pos
                )

    return data
    
def test_sample(checkpoint, data, device):

    config = checkpoint["hyperparameters"]

    list_res = []
    list_mse = []

    config["k"] = 150

    print("Default config : ", config)

    nb_iterations = config["k"]

    # Load the model
    DSGPSModel = model.ModelDSGPS(config)
    DSGPSModel.load_state_dict(checkpoint['state_dict'])
    DSGPSModel = DataParallel(DSGPSModel).to(device)

    DSGPSModel.eval()

    with torch.no_grad() :

        U_sol, loss_dic = DSGPSModel([data])

    table_data = []
    table_data.append([ data.x.size(0), 
                        loss_dic["residual_loss"][str(nb_iterations)], 
                        loss_dic["mse_loss"][str(nb_iterations)], 
                        loss_dic["mse_dirichlet_loss"][str(nb_iterations)],  
                        ]
                    )
    headers = ['Nb nodes','Residual', 'MSE', 'MSEDirichlet']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    out = {}
    out["sol_dic"] = [U_sol[str(i)].cpu() for i in range(nb_iterations)]
    out["res_dic"] = [loss_dic["residual_loss"][str(i)].item() for i in range(nb_iterations)]
    out["mse_dic"] = [loss_dic["mse_loss"][str(i)].item() for i in range(nb_iterations)]
    out["mse_dirichlet_dic"] = [loss_dic["mse_dirichlet_loss"][str(i)].item() for i in range(nb_iterations)]

    list_res.append(out["res_dic"])
    list_mse.append(out["mse_dic"])

    plt.figure(figsize=[15,6]) 
    plt.plot(out["res_dic"], 'r-', label = "Residual")
    plt.plot(out["mse_dic"], 'b-', label = "MSE w/LU")
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize = 25)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.title("Residual")
    plt.show()

def test_intermediate_states(args, checkpoint, data, device, solver = None, fw_tol = None, fw_thres = None):
    
    config_model = checkpoint["hyperparameters"]

    config_model["solver"] = solver
    config_model["fw_tol"] = fw_tol 
    config_model["fw_thres"] = fw_thres
    config_model["write_files"] = False

    DEQDSSModel = model.ModelDEQDSSTest(config_model)    
    DEQDSSModel = DataParallel(DEQDSSModel)
    DEQDSSModel.load_state_dict(checkpoint['state_dict'])

    DEQDSSModel.eval()

    with torch.no_grad() :

        # Output of the model
        out = DEQDSSModel([data.to(device)])

    print("Number of nodes : {} \t RES : {:.5e} \t MSE: {:.5e} \t BOUND_MSE: {:.5e} \t INTER_MSE: {:.5e} \t NSTEPS {}".format(
            data.sol.size(0), 
            out["res_dic"][-1], 
            out["mse_dic"][-1], 
            out["bound_mse_dic"][-1], 
            out["inter_mse_dic"][-1], 
            out['nstep'])
            )

    vis.plot_iterative_updates(data.cpu(), out, "img/test.png")
    vis.plot_mse_residual(out, "img/forward/loss.png")
    vis.plot_specific_updates(data.cpu(), out, "img/forward/")
    # vis.plot_poster(data.cpu(), out)
    # # vis.plot_final_gif(data.cpu(), out)
    # vis.save_img_gif(data.cpu(), out)
    # vis.save_images_for_gif(data.cpu(), out)

def test_poster(args, checkpoint, data, device, solver = None, fw_tol = None, fw_thres = None):
    
    config_model = checkpoint["hyperparameters"]

    config_model["solver"] = solver
    config_model["fw_tol"] = fw_tol 
    config_model["fw_thres"] = fw_thres
    config_model["write_files"] = False

    DEQDSSModel = model.ModelDEQDSSTest(config_model)
    DEQDSSModel = DataParallel(DEQDSSModel)
    DEQDSSModel.load_state_dict(checkpoint['state_dict'])

    DEQDSSModel.eval()

    with torch.no_grad() :

        # Output of the model
        out = DEQDSSModel([data.to(device)])

    print("Number of nodes : {} \t RES : {:.5e} \t MSE: {:.5e} \t BOUND_MSE: {:.5e} \t INTER_MSE: {:.5e} \t NSTEPS {}".format(
            data.sol.size(0), 
            out["res_dic"][-1], 
            out["mse_dic"][-1], 
            out["bound_mse_dic"][-1], 
            out["inter_mse_dic"][-1], 
            out['nstep'])
            )

    vis.plot_poster(data.cpu(), out)
    # vis.save_img_gif(data.cpu(), out)

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description = 'Test for unseen geometries')

    parser.add_argument("--archi",  type=str, default="autoenc", choices=["baseline", "autoenc", "double_opti"])

    parser.add_argument("--path_mesh",  type=str, default="test_mesh/")
    parser.add_argument("--saved_mesh",  type=str, default="saved_mesh/")
    parser.add_argument("--geometry",  type=str, default="mesh_f1.py")
    parser.add_argument("--name",  type=str, default="f1")
    parser.add_argument("--size_boundary",  type=str, default=str(0.09))

    parser.add_argument("--path_results",  type=str, default="results/test_2_gpu/")
    parser.add_argument("--folder_ckpt",  type=str, default="model_saved/")

    args = parser.parse_args()

    ## Initial parameters 
    precision = torch.float 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Build mesh 
    # subprocess.run(['python3', os.path.join(args.path_mesh, args.geometry), '--path', os.path.join(args.path_mesh, args.saved_mesh), '--size_boundary', args.size_boundary, '--name', args.name])

    ## Extract data 
    data = build_data(os.path.join(args.path_mesh, args.saved_mesh))

    ## Load the model
    checkpoint = torch.load(os.path.join(args.path_results, args.folder_ckpt, "best_model.pt"))

    ## Test for one sample 
    test_sample(checkpoint, data, device)

    ## Test with intermediate states 
    # test_intermediate_states(args, checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 1000)
    # test_intermediate_states(checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 20)
    # test_intermediate_states(args, checkpoint, data, device, solver = solver.broyden, fw_tol = 1.e-5, fw_thres = 1000)

    # test_poster(args, checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 1000)

