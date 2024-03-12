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

import time 

import torch
from torch_geometric.data import Data
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from fenics import *
from mshr import *
from dolfin import *

from utilities import solver 
from utilities import vis 

import model_psignn as psignn
import model_dss as dss
import model_dsgps as dsgps

################################################################################
################################################################################

def solve_poisson(mesh_path, radius):

    ### Initial information
    # np.random.seed(28)
    
    param_f = np.random.uniform(-10, 10, 3)
    print("Force function : ", param_f)
    param_g = np.random.uniform(-10, 10, 6)
    print("Boundary function : ", param_g)

    L = radius

    ## Define expressions of function f and g
    f = Expression( 'A*((x[0]/L)-1)*((x[0]/L)-1) + B*(x[1]/L)*(x[1]/L) + C',
                    A = param_f[0], B = param_f[1], C = param_f[2], L = L,
                    degree = 2 
                    )
    
    g = Expression( 'A*(x[0]/L)*(x[0]/L) + B*(x[0]/L)*(x[1]/L) + C*(x[1]/L)*(x[1]/L) + D*(x[0]/L) + E*(x[1]/L) + F',
                    A = param_g[0], B = param_g[1], C = param_g[2], D = param_g[3], E = param_g[4], F = param_g[5], L = L,
                    degree = 2 
                    )
    
    # np.random.seed(1234)

    # param_f = np.random.randint(-10,10)
    # print("Force function : ", param_f)
    # param_g = np.random.randint(-10,10)
    # print("Boundary function : ", param_g)

    # f = Constant(param_f)
    # g = Constant(param_g)

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

def build_data(path_mesh, radius) : 
    
    ## Extract data and create Data object
    path_mesh = os.path.join(path_mesh)
    A_sparse_matrix, b_matrix, dof_coordinates, sol, prb_data, tags, distance = solve_poisson(path_mesh, radius)

    A_i = A_sparse_matrix.toarray()
    b_i = np.copy(b_matrix)

    row, col = np.where(A_i == 1)
    np.fill_diagonal(A_i, 0)

    C = np.c_[b_i, np.zeros(len(b_i)), np.zeros(len(b_i))]
    C[row,2] = C[row,0]
    C[row,1] = 1
    C[row,0] = 0

    A_prime = csr_matrix(A_i)
    b_prime = C

    precision = torch.float 

    prb_data_mean = torch.tensor([0.0464, -0.0006], dtype = precision)
    prb_data_std = torch.tensor([9.6267, 3.2935], dtype = precision)

    distance_mean = torch.tensor([0.0, 0.0, 0.0655], dtype = precision)
    distance_std = torch.tensor([0.0507, 0.0507, 0.0293], dtype = precision)

    aij_mean = torch.tensor(-0.5838, dtype = precision)
    aij_std = torch.tensor(0.0924, dtype = precision)

    b_prime_mean = torch.tensor([0.0002, 0.1435, -0.0006], dtype = precision)
    b_prime_std = torch.tensor([0.0507, 0.3506, 3.2935], dtype = precision)

    # Build edge_index and a_ij
    coefficients_psignn = np.asarray(find(A_sparse_matrix))
    edge_index_psignn = torch.tensor(coefficients_psignn[:2,:].astype('int'), dtype=torch.long)
    a_ij_psignn = torch.tensor(coefficients_psignn[2,:].reshape(-1,1), dtype=precision)

    # Build edge_index and a_ij
    coefficients_dss = np.asarray(sc.sparse.find(A_prime))
    edge_index_dss = torch.tensor(coefficients_dss[:2,:].astype('int'), dtype=torch.long)
    a_ij_dss = torch.tensor(coefficients_dss[2,:].reshape(-1,1), dtype=precision)
    a_ij_dss_norm = (a_ij_dss - aij_mean)/aij_std

    # Build b tensor
    b_prime =  torch.tensor(b_prime, dtype = precision)            
    b_prime_norm = (b_prime - b_prime_mean)/ b_prime_std

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

    data_psignn = Data( x = x, edge_index = edge_index_psignn, 
                        edge_attr = edge_attr, a_ij = a_ij_psignn, y = b, 
                        sol = sol, prb_data = prb_data, tags = tags,  
                        pos = pos
                    )
    
    data_dss = Data(x = sol, edge_index = edge_index_dss, edge_attr= a_ij_dss,
                    edge_attr_norm = a_ij_dss_norm, b_prime = b_prime, 
                    b_prime_norm = b_prime_norm, pos = pos, tags = tags, sol = sol
                )

    return data_psignn, data_dss

def test_sample(checkpoint, data, device):

    ### Model DSS ### 
    print("Results for Deep Statistical Solvers")

    config_model_dss = checkpoint[0]["hyperparameters"]

    DSS = dss.DeepStatisticalSolver(config_model_dss)
    DSS.load_state_dict(checkpoint[0]['state_dict'])
    DSS = DSS.to(device)

    DSS.eval()
    with torch.no_grad() :

        # Output of the model
        start_dss = time.time()

        U_sol_dss, loss_dic_dss = DSS(data[0].to(device))
        
        torch.cuda.synchronize()
        end_dss = time.time()
        dss_time = end_dss - start_dss
        
        # Compute metrics
        res_loss_dss = loss_dic_dss["residual_loss"][str(config_model_dss["k"])].item()
        mse_loss_dss = loss_dic_dss["mse_loss"][str(config_model_dss["k"])].mean().item()
        mse_dirichlet_dss = loss_dic_dss["mse_dirichlet_loss"][str(config_model_dss["k"])].item()

        relative_dss_err = np.linalg.norm(U_sol_dss[str(len(U_sol_dss)-1)].cpu().numpy() - data[0].x.cpu().numpy())/np.linalg.norm(data[0].x.cpu().numpy())
        max_square_dss_error = np.max((U_sol_dss[str(len(U_sol_dss)-1)].cpu().numpy() - data[0].x.cpu().numpy())**2)

    table_data = []
    table_data.append([data[0].x.size(0), res_loss_dss, mse_loss_dss, relative_dss_err*100, max_square_dss_error, mse_dirichlet_dss])
    headers = ['Nb nodes','Residual', 'MSE', 'REL(%)', 'MAX', 'MSEDirichlet']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    ### Model DSGPS ### 
    print("Results for DSGPS")

    config_model_dsgps = checkpoint[1]["hyperparameters"]
    config_model_dsgps["k"] = 100

    DSGPS = dsgps.ModelDSGPS(config_model_dsgps)
    DSGPS.load_state_dict(checkpoint[1]['state_dict'])
    DSGPS = DSGPS.to(device)

    DSGPS.eval()
    with torch.no_grad() :

        # Output of the model
        start_dsgps = time.time()
        U_sol_dsgps, loss_dic_dsgps = DSGPS(data[1].to(device))
        torch.cuda.synchronize()
        end_dsgps = time.time()
        dsgps_time = end_dsgps - start_dsgps
      
        # Compute metrics
        res_loss_dsgps = loss_dic_dsgps["residual_loss"][str(config_model_dsgps["k"])].item()
        mse_loss_dsgps = loss_dic_dsgps["mse_loss"][str(config_model_dsgps["k"])].mean().item()
        mse_dirichlet_dsgps = loss_dic_dsgps["mse_dirichlet_loss"][str(config_model_dsgps["k"])].item()

        relative_dsgps_err = np.linalg.norm(U_sol_dsgps[str(len(U_sol_dsgps)-1)].cpu().numpy() - data[1].sol.cpu().numpy())/np.linalg.norm(data[0].sol.cpu().numpy())
        max_square_dsgps_error = np.max((U_sol_dsgps[str(len(U_sol_dsgps)-1)].cpu().numpy() - data[1].sol.cpu().numpy())**2)
       
    table_data = []
    table_data.append([data[1].x.size(0), res_loss_dsgps, mse_loss_dsgps, relative_dsgps_err*100, max_square_dsgps_error, mse_dirichlet_dsgps])
    headers = ['Nb nodes','Residual', 'MSE', 'REL(%)', 'MAX', 'MSEDirichlet']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    ### Model PSIGNN ### 
    print("Results for PSIGNN")

    config_model_psignn = checkpoint[2]["hyperparameters"]
    # modify hyperparameters for larger geometries
    config_model_psignn["fw_tol"] = 1.e-5
    config_model_psignn["fw_thres"] = 1500

    PSIGNN = psignn.ModelPSIGNN(config_model_psignn)
    PSIGNN.load_state_dict(checkpoint[2]['state_dict'])
    PSIGNN = PSIGNN.to(device)

    PSIGNN.eval()
    with torch.no_grad() :

        # Output of the model
        start_psignn = time.time()
        U_sol_psignn, loss_dic_psignn = PSIGNN(data[1].to(device))
        torch.cuda.synchronize()
        end_psignn = time.time()
        psignn_time = end_psignn - start_psignn

        # Compute metrics
        res_loss_psignn = loss_dic_psignn["residual_loss"].item()
        mse_loss_psignn = loss_dic_psignn["mse_loss"].mean().item()
        mse_dirichlet_psignn = loss_dic_psignn["mse_dirichlet_loss"].item()
        nsteps_psignn = loss_dic_psignn["nsteps"]

        relative_psignn_err = np.linalg.norm(U_sol_psignn.cpu().numpy() - data[1].sol.cpu().numpy())/np.linalg.norm(data[1].sol.cpu().numpy())
        max_square_psignn_error = np.max((U_sol_psignn.cpu().numpy() - data[1].sol.cpu().numpy())**2)

    table_data = []
    table_data.append([data[0].x.size(0), 
                       res_loss_psignn, 
                       mse_loss_psignn, 
                       relative_psignn_err*100,
                       max_square_psignn_error,
                       mse_dirichlet_psignn, 
                       nsteps_psignn])
    headers = ['Nb nodes','Residual', 'MSE', 'REL(%)', 'MAX', 'MSEDirichlet', 'Nstep']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    torch.cuda.empty_cache()
    
    # list_sol_ml = [U_sol_dss[str(len(U_sol_dss)-1)].cpu().numpy(), U_sol_dsgps[str(len(U_sol_dsgps)-1)].cpu().numpy(), U_sol_psignn.cpu().numpy()]
    # list_sol_ex = [data[0].x.cpu().numpy(), data[1].sol.cpu().numpy()]

    # return list_sol_ml, list_sol_ex #mse_loss_dss, mse_loss_dsgps, mse_loss_psignn, nsteps_psignn, dss_time, dsgps_time, psignn_time
    return mse_loss_dss, mse_loss_dsgps, mse_loss_psignn, nsteps_psignn, dss_time, dsgps_time, psignn_time

def test_intermediate_states(checkpoint, data, device): 
    
    ckptpsignn = checkpoint[0]
    config_model = ckptpsignn["hyperparameters"]

    DEQDSSModel = psignn.ModelIterative(config_model)    
    DEQDSSModel.load_state_dict(ckptpsignn['state_dict'])
    DEQDSSModel = DEQDSSModel.to(device)

    DEQDSSModel.eval()

    with torch.no_grad() :

        # Output of the model
        out = DEQDSSModel(data[0].to(device))

    ckptdss = checkpoint[1]
    config_model = ckptdss["hyperparameters"]

    DEQDSSModel = dss.Model(config_model)    
    DEQDSSModel.load_state_dict(ckptdss['state_dict'])
    DEQDSSModel = DEQDSSModel.to(device)

    DEQDSSModel.eval()

    with torch.no_grad() :

        # Output of the model
        U_sol_dss, loss_dic = DEQDSSModel(data[1].to(device))

    # print("Number of nodes : {} \t RES : {:.5e} \t MSE: {:.5e} \t BOUND_MSE: {:.5e} \t INTER_MSE: {:.5e} \t NSTEPS {}".format(
    #         data.sol.size(0), 
    #         out["res_dic"][-1], 
    #         out["mse_dic"][-1], 
    #         out["bound_mse_dic"][-1], 
    #         out["inter_mse_dic"][-1], 
    #         out['nstep'])
    #         )

    # vis.plot_iterative_updates(data.cpu(), out, "img/test.png")
    # vis.plot_mse_residual(out, "img/forward/loss.png")
    # vis.plot_specific_updates(data.cpu(), out, "img/forward/")
    # vis.plot_poster(data.cpu(), out)
    # # vis.plot_final_gif(data.cpu(), out)
    # vis.save_img_gif(data.cpu(), out)
    # vis.save_images_for_gif(data.cpu(), out)

    return out["bound_mse_dic"], loss_dic["mse_dirichlet"]

def test_several_init(checkpoint, data, device):
    
    config_model = checkpoint["hyperparameters"]
    config_model["fw_tol"] = 5.e-5

    DEQDSSModel = psignn.ModelPSIGNNIterative(config_model)    
    DEQDSSModel.load_state_dict(checkpoint['state_dict'])
    DEQDSSModel = DEQDSSModel.to(device)
    
    index_interior = torch.where(data.tags==0)[0]
    index_dirichlet = torch.where(data.tags==1)[0]

    res_list = []

    for k in range(3):

        if k == 1 :
            a, b = -1000, 1000
            data.x[index_interior,:] = data.sol[index_interior,:] + (b - a)*torch.rand_like(data.sol[index_interior,:]) + a
            data.x[index_dirichlet,:] = data.sol[index_dirichlet,:] 
        if k == 2 :
            a, b = 0, 0.1
            data.x[index_interior,:] = data.sol[index_interior,:] + (b - a)*torch.rand_like(data.sol[index_interior,:]) + a
            data.x[index_dirichlet,:] = data.sol[index_dirichlet,:] 

        DEQDSSModel.eval()

        with torch.no_grad() :

            # Output of the model
            out = DEQDSSModel(data.to(device))

        res_list.append(out["res_dic"])

    return res_list

def test_poster(args, checkpoint, data, device, solver = None, fw_tol = None, fw_thres = None):
    
    config_model = checkpoint["hyperparameters"]

    config_model["solver"] = solver
    config_model["fw_tol"] = fw_tol 
    config_model["fw_thres"] = fw_thres
    config_model["write_files"] = False

    DEQDSSModel = psignn.Model(config_model)
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
    subprocess.run(['python3', os.path.join(args.path_mesh, args.geometry), '--path', os.path.join(args.path_mesh, args.saved_mesh), '--size_boundary', args.size_boundary, '--name', args.name])

    ## Extract data 
    data = build_data(args)

    ## Load the model
    checkpoint = torch.load(os.path.join(args.path_results, args.folder_ckpt, "best_model.pt"))

    ## Test for one sample 
    # test_sample(args, checkpoint, data, device)

    ## Test with intermediate states 
    test_intermediate_states(args, checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 1000)
    # test_intermediate_states(checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 20)
    # test_intermediate_states(args, checkpoint, data, device, solver = solver.broyden, fw_tol = 1.e-5, fw_thres = 1000)

    # test_poster(args, checkpoint, data, device, solver = solver.forward_iteration, fw_tol = 1.e-5, fw_thres = 1000)

