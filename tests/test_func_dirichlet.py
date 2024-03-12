### PACKAGES ###
import warnings
warnings.filterwarnings('ignore')

import sys 
sys.path.append("..")

import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

import torch
from torch_sparse import SparseTensor

import vis

from dirichlet.dss import model as dss
from dirichlet.dsgps import model as dsgps
from dirichlet.psignn import model as psignn

from utilities import reader

#################

def dss_residual(U, batch):

    y = batch.b_prime
    edge_index = batch.edge_index
    a_ij = batch.a_ij

    B0 = y[:,0].reshape(-1,1)
    B1 = y[:,1].reshape(-1,1)
    B2 = y[:,2].reshape(-1,1)

    p1 = (1 - B1)*(-B0) + B1*(U - B2)

    from_ = edge_index[0,:].reshape(-1,1).type(torch.int64)
    to_ = edge_index[1,:].reshape(-1,1).type(torch.int64)
    u_i = torch.gather(U, 0, from_)
    u_j = torch.gather(U, 0, to_)

    F_bar = a_ij*(u_j-u_i)
    M = U*0
    F_bar_sum = M.scatter_add(0,from_,F_bar)

    residual = p1 + F_bar_sum

    return residual

def sparse_residual(u, batch):

    sparse_matrix = SparseTensor(   row = batch.edge_index[0], 
                                    col = batch.edge_index[1], 
                                    value = batch.a_ij.ravel(), 
                                    sparse_sizes=(batch.num_nodes, batch.num_nodes)
                                )
    
    residual =  sparse_matrix.matmul(u) - batch.y

    return residual

def errors_batch(u, batch, id):
    
    if id == 0 : 
        residual = dss_residual(u, batch)
    else : 
        residual = sparse_residual(u, batch)
    
    bs = list(torch.unique(batch.batch).cpu().numpy())

    res_list = []
    res_norm_list = []
    mse_list = []
    rel_list = []
    mse_bound_list = []

    for i in bs :   
        
        index = torch.where(batch.batch == i)[0]
        
        tags = batch.tags[index]
        bound = torch.where(tags == 1)[0]

        square_residual = torch.mean(residual[index,:]**2)
        res_list.append(square_residual.item())

        if id == 0 : 
            rhs = (batch.b_prime[:,0] + batch.b_prime[:,2]).view(-1,1)
            normalized_residual = torch.linalg.norm(residual[index,:]) / torch.linalg.norm(rhs[index,:])        
        else : 
            normalized_residual = torch.linalg.norm(residual[index,:]) / torch.linalg.norm(batch.y[index,:])
        
        res_norm_list.append(normalized_residual.item())
        
        mse_val = torch.mean((u[index,:] - batch.sol[index,:])**2)
        mse_list.append(mse_val.item())

        normalized_relative = torch.linalg.norm(u[index,:] - batch.sol[index,:])/torch.linalg.norm(batch.sol[index,:])
        rel_list.append(normalized_relative.item())

        mse_bound = torch.mean((u[index,:][bound] - batch.sol[index,:][bound])**2)
        mse_bound_list.append(mse_bound.item())

    return res_list, res_norm_list, mse_list, rel_list, mse_bound_list

def select_model(config, checkpoint, id, device):

    # Load the model
    if id == 0 : 
        model = dss.DeepStatisticalSolver(config)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    elif id == 1 : 
        model = dsgps.ModelDSGPS(config)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    elif id == 2 : 
        model = psignn.ModelDEQDSS(config)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    return model 

def test_model(list_ckpt, list_names, list_ids, list_dataloader, device):

    for i in range(len(list_ckpt)):
        
        data = []
        
        checkpoint = list_ckpt[i]
        
        config = checkpoint["hyperparameters"]

        model = select_model(config, checkpoint, list_ids[i], device)

        res_list, res_norm_list, mse_list, rel_list, mse_bound_list = [], [], [], [], []

        # Test the model on the test dataset
        model.eval()
        
        dataloader = list_dataloader[i]

        with torch.no_grad() :

            for test_data in tqdm(dataloader) :
                
                # Output of the model
                U_sol = model.inference(test_data.to(device))

                res, res_norm, mse, rel, mse_bound = errors_batch(U_sol, test_data, list_ids[i])

                res_list += res
                res_norm_list += res_norm
                mse_list += mse
                rel_list += rel
                mse_bound_list += mse_bound

        data.append([list_names[i],
                np.mean(res_list),
                np.mean(res_norm_list),
                np.mean(mse_list),
                np.mean(rel_list),
                np.mean(mse_bound_list)
                ])
        
        print("std Res : ", np.std(res_list))
        print("std ResNorm : ", np.std(res_norm_list))
        print("std MSE : ", np.std(mse_list))
        print("std Rel : ", np.std(rel_list))
        print("std MSEBound : ", np.std(mse_bound_list))

        headers = ['Name','Residual', 'ResidualNorm', 'MSE', 'Rel', 'MSEBound']
        print(tabulate(data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

def solution_sample(checkpoint, data, id, device):
    
    config = checkpoint["hyperparameters"]
    print("Default config : ", config)

    model = select_model(config, checkpoint, id)

    model.eval()
    
    with torch.no_grad() :

        out = model.iterative_inference([data])

    table_data = []
    table_data.append([ data.sol.size(0), 
                        out["res_dic"][-1], 
                        out["mse_dic"][-1], 
                        out["bound_mse_dic"][-1], 
                        out["inter_mse_dic"][-1], 
                        out['nstep']
                        ]
                    )
    headers = ['Nb nodes','Residual', 'MSE', 'MSEDirichlet', 'MSEInterior', 'Nstep']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    # out = {}
    # out["sol_dic"] = [U_sol[str(i)].cpu() for i in range(nb_iterations)]
    # out["res_dic"] = [loss_dic["residual_loss"][str(i)].item() for i in range(nb_iterations)]
    # out["mse_dic"] = [loss_dic["mse_loss"][str(i)].item() for i in range(nb_iterations)]
    # out["mse_dirichlet_dic"] = [loss_dic["mse_dirichlet_loss"][str(i)].item() for i in range(nb_iterations)]
    
    vis.extract_images_results(data.cpu(), out, "img")
    # vis.plot_paper_2(data.cpu(), out, "img")