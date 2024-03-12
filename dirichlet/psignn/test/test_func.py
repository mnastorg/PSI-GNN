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
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader, DataLoader
from torch_sparse import SparseTensor

import vis
import model_test as model 
from utilities import reader

from importlib import reload
reload(model)

#################

def errors_batch(u, batch):
    
    sparse_matrix = SparseTensor(   row = batch.edge_index[0], 
                                    col = batch.edge_index[1], 
                                    value = batch.a_ij.ravel(), 
                                    sparse_sizes=(batch.num_nodes, batch.num_nodes)
                                )
    
    residual =  sparse_matrix.matmul(u) - batch.y

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

        normalized_residual = torch.linalg.norm(residual[index,:])/torch.linalg.norm(batch.y[index,:])
        res_norm_list.append(normalized_residual.item())
        
        mse_val = torch.mean((u[index,:] - batch.sol[index,:])**2)
        mse_list.append(mse_val.item())

        normalized_relative = torch.linalg.norm(u[index,:] - batch.sol[index,:])/torch.linalg.norm(batch.sol[index,:])
        rel_list.append(normalized_relative.item())

        mse_bound = torch.mean((u[index,:][bound] - batch.sol[index,:][bound])**2)
        mse_bound_list.append(mse_bound.item())

    return res_list, res_norm_list, mse_list, rel_list, mse_bound_list

def test_dataset(list_ckpt, list_names, dataloader, device):

    for i in range(len(list_ckpt)):
        
        data = []
        
        print("Evaluation model : ", list_names[i])
        checkpoint = list_ckpt[i]

        config = checkpoint["hyperparameters"]
        print("Default config : ", config)

        # Load the model
        PSIGNNModel = model.ModelPSIGNN(config)
        PSIGNNModel.load_state_dict(checkpoint['state_dict'])
        PSIGNNModel = PSIGNNModel.to(device)

        res_list, res_norm_list, mse_list, rel_list, mse_bound_list = [], [], [], [], []

        # Test the model on the test dataset
        PSIGNNModel.eval()
        
        with torch.no_grad() :

            for test_data in tqdm(dataloader) :
                
                # Output of the model
                U_sol = PSIGNNModel(test_data.to(device))

                res, res_norm, mse, rel, mse_bound = errors_batch(U_sol, test_data)

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

def solution_sample(checkpoint, data, device):
    
    config = checkpoint["hyperparameters"]
    print("Default config : ", config)

    # config["k"] = 200

    # nb_iterations = config["k"]

    # Load the model
    PSIGNNModelIterative = model.ModelPSIGNNIterative(config)
    PSIGNNModelIterative.load_state_dict(checkpoint['state_dict'])
    PSIGNNModelIterative = DataParallel(PSIGNNModelIterative).to(device)

    PSIGNNModelIterative.eval()
    
    with torch.no_grad() :

        out = PSIGNNModelIterative([data])

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

if __name__ == '__main__' :

    parser  =  argparse.ArgumentParser(description  =  'Test Deep Statistical Solvers')
    parser.add_argument("--path_dataset",   type = str,     default = "dataset/",   
                                            help = "Path to read data files")
    parser.add_argument("--path_results",   type = str,     default = "results/",       
                                            help = "Path to save results")
    parser.add_argument("--batch_size",     type = int,       default = 50,
                                            help = 'Size of the batch')
    args  =  parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on {}. GPU model : {}".format(device, torch.cuda.get_device_name(0)))

    dataset_test = reader.BuildDataset(root = args.path_dataset, mode = 'test', precision = torch.float)
    loader_test = DataListLoader(dataset_test,  batch_size = args.batch_size, shuffle = False, num_workers = 0)
    print("Number of samples in the test dataset : ", len(dataset_test))

    best_model = torch.load(args.path_results)
    test_dataset([best_model], ['DSS'], loader_test, device)