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

import vis
import model_test as model 
from utilities import reader
#################

def test_dataset(list_ckpt, list_names, dataloader, device):

    data = []

    for i in range(len(list_ckpt)):
        
        print("Evaluation model : ", list_names[i])
        checkpoint = list_ckpt[i]

        config = checkpoint["hyperparameters"]
        print("Default config : ", config)

        nb_iterations = config["k"]

        # Load the model
        DSSModel = model.DeepStatisticalSolver(config)
        DSSModel.load_state_dict(checkpoint['state_dict'])
        DSSModel = DataParallel(DSSModel).to(device)

        cumul_test_res_loss = []
        cumul_test_mse_loss = []
        cumul_dirichlet = []

        # Test the model on the test dataset
        DSSModel.eval()
        
        with torch.no_grad() :

            for test_data in tqdm(dataloader) :
                
                # Output of the model
                U_sol, loss_dic = DSSModel(test_data)

                cumul_test_res_loss.append(loss_dic["residual_loss"][str(nb_iterations)].mean().item())
                cumul_test_mse_loss.append(loss_dic["mse_loss"][str(nb_iterations)].mean().item())
                cumul_dirichlet.append(loss_dic["mse_dirichlet_loss"][str(nb_iterations)].mean().item())

        data.append([list_names[i],
                np.mean(cumul_test_res_loss),
                np.mean(cumul_test_mse_loss),
                np.mean(cumul_dirichlet),
                ])

    print("std Res : ", np.std(cumul_test_res_loss))
    print("std MSE : ", np.std(cumul_test_mse_loss))
    print("std Dirichlet : ", np.std(cumul_dirichlet))
    headers = ['Name','Residual', 'MSE', 'MSEDirichlet']
    print(tabulate(data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

def solution_sample(checkpoint, data, device):
    
    config = checkpoint["hyperparameters"]
    print("Default config : ", config)
    
    nb_iterations = config["k"]

    # Load the model
    DSSModel = model.DeepStatisticalSolver(config)
    DSSModel.load_state_dict(checkpoint['state_dict'])
    DSSModel = DataParallel(DSSModel).to(device)

    DSSModel.eval()
    
    with torch.no_grad() :

        U_sol, loss_dic = DSSModel([data])

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
    
    vis.extract_images_results(data.cpu(), out, "img")

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