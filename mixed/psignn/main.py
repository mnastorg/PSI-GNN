##################### PACKAGES #################################################
################################################################################

import os
import sys
import shutil

import torch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel

from utilities import reader
from utilities import solver
from utilities import utils

import model 
import training_class

utils.set_seed()

################################################################################
################################################################################

if __name__ == '__main__' :

    # Get argument parser     
    args = utils.get_train_parser()
    
    # Build folder to save ckpt and logs
    if os.path.exists(args.path_results):
        shutil.rmtree(args.path_results)
    os.makedirs(args.path_results)

    path_ckpt = os.path.join(args.path_results, "ckpt")
    path_logs = os.path.join(args.path_results, "logs")
    os.makedirs(path_ckpt)
    os.makedirs(path_logs)

    # Initialize log files 
    with open(os.path.join(path_logs,'train_metrics.csv'), 'a') as f : 
        f.write("Train Metrics")
        f.close()
    with open(os.path.join(path_logs,'forward_iteration.csv'), 'a') as f : 
        f.write("Residual \t Iterations")
        f.close()
    with open(os.path.join(path_logs,'backward_iteration.csv'), 'a') as f : 
        f.write("Residual \t Iterations")
        f.close()
    with open(os.path.join(path_logs,'spectral_radius.csv'), 'a') as f : 
        f.write("Spectral Radius")
        f.close()

    # Get device CUDA if available else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read and load datasets
    dataset_train   = reader.BuildDataset(  root = args.path_dataset,     
                                            mode = 'train',   
                                            precision = args.precision)
    
    dataset_val     = reader.BuildDataset(  root = args.path_dataset,     
                                            mode = 'val',     
                                            precision = args.precision)
    
    dataset_test    = reader.BuildDataset(  root = args.path_dataset,     
                                            mode = 'test',    
                                            precision = args.precision)

    # Load batches
    loader_train    = DataListLoader(   dataset_train,  batch_size = args.batch_size,  
                                        shuffle = True, num_workers = args.num_workers)

    loader_val      = DataListLoader(   dataset_val,  batch_size = args.batch_size,  
                                        shuffle = False, num_workers = args.num_workers)
    
    loader_test     = DataListLoader(   dataset_test,  batch_size = args.batch_size,  
                                        shuffle = False, num_workers = args.num_workers) 


    # Choose appropriate fixed point solver 
    if args.solver == 'broyden' : 
        fp_solver = solver.broyden
    elif args.solver == 'forward_iteration' :
        fp_solver = solver.forward_iteration
    elif args.solver == 'anderson' : 
        fp_solver = solver.anderson
    elif args.solver == 'newton' : 
        fp_solver = solver.newton
    else :
        sys.exit("Please select an existing solver")

    # Parameters of the model
    config_model = {    "latent_dim"    : args.latent_dim,
                        "hidden_dim"    : args.hidden_dim,
                        "n_layers"      : args.n_layers,
                        "fw_tol"        : args.fw_tol,
                        "fw_thres"      : args.fw_thres,
                        "bw_tol"        : args.bw_tol,
                        "bw_thres"      : args.bw_thres,
                        "solver"        : fp_solver,
                        "path_logs"     : path_logs
                    }

    # Build the model
    DEQDSSModel = model.ModelDEQDSS(config_model)
    DEQDSSModel = DataParallel(DEQDSSModel).to(device)

    # Parameters for training 
    config_train = {    "model"         : DEQDSSModel,
                        "config_model"  : config_model,
                        "loader_train"  : loader_train,
                        "loader_val"    : loader_val,
                        "sup_weight"    : args.sup_weight,
                        "ae_weight"     : args.ae_weight,
                        "jac_weight"    : args.jac_weight,
                        "gradient_clip" : args.gradient_clip,
                        "lr_deq"        : args.lr_deq,
                        "sched_step_deq": args.sched_step_deq,
                        "lr_ae"         : args.lr_ae,
                        "sched_step_ae" : args.sched_step_ae,
                        "max_epochs"    : args.max_epochs,
                        "min_loss_save" : args.min_loss_save,
                        "path_ckpt"     : path_ckpt,
                    }   

    with open(os.path.join(path_logs,"model_config.csv"), 'w') as f : 
        
        f.write("Number of GPU used : {} \n".format(torch.cuda.device_count()))
        f.write("\n")
        f.write("Includes {} train samples, {} val samples, {} test samples \n".format(len(dataset_train),len(dataset_val), len(dataset_test)))
        f.write("Batch size {} \n".format(args.batch_size))
        f.write("\n")
        f.write("Comment on the model : {} \n".format(args.comment))
        f.write("\n")
        f.write("Model configuration : \n")
        f.write("{\n")
        for k in config_model.keys():
            f.write("'{}':'{}'\n".format(k, config_model[k]))
        f.write("}\n")
        f.write("\n")
        f.write("Training configuration : \n")
        f.write("{\n")
        for k in config_train.keys():
            f.write("'{}':'{}'\n".format(k, config_train[k]))
        f.write("}\n")
        f.write("\n")
        f.write("Number of parameters : {} \n".format(sum(p.numel() for p in DEQDSSModel.parameters() if p.requires_grad)))
        f.close()

    make_train = training_class.TrainModel(config_train)

    make_train.train_model()
    
    print("Training finished")

