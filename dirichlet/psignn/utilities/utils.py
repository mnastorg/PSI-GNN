##################### PACKAGES #################################################
################################################################################
import os 
import random
import torch 
import numpy as np
import argparse
################################################################################
################################################################################

def get_train_parser():

    parser  =  argparse.ArgumentParser(description  =  'Pytorch DEQ-DSS Model Train')
    
    # Parameters for reading and saving files 
    parser.add_argument("--path_dataset",   type = str,     default = "dataset/",   
                                            help = "Path to read data files")
    parser.add_argument("--path_results",   type = str,     default = "results/",       
                                            help = "Path to save results")
    parser.add_argument("--comment",        type = str,     default = "Any Comments",       
                                            help = "Comment on the model")

    # Training parameters
    parser.add_argument("--seed",           type = int,         default = 123,          
                                            help = "Seed for reproducibility")
    parser.add_argument("--max_epochs",     type = int,         default = 500,          
                                            help = "Max number of epochs")
    parser.add_argument("--precision",      type = torch.dtype, default = torch.float,  
                                            help = "Precision of the model")
    parser.add_argument("--batch_size",     type = int,         default = 100,          
                                            help = "Batch size")
    parser.add_argument("--num_gpus",       type = int,         default = -1, 
                                            help = "Number of GPUs")
    parser.add_argument("--num_workers",    type = int,         default = 4,            
                                            help = "Number of workers")
    parser.add_argument("--min_loss_save",  type = float,       default = 1e10,       
                                            help = "Minimum validation value for saving")
    parser.add_argument("--gradient_clip",  type = float,       default = 1.e-2,        
                                            help = "Value of gradient clipping")
    parser.add_argument("--sup_weight",     type = float,   default = 0.0,          
                                            help = "Weight jacobian regularizer")
    parser.add_argument("--ae_weight",     type = float,   default = 0.0,          
                                            help = "Weight Autoencoder")
                                            
    # Optimizers
    parser.add_argument('--lr_deq',         type = float,   default = 0.01,                                                 
                                            help = 'Learning rate for DEQ model')
    parser.add_argument('--sched_step_deq', type = float,   default = 0.5,                                                   
                                            help = 'Step of the scheduler Reduce Plateau for DEQ optimizer')
    parser.add_argument('--lr_ae',          type = float,   default = 0.01,                                                 
                                            help = 'Learning rate for the Autoencoder process')
    parser.add_argument('--sched_step_ae',  type = float,   default = 0.5,                                                   
                                            help = 'Step of the scheduler Reduce Plateau for the Autoencoder process')

    # Solver and Jacobian parameters
    parser.add_argument("--solver",         type = str,     default = "broyden",    choices = ["broyden", "forward_iteration", "anderson", "newton"], 
                                            help = "Solver to choose") 
    parser.add_argument("--jac_weight",     type = float,   default = 0.0,          
                                            help = "Weight jacobian regularizer")

    # Model and DEQ parameters
    parser.add_argument("--latent_dim",     type = int,     default = 10, 
                                            help = "Dimension of latent space")
    parser.add_argument("--hidden_dim",     type = int,     default = 10, 
                                            help = "Dimension of hidden layers")
    parser.add_argument("--n_layers",       type = int,     default = 1, 
                                            help = "Number of layers in the GNN function")
    parser.add_argument("--fw_tol",         type = float,   default = 1.e-5, 
                                            help = "Tolerance solver forward pass")
    parser.add_argument("--fw_thres",       type = int,     default = 300, 
                                            help = "Threshold solver forward pass")
    parser.add_argument("--bw_tol",         type = float,   default = 1.e-8, 
                                            help = "Tolerance solver backward pass")
    parser.add_argument("--bw_thres",       type = int,     default = 300, 
                                            help = "Threshold solver backward pass")
    args  =  parser.parse_args()

    return args 

def get_test_parser():

    parser  =  argparse.ArgumentParser(description  =  'Pytorch DEQ-DSS Model Test')
    
    parser.add_argument("--path_dataset",   type = str,     default = "../dataset/",   
                                            help = "Path to read data files")
    parser.add_argument("--path_results",   type = str,     default = "results/",       
                                            help = "Path to save results")
    parser.add_argument("--folder_ckpt",    type = str,     default = "model_saved/",   
                                            help = "Folder to save ckpt files in results")
    parser.add_argument("--folder_logs",    type = str,     default = "logs/",          
                                            help = "Folder to save logs files in results")
    parser.add_argument("--precision",      type = torch.dtype, default = torch.float,  
                                            help = "Precision of the model")
    parser.add_argument("--seed",           type = int,         default = 123,          
                                            help = "Seed for reproducibility")
    parser.add_argument("--batch_size",     type = int,         default = 100,          
                                            help = "Batch size")
    parser.add_argument("--num_gpus",       type = int,         default = -1, 
                                            help = "Number of GPUs")
    parser.add_argument("--num_workers",    type = int,         default = 4,            
                                            help = "Number of workers")
                                            
    args  =  parser.parse_args()

    return args 

def set_seed(seed = 1234) :
    """Set the seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)