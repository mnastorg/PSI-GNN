##################### PACKAGES #################################################
################################################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from math import *
import os

import numpy as np
import time
import torch

from utilities import utils

utils.set_seed()
################################################################################
################################################################################

class TrainModel:
    def __init__(self, config):

        ### Loaders and Device
        self.loader_train   = config["loader_train"]
        self.loader_val     = config["loader_val"]

        ### Model
        self.model          = config["model"]
        self.config_model   = config["config_model"]

        ### Optimizer and scheduler
        self.lr         = config["lr"]

        ### Train parameters
        self.path_ckpt      = config["path_ckpt"]
        self.path_logs      = self.config_model["path_logs"]
        self.min_loss_save  = config["min_loss_save"]
        self.max_epochs     = config["max_epochs"]
        self.gradient_clip  = config["gradient_clip"]
        # self.sup_weight     = config["sup_weight"]

        self.training_time  = 0
        self.hist_train     = {"loss":[], "residual_loss":[], "encoder_loss":[], "autoencoder_loss":[], "mse_loss":[]}
        self.hist_val       = {"loss":[], "residual_loss":[], "encoder_loss":[], "autoencoder_loss":[], "mse_loss":[]}

        self.createOptimizerAndScheduler()

    def createOptimizerAndScheduler(self):

        self.opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        # self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode = 'min', factor = self.sched_step_deq)

    def save_model(self, state, dirName = None, model_name = None):

        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(dirName, model_name)
        path = open(save_path, mode="wb")
        torch.save(state, path)
        path.close()

    def load_model(self, path):

        #Load checkpoint
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.opt.load_state_dict(checkpoint['opt_deq'])
        self.min_loss_save = checkpoint['min_loss_save']
        self.hist_train = checkpoint['hist_train']
        self.hist_val = checkpoint['hist_val']
        self.training_time = checkpoint['training_time']

    def make_plot(self, ax, ckpt_train, ckpt_val, xlabel, ylabel):
        ax.plot(ckpt_train, '-b', linewidth=1, label = "Train")
        ax.plot(ckpt_val, '-r', linewidth=1, label = "Valid")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.legend()

    def plot_losses(self, checkpoint):
        
        fig = plt.figure(figsize = [10,8], constrained_layout=True)
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure = fig)

        ax1 = fig.add_subplot(spec[0,0])
        self.make_plot(ax1, checkpoint["hist_train"]["loss"], checkpoint["hist_val"]["loss"], "Epoch", "Training Loss")

        ax2 = fig.add_subplot(spec[0,1])
        self.make_plot(ax2, checkpoint["hist_train"]["residual_loss"], checkpoint["hist_val"]["residual_loss"], "Epoch", "Residual Loss")
        
        ax4 = fig.add_subplot(spec[1,0])
        self.make_plot(ax4, checkpoint["hist_train"]["mse_loss"], checkpoint["hist_val"]["mse_loss"], "Epoch", "MSE Loss")

        ax5 = fig.add_subplot(spec[1,1])
        self.make_plot(ax5, checkpoint["hist_train"]["encoder_loss"], checkpoint["hist_val"]["encoder_loss"], "Epoch", "Encoder Loss")

        ax6 = fig.add_subplot(spec[2,0])
        self.make_plot(ax6, checkpoint["hist_train"]["autoencoder_loss"], checkpoint["hist_val"]["autoencoder_loss"], "Epoch", "Autoencoder Loss")

        fig.suptitle("Evolution of training losses through epoch")

        fig.savefig(os.path.join(self.path_logs, "track_losses.png"), dpi = 100)
        
        plt.close(fig)

    def plot_gradients(self, current_epoch):

        names = [n.replace('module.', '') for n, p in self.model.named_parameters() if p.grad is not None and p.requires_grad]
        norm_grad = [p.grad.detach().data.norm(2).cpu() for n, p in self.model.named_parameters() if p.grad is not None and p.requires_grad]

        fig = plt.figure(figsize = [15,10])
        plt.bar(names, norm_grad, width = 0.5, linewidth = 1.0)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel("Gradient norm")
        plt.title("Gradient Norm at epoch {}".format(current_epoch))
        plt.savefig(os.path.join(self.path_logs, "gradients.png"), bbox_inches='tight')
        plt.close(fig)

    def train_loop(self, current_epoch) : 

        # Initialize losses
        cumul_loss, cumul_residual_loss, cumul_mse_loss = 0, 0, 0
        cumul_encoder_loss, cumul_autoencoder_loss = 0, 0

        run_loss, run_residual_loss, run_mse_loss = 0, 0, 0
        run_encoder_loss, run_autoencoder_loss = 0, 0

        # Model in train mode
        self.model.train()
        
        cumul = 0

        # Batch loop  
        for i, train_batch in enumerate(self.loader_train):
            
            self.opt.zero_grad()

            # Output of the model
            U_sol, loss_dic = self.model(train_batch)

            # Compute and optimize the training loss
            loss =  loss_dic["train_loss"].mean()

            loss.backward()
            
            # Apply gradient clipping on the deqdss module
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.gradient_clip)
            
            self.opt.step()
                        
            # Accumulate the losses
            cumul_loss += loss.item()
            cumul_residual_loss += loss_dic["residual_loss"][str(self.config_model["k"])].mean().item()
            cumul_mse_loss += loss_dic["mse_loss"][str(self.config_model["k"])].mean().item()
            cumul_encoder_loss += loss_dic["encoder_loss"][str(self.config_model["k"])].mean().item()
            cumul_autoencoder_loss += loss_dic["autoencoder_loss"][str(self.config_model["k"])].mean().item()

            # Accumulate running loss and print %
            run_loss += loss.item()
            run_residual_loss += loss_dic["residual_loss"][str(self.config_model["k"])].mean().item()
            run_mse_loss += loss_dic["mse_loss"][str(self.config_model["k"])].mean().item()
            run_encoder_loss += loss_dic["encoder_loss"][str(self.config_model["k"])].mean().item()
            run_autoencoder_loss += loss_dic["autoencoder_loss"][str(self.config_model["k"])].mean().item()

            cumul += 1
            if i == ceil(0.25*len(self.loader_train)) or i == ceil(0.5*len(self.loader_train)) or i == ceil(0.75*len(self.loader_train)) : 
                with open(os.path.join(self.path_logs, 'train_metrics.csv'), 'a') as f : 
                    f.write(
                    "\nEpoch {}, {:d}% \t Loss : {:.4e} \t Res : {:.4e} \t Enc : {:.4e} \t AEnc : {:.4e} \t MSE : {:.4e}".format(
                    current_epoch,
                    int(i * 100 / len(self.loader_train)),
                    run_loss / cumul ,
                    run_residual_loss / cumul,
                    run_encoder_loss / cumul,
                    run_autoencoder_loss / cumul,
                    run_mse_loss / cumul)
                    )
                    f.close()
                run_loss, run_residual_loss, run_mse_loss, run_encoder_loss, run_autoencoder_loss = 0, 0, 0, 0, 0
                cumul = 0

        self.hist_train["loss"].append(cumul_loss / len(self.loader_train))
        self.hist_train["residual_loss"].append(cumul_residual_loss / len(self.loader_train))
        self.hist_train["encoder_loss"].append(cumul_encoder_loss / len(self.loader_train))
        self.hist_train["autoencoder_loss"].append(cumul_autoencoder_loss / len(self.loader_train))
        self.hist_train["mse_loss"].append(cumul_mse_loss / len(self.loader_train))
        
        with open(os.path.join(self.path_logs,'train_metrics.csv'), 'a') as f : 
            f.write(
                "\nTraining Epoch {} : \t Train : {:.5e} \t Res : {:.5e} \t Enc : {:.5e} \t AE : {:.5e} \t MSE : {:.5e}".format(
                current_epoch,
                cumul_loss / len(self.loader_train),
                cumul_residual_loss / len(self.loader_train),
                cumul_encoder_loss / len(self.loader_train),
                cumul_autoencoder_loss / len(self.loader_train),
                cumul_mse_loss / len(self.loader_train))
            )
            f.close()

    def validation_loop(self, current_epoch):
        
        # Initialize losses 
        cumul_val_loss, cumul_val_residual_loss, cumul_val_mse_loss = 0, 0, 0
        cumul_val_encoder_loss, cumul_val_autoencoder_loss = 0, 0

        # Model in eval mode
        self.model.eval()
        
        # Deactivate gradients
        with torch.no_grad():
            
            # Batch loop
            for val_batch in self.loader_val:
            
                # Output of the model
                U_sol, loss_dic = self.model(val_batch)
                
                loss =  loss_dic["train_loss"].mean()

                # Accumulate the losses
                cumul_val_loss += loss.item()
                cumul_val_residual_loss += loss_dic["residual_loss"][str(self.config_model["k"])].mean().item()
                cumul_val_mse_loss += loss_dic["mse_loss"][str(self.config_model["k"])].mean().item()
                cumul_val_encoder_loss += loss_dic["encoder_loss"][str(self.config_model["k"])].mean().item()
                cumul_val_autoencoder_loss += loss_dic["autoencoder_loss"][str(self.config_model["k"])].mean().item()

        self.hist_val["loss"].append(cumul_val_loss / len(self.loader_val))
        self.hist_val["residual_loss"].append(cumul_val_residual_loss / len(self.loader_val))
        self.hist_val["encoder_loss"].append(cumul_val_encoder_loss / len(self.loader_val))
        self.hist_val["autoencoder_loss"].append(cumul_val_autoencoder_loss / len(self.loader_val))
        self.hist_val["mse_loss"].append(cumul_val_mse_loss / len(self.loader_val))

        with open(os.path.join(self.path_logs,'train_metrics.csv'), 'a') as f : 
            f.write(
                "\nValidation Epoch {} : \t Train : {:.5e} \t Res : {:.5e} \t Enc : {:.5e} \t AE : {:.5e} \t MSE : {:.5e}".format(
                current_epoch,
                cumul_val_loss / len(self.loader_val),
                cumul_val_residual_loss / len(self.loader_val),
                cumul_val_encoder_loss / len(self.loader_val),
                cumul_val_autoencoder_loss / len(self.loader_val),
                cumul_val_mse_loss / len(self.loader_val))
            )
            f.close()

    def train_model(self):
        
        # Epoch loop
        for epoch in range(self.max_epochs):

            time_counter = time.time()

            print("Epoch number : ", epoch)

            self.train_loop(epoch)
            
            self.validation_loop(epoch)
            
            # self.sched.step(self.hist_val["residual_loss"][-1])
            # self.sched.step(self.hist_val["loss"][-1])
            
            self.training_time = self.training_time + (time.time() - time_counter)

            if self.opt.param_groups[0]["lr"] <= 1.e-7 : 
                with open(self.logs_train, 'a') as f : 
                    f.write("\nTraining exit because both learning rates too low !")
                break

            # checkpoint current model
            checkpoint = {  'epoch'           : epoch,
                            'hyperparameters' : self.config_model,
                            'state_dict'      : self.model.module.state_dict(),                            
                            'hist_train'      : self.hist_train,
                            'hist_val'        : self.hist_val,
                            'opt_deq'         : self.opt.state_dict(),
                            'training_time'   : self.training_time
                            }
            self.save_model(checkpoint, dirName = self.path_ckpt, model_name = "running_model")

            # save checkpoint to best_model if residual validation loss is <= min loss save
            if self.hist_val["residual_loss"][-1] <= self.min_loss_save :
                self.save_model(checkpoint, dirName = self.path_ckpt, model_name = "best_model")
                self.min_loss_save = self.hist_val["residual_loss"][-1]    
                with open(os.path.join(self.path_logs,'train_metrics.csv'), 'a') as f : 
                    f.write("\nTraining Epoch {} finished, took current epoch {:.2f}s, cumulative time {:.2f}s".format(epoch, time.time() - time_counter, self.training_time))
                    f.write("\nCurrent Learning rate : {}".format(self.opt.param_groups[0]["lr"]))
                    f.write("\nMODEL SAVED")
                    f.close()

            else:
                with open(os.path.join(self.path_logs,'train_metrics.csv'), 'a') as f : 
                    f.write("\nTraining Epoch {} finished, took current epoch {:.2f}s, cumulative time {:.2f}s".format(epoch, time.time() - time_counter, self.training_time))
                    f.write("\nCurrent Learning rate : {}".format(self.opt.param_groups[0]["lr"]))
                    f.close()

            if epoch % 2 == 0 :
                self.plot_losses(checkpoint)
                self.plot_gradients(epoch)
                                
        #save model
        self.save_model(checkpoint, dirName=self.path_ckpt, model_name="final_model")

        return self.model