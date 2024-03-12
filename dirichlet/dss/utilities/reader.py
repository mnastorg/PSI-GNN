##################### PACKAGES #################################################
################################################################################

import os
import sys

import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split

import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data

################################################################################
################################################################################

class BuildDataset(InMemoryDataset):

    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None, mode = None, precision = torch.float):
        
        self.mode = mode
        self.precision = precision
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if self.mode == 'train' :
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == 'val' :
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.mode == 'test' :
            self.data, self.slices = torch.load(self.processed_paths[2])
        else :
            sys.exit()

    @property
    def raw_file_names(self):
        files = ['A_prime.npy', 'b_prime.npy', 'sol.npy', 'coordinates.npy', 'tags.npy']
        return files

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'data/')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_dss/')

    def process(self):

        data_list = []

        list_A_prime = np.load(self.raw_dir + self.raw_file_names[0], allow_pickle = True)
        list_b_prime = np.load(self.raw_dir + self.raw_file_names[1], allow_pickle = True)
        list_sol = np.load(self.raw_dir + self.raw_file_names[2], allow_pickle = True)
        list_coordinates = np.load(self.raw_dir + self.raw_file_names[3], allow_pickle = True)
        list_tags = np.load(self.raw_dir + self.raw_file_names[4], allow_pickle = True)

        self.aij_mean = torch.tensor(-0.5838, dtype = self.precision)
        self.aij_std = torch.tensor(0.0924, dtype = self.precision)

        self.b_prime_mean = torch.tensor([0.0002, 0.1435, -0.0006], dtype = self.precision)
        self.b_prime_std = torch.tensor([0.0507, 0.3506, 3.2935], dtype = self.precision)

        for i in range(len(list_A_prime)):
            
            # Build edge_index and a_ij
            coefficients = np.asarray(sc.sparse.find(list_A_prime[i]))
            edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)
            a_ij = torch.tensor(coefficients[2,:].reshape(-1,1), dtype=self.precision)
            a_ij_norm = (a_ij - self.aij_mean)/self.aij_std

            # Build b tensor
            b_prime =  torch.tensor(list_b_prime[i], dtype = self.precision)            
            b_prime_norm = (b_prime - self.b_prime_mean)/ self.b_prime_std

            # Extract exact solution
            sol = torch.tensor(list_sol[i], dtype = self.precision)

            # Extract coordinates
            pos = torch.tensor(list_coordinates[i], dtype = self.precision)

            tags = torch.tensor(list_tags[i], dtype = self.precision)

            data = Data(    x = sol, edge_index = edge_index, a_ij= a_ij,
                            a_ij_norm = a_ij_norm, b_prime = b_prime, 
                            b_prime_norm = b_prime_norm, pos = pos, tags = tags,
                            sol = sol
                        )

            data_list.append(data)

        data_train_, data_val = train_test_split(data_list, test_size=0.2, shuffle=False)
        data_train, data_test = train_test_split(data_train_, test_size=0.25, shuffle=False)
        
        if self.mode == 'train' :
            data, slices = self.collate(data_train)
            torch.save((data, slices), self.processed_paths[0])
        elif self.mode == 'val' :
            data, slices = self.collate(data_val)
            torch.save((data, slices), self.processed_paths[1])
        elif self.mode == 'test':
            data, slices = self.collate(data_test)
            torch.save((data, slices), self.processed_paths[2])
