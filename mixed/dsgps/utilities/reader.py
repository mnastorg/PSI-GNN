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
        files = ['A_sparse_matrix.npy', 
                 'b_matrix.npy', 
                 'sol.npy', 
                 'prb_data.npy', 
                 'tags.npy', 
                 'coordinates.npy', 
                 'distance.npy',
                 'unit_normal_vector.npy']
        
        return files

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'data/')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_dsgps/')

    def process(self):

        data_list = []

        list_A_sparse_matrix = np.load(self.raw_dir + self.raw_file_names[0], allow_pickle = True)
        list_b_matrix = np.load(self.raw_dir + self.raw_file_names[1], allow_pickle = True)
        list_sol = np.load(self.raw_dir + self.raw_file_names[2], allow_pickle = True)
        list_prb_data = np.load(self.raw_dir + self.raw_file_names[3], allow_pickle = True)
        list_tags = np.load(self.raw_dir + self.raw_file_names[4], allow_pickle = True)
        list_coordinates = np.load(self.raw_dir + self.raw_file_names[5], allow_pickle = True)
        list_distance = np.load(self.raw_dir + self.raw_file_names[6], allow_pickle = True)
        list_unit_normal_vector = np.load(self.raw_dir + self.raw_file_names[7], allow_pickle = True)

        self.prb_data_mean = torch.tensor([-0.4319, 0.0289, -0.0189], dtype = self.precision)
        self.prb_data_std = torch.tensor([8.4245, 2.1942, 2.8585], dtype = self.precision)

        self.distance_mean = torch.tensor([0.0, 0.0, 0.0572], dtype = self.precision)
        self.distance_std = torch.tensor([0.0445, 0.0443, 0.0258], dtype = self.precision)

        self.unit_normal_vector_mean = torch.tensor([0.0007, -0.0004], dtype = self.precision)
        self.unit_normal_vector_std = torch.tensor([0.2773, 0.2959], dtype = self.precision)

        for i in range(len(list_A_sparse_matrix)):
            
            # Build edge_index and a_ij
            A_sparse_matrix = list_A_sparse_matrix[i]
            coefficients = np.asarray(sc.sparse.find(A_sparse_matrix))
            edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)
            a_ij = torch.tensor(coefficients[2,:].reshape(-1,1), dtype=self.precision)

            # Build b tensor
            b =  torch.tensor(list_b_matrix[i], dtype = self.precision)            

            # Extract exact solution
            sol = torch.tensor(list_sol[i], dtype = self.precision)

            # Extract prb_data
            prb_data = torch.tensor(list_prb_data[i], dtype = self.precision)
            prb_data = (prb_data - self.prb_data_mean) / self.prb_data_std

            # Extract prb_data
            edge_attr = torch.tensor(list_distance[i], dtype = self.precision)
            edge_attr = (edge_attr - self.distance_mean) / self.distance_std

            # Unit normal vector 
            unit_normal_vector = torch.tensor(list_unit_normal_vector[i], dtype = self.precision)
            unit_normal_vector = (unit_normal_vector - self.unit_normal_vector_mean) / self.unit_normal_vector_std
            
            # Extract tags to differentiate nodes 
            tags = torch.tensor(list_tags[i], dtype=self.precision)
        
            # Extract coordinates
            pos = torch.tensor(list_coordinates[i], dtype = self.precision)

            # Extract initial condition
            x = torch.zeros_like(sol)
            index_boundary = torch.where(tags[:,1]==1)[0]
            x[index_boundary,:] = b[index_boundary,:]

            data = Data(    x = x, edge_index = edge_index, 
                            edge_attr = edge_attr, a_ij = a_ij, y = b, 
                            sol = sol, prb_data = prb_data, tags = tags,  
                            pos = pos, unit_normal_vector = unit_normal_vector
                        )

            data_list.append(data)
            
        data_train_, data_test = train_test_split(data_list, test_size = 0.2, shuffle = True)
        data_train, data_val = train_test_split(data_train_, test_size = 0.25, shuffle = True)

        if self.mode == 'train' :
            data, slices = self.collate(data_train)
            torch.save((data, slices), self.processed_paths[0])
        elif self.mode == 'val' :
            data, slices = self.collate(data_val)
            torch.save((data, slices), self.processed_paths[1])
        elif self.mode == 'test':
            data, slices = self.collate(data_test)
            torch.save((data, slices), self.processed_paths[2])