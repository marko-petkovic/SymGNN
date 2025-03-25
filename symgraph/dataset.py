from torch_geometric.loader import DataLoader
from torch.utils.data import BatchSampler

import os

import numpy as np

from math import ceil

from symgraph.data_utils import create_graphs
from symgraph.sys_utils import PROJECT_ROOT

class ZeoSampler(BatchSampler):
    def __init__(self, zeolite_codes: list, batch_size, sample='over_sample', n_samples=None):
        '''
        Custom batch sampler for balanced sampling of zeolites
        sub_sample (bool): If True, in a single epoch, each zeolite will be sampled the number of times of the least occuring zeolite
        If False, each zeolite will be sampled the number of times of the most occuring zeolite by repeating the indices and sampling the remaining indices randomly
        '''

        assert sample in ['over_sample', 'under_sample'] or isinstance(n_samples, int), 'sample must be one of ["over_sample", "under_sample"] or n_samples must be an integer'


        self.batch_size = batch_size
        self.sample = sample

        self.zeolite_codes = zeolite_codes
        self.unique_zeo_codes = set(zeolite_codes)
        self.unique_zeo_codes_num = len(self.unique_zeo_codes)
        
        
        if n_samples is not None:
            self.samples_per_zeo = n_samples
            self.sample = 'user_defined'
        else:
            if self.sample == 'under_sample':
                self.samples_per_zeo = min([zeolite_codes.count(zeo_code) for zeo_code in self.unique_zeo_codes])
            else:
                self.samples_per_zeo = max([zeolite_codes.count(zeo_code) for zeo_code in self.unique_zeo_codes])

    def __iter__(self):
        batch_indices = []

        # Sample indices for each zeolite code
        for zeo_code in self.unique_zeo_codes:
            
            
            zeo_code_indices = [idx for idx, value in enumerate(self.zeolite_codes) if zeo_code == value]
            
            if self.sample == 'under_sample' or (self.sample == 'user_defined' and self.samples_per_zeo < len(zeo_code_indices)):

                zeo_code_indices = np.random.choice(zeo_code_indices, size=self.samples_per_zeo, replace=False).tolist()
            else: 
                repeats = np.floor(self.samples_per_zeo // len(zeo_code_indices)).astype(int)

                n_to_sample = self.samples_per_zeo % len(zeo_code_indices)

                zeo_code_indices = np.tile(zeo_code_indices, repeats).tolist()

                if n_to_sample > 0:

                    zeo_code_indices.extend(np.random.choice(zeo_code_indices, size=n_to_sample, replace=False).tolist())

            batch_indices.extend(zeo_code_indices)

        # Shuffle and pad indices to fit batches
        np.random.shuffle(batch_indices)
        indices_to_add = self.batch_size - (len(batch_indices) % self.batch_size)
        if indices_to_add < self.batch_size:
            batch_indices.extend([-1] * indices_to_add)

        # Yield one batch at a time
        for i in range(0, len(batch_indices), self.batch_size):
            batch = batch_indices[i:i + self.batch_size]
            # print([idx for idx in batch if idx != -1])
            yield [idx for idx in batch if idx != -1]  # Remove padded indices

    def __len__(self):
        return int(np.ceil(len(self.unique_zeo_codes) * self.samples_per_zeo / self.batch_size))
    



def create_dataloaders(train_codes, val_codes=None, test_codes=None, batch_size=32, sample='over_sample', val_split=0.1, test_split=0.1, n_samples=None, edge_type='zeo', radius=10.,seed=None, samples_per_code=None, subsample_zeolites=None):
    """
    Create dataloaders for training, validation and testing

    Args:
    train_codes (list): List of zeolite codes to include in the training set. If 'all', all zeolites in the data folder will be included
    val_codes (list): List of zeolite codes to include in the validation set
    test_codes (list): List of zeolite codes to include in the test set
    batch_size (int): Batch size for training and validation
    sample (str): Sampling strategy for the batch sampler. One of ['over_sample', 'under_sample']
    val_split (float): Fraction of the training set to use for validation. Overrides val_codes
    test_split (float): Fraction of the training set to use for testing. Overrides test_codes
    n_samples (int): Number of samples per zeolite to include in the batch sampler. Overrides sample
    edge_type (str): Type of edge to use in the graph. One of ['zeo', 'dist']
    radius (float): Radius for the distance edge type
    seed (int): Seed for the random number generator
    subsample_train (float): Fraction of the training set to use for training. 

    Returns:
    trainloader (DataLoader): DataLoader for the training set
    valloader (DataLoader): DataLoader for the validation set
    testloader (DataLoader): DataLoader for the test set
    mu (float): Mean of the HOA values
    std (float): Standard deviation of the HOA values
    """



    print(os.listdir())

    if train_codes == 'all':
        train_codes = os.listdir(str(PROJECT_ROOT / 'Data_numpy'))
        

    if val_codes is not None:
        val_split = 0
        print('Val codes provided, splitting train set')
        print('-------------------------------------')
        val_graphs, _, _ = create_graphs(val_codes, edge_type=edge_type, radius=radius)

        train_codes = [code for code in train_codes if code not in val_codes]



    if test_codes is not None:
        test_split = 0
        print('Test codes provided, splitting train set')
        print('-------------------------------------')
        test_graphs, _, _ = create_graphs(test_codes, edge_type=edge_type, radius=radius)

        train_codes = [code for code in train_codes if code not in test_codes]

    if subsample_zeolites is not None:
        assert subsample_zeolites > 0. and subsample_zeolites < 1., 'subsample_zeolites must be between 0 and 1'

        print(f'Subsample fraction: {subsample_zeolites}')
        np.random.seed(seed)
        old_train_codes = train_codes.copy()

        # ensure that codes with iso values are included

        train_codes_iso = ['MOR', 'MFI', 'TON', 'MEL']
        train_codes_no_iso = [code for code in train_codes if code not in train_codes_iso]

        n_no_iso = int(subsample_zeolites*len(train_codes_no_iso))
        n_iso = max(ceil(subsample_zeolites*len(train_codes_iso)*2), 1)

        np.random.shuffle(train_codes_no_iso)
        np.random.shuffle(train_codes_iso)

        train_codes = train_codes_no_iso[:n_no_iso] + train_codes_iso[:n_iso]

        dropped_codes = [code for code in old_train_codes if code not in train_codes]
        print(f'Dropped codes: {dropped_codes}')

    if val_split > 0 and test_split > 0:
        print('Splitting train, val and test sets based on the same codes')
        print('-------------------------------------')
        train_graphs, val_graphs, test_graphs = create_graphs(train_codes, val_split=val_split, test_split=test_split, edge_type=edge_type, radius=radius)
    elif val_split > 0:
        print('Splitting train and val sets based on the same codes')
        print('-------------------------------------')
        train_graphs, val_graphs, _ = create_graphs(train_codes, val_split=val_split, edge_type=edge_type, radius=radius)
    elif test_split > 0:
        print('Splitting train and test sets based on the same codes')
        print('-------------------------------------')
        train_graphs, _, test_graphs = create_graphs(train_codes, test_split=test_split, edge_type=edge_type, radius=radius)
    else:
        print('Not splitting train, val and test sets')
        print('-------------------------------------')
        train_graphs, _, _ = create_graphs(train_codes, edge_type=edge_type, radius=radius)

    if train_codes == 'all' and test_codes is not None:
        for g in train_graphs:
            assert g.zeo not in test_codes, 'Test codes are in train set'
    
    if train_codes == 'all' and val_codes is not None:
        for g in train_graphs:
            assert g.zeo not in val_codes, 'Val codes are in train set'
                
    if samples_per_code is not None:
        new_train_graphs = []
        print(f'Sampling {samples_per_code} samples per zeolite code')

        np.random.seed(seed)

        codes = np.unique([g.zeo for g in train_graphs])

        for code in codes:
            code_graphs = [g for g in train_graphs if g.zeo == code]
            n_samples = int(samples_per_code)
            # shuffle code_graphs
            np.random.shuffle(code_graphs)
            new_train_graphs.extend(code_graphs[:n_samples])
            # new_train_graphs.extend(np.random.choice(code_graphs, size=n_samples, replace=False).tolist())

        train_graphs = new_train_graphs

    train_hoas = [g.y for g in train_graphs]

    mu, std = np.mean(train_hoas), np.std(train_hoas)

    print('Normalizing HOA values')
    print(f'Mean: {mu:.3f}, Std: {std:.3f}')

    for g in train_graphs:
        g.y = (g.y - mu)/std
    
    for g in val_graphs:
        g.y = (g.y - mu)/std
    
    for g in test_graphs:
        g.y = (g.y - mu)/std


    trainloader = DataLoader(train_graphs, batch_sampler = ZeoSampler([g.zeo for g in train_graphs], batch_size, sample=sample, n_samples=n_samples))
    # valloader = DataLoader(val_graphs, batch_sampler = ZeoSampler([g.zeo for g in val_graphs], batch_size, sample=sample, n_samples=n_samples))
    valloader = DataLoader(test_graphs, batch_size = batch_size, shuffle=False)
    testloader = DataLoader(test_graphs, batch_size = batch_size, shuffle=False)

    return trainloader, valloader, testloader, mu, std