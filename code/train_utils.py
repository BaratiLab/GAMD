import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WaterDataNew(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,   # per seed
                 case_prefix='data_',
                 seed_num=10,
                 m_num=258,    # tip3p 258, tip4p 251
                 split=(0.9, 0.1),
                 mode='train',
                 data_type='tip3p',
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.case_prefix = case_prefix
        self.seed_num = seed_num

        self.data_type = data_type
        particle_type = []
        for i in range(m_num * 3):
            particle_type.append(1 if i % 3 == 0 else 0)
        self.particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)
        # transform into one hot encoding
        self.particle_type_one_hot = np.zeros((self.particle_type.size, 1), dtype=np.float32)
        self.particle_type_one_hot[self.particle_type.reshape(-1) == 1] = 1
        self.num_atom_type = self.particle_type.max() + 1
        print(f'Including atom type: {self.num_atom_type}')

        self.mode = mode
        assert mode in ['train', 'test']
        idxs = np.arange(seed_num*sample_num)
        np.random.seed(0)   # fix same random seed
        np.random.shuffle(idxs)
        ratio = split[0]
        if mode == 'train':
            self.idx = idxs[:int(len(idxs)*ratio)]
        else:
            self.idx = idxs[int(len(idxs)*ratio):]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        seed = idx // self.sample_num
        fname = f'data_{seed}_{sample_to_read}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            if self.data_type == 'tip4p':
                pos = pos[np.mod(np.arange(pos.shape[0]), 4) < 3]
            data['pos'] = pos
            data['feat'] = self.particle_type_one_hot
            forces = raw_data['forces'].astype(np.float32)
            if self.data_type == 'tip4p':
                forces = forces[np.mod(np.arange(forces.shape[0]), 4) < 3]
            data['forces'] = forces
        if get_path_name:
            return data, data_path
        return data


class LJDataNew(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,   # per seed
                 case_prefix='data_',
                 seed_num=10,
                 split=(0.9, 0.1),
                 mode='train',
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.case_prefix = case_prefix
        self.seed_num = seed_num

        self.mode = mode
        assert mode in ['train', 'test']
        idxs = np.arange(seed_num*sample_num)
        np.random.seed(0)   # fix same random seed
        np.random.shuffle(idxs)
        ratio = split[0]
        if mode == 'train':
            self.idx = idxs[:int(len(idxs)*ratio)]
        else:
            self.idx = idxs[int(len(idxs)*ratio):]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        seed = idx // self.sample_num
        fname = f'data_{seed}_{sample_to_read}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
        if get_path_name:
            return data, data_path
        return data


class WaterDataRealLarge(Dataset):
    def __init__(self,
                 dataset_path,
                 mode='train',
                 use_part=False
                 ):
        self.dataset_path = dataset_path
        self.use_part = use_part
        with np.load(self.dataset_path, allow_pickle=True) as npz_data:
            train_idx = npz_data['train_idx']
            test_idx = npz_data['test_idx']
            self.pos = npz_data['pos']
            self.forces = npz_data['force']
            self.box_size = npz_data['box']
            self.atom_type = npz_data['atom_type']
        if use_part:
            print(f'Using 1500 training samples')
        else:
            print(f'Using {len(train_idx)} training samples')
        print(f'Using {len(test_idx)} testing samples')

        if mode == 'train':
            if not use_part:
                self.idx = train_idx
            else:
                self.idx = train_idx[:1500]
        else:
            self.idx = test_idx

    def __len__(self):
        return len(self.idx)

    def generate_atom_emb(self, particle_type):
        particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)
        # transform into one hot encoding
        particle_type_one_hot = np.zeros((particle_type.size, 1), dtype=np.float32)
        particle_type_one_hot[particle_type.reshape(-1) == 1] = 1
        return particle_type_one_hot

    def __getitem__(self, idx):
        data = {}
        data['pos'] = self.pos[self.idx[idx]].copy().astype(np.float32)
        data['feat'] = self.generate_atom_emb(self.atom_type[self.idx[idx]])
        data['forces'] = self.forces[self.idx[idx]].copy().astype(np.float32)
        data['box_size'] = self.box_size[self.idx[idx]].copy().astype(np.float32)

        return data