import argparse
import os, sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import jax
import jax.numpy as jnp
import cupy

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import SimpleMDNetNew
from train_utils import LJDataNew
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# for water box
CUTOFF_RADIUS = 7.5
BOX_SIZE = 27.27

NUM_OF_ATOMS = 258

# NUM_OF_ATOMS = 251 * 3  # tip4p
# CUTOFF_RADIUS = 3.4

LAMBDA1 = 100.
LAMBDA2 = 1e-3


def get_rotation_matrix():
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    if np.random.uniform() < 0.3:
        angles = np.random.randint(-2, 2, size=(3,)) * np.pi
    else:
        angles = [0., 0., 0.]
    Rx = np.array([[1., 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=np.float32)
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=np.float32)
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]], dtype=np.float32)
    rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))

    return rotation_matrix


def center_positions(pos):
    offset = np.mean(pos, axis=0)
    return pos - offset, offset

def build_model(args, ckpt=None):

    param_dict = {
                  'encoding_size': args.encoding_size,
                  'out_feats': 3,
                  'hidden_dim': args.hidden_dim,
                  'edge_embedding_dim': args.edge_embedding_dim,
                  'conv_layer': 4,
                  'drop_edge': args.drop_edge,
                  'use_layer_norm': args.use_layer_norm,
                  'box_size': BOX_SIZE,
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)
    model = SimpleMDNetNew(**param_dict)

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class ParticleNetLightning(pl.LightningModule):
    def __init__(self, args, num_device=1, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1000,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super(ParticleNetLightning, self).__init__()
        self.pnet_model = build_model(args, model_weights_ckpt)
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_device = num_device
        self.log_freq = log_freq
        self.train_data_scaler = StandardScaler()
        self.training_mean = np.array([0.])
        self.training_var = np.array([1.])

        if scaler_ckpt is not None:
            self.load_training_stats(scaler_ckpt)

        self.cutoff = CUTOFF_RADIUS
        self.nbr_searcher = NeighborSearcher(BOX_SIZE, self.cutoff)
        self.nbrlst_to_edge_mask = jax.jit(graph_network_nbr_fn(self.nbr_searcher.displacement_fn,
                                                                    self.cutoff,
                                                                    NUM_OF_ATOMS))
        self.nbr_cache = {}
        self.rotate_aug = args.rotate_aug
        self.data_dir = args.data_dir
        self.loss_fn = args.loss
        assert self.loss_fn in ['mae', 'mse']

    def load_training_stats(self, scaler_ckpt):
        if scaler_ckpt is not None:
            scaler_info = np.load(scaler_ckpt)
            self.training_mean = scaler_info['mean']
            self.training_var = scaler_info['var']

    def forward(self, pos, feat, edge_idx_tsr):
        return self.denormalize(self.pnet_model(pos, feat, edge_idx_tsr.long()), self.training_var, self.training_mean)

    def denormalize(self, normalized_force, var, mean):
        return normalized_force * \
                np.sqrt(var) +\
                mean

    def predict_forces(self, pos: np.ndarray, verbose=False):
        nbr_start = time.time()
        edge_idx_tsr = self.search_for_neighbor(pos,
                                                self.nbr_searcher,
                                                self.nbrlst_to_edge_mask,
                                                'all')
        nbr_end = time.time()
        # enforce periodic boundary
        pos = np.mod(pos, np.array(BOX_SIZE))
        pos = torch.from_numpy(pos).float().cuda()
        force_start = time.time()
        pred = self.pnet_model([pos],
                               [edge_idx_tsr],
                               )
        force_end = time.time()
        if verbose:
            print('=============================================')
            print(f'Nbr search used time: {nbr_end - nbr_start}')
            print(f'Force eval used time: {force_end - force_start}')

        pred = pred.detach().cpu().numpy()

        pred = self.denormalize(pred, self.training_var, self.training_mean)

        return pred

    def scale_force(self, force, scaler):
        b_pnum, dims = force.shape
        force_flat = force.reshape((-1, 1))
        scaler.partial_fit(force_flat)
        force = torch.from_numpy(scaler.transform(force_flat)).float().view(b_pnum, dims)
        return force

    def get_edge_idx(self, nbrs, pos_jax, mask):
        dummy_center_idx = nbrs.idx.copy()
        dummy_center_idx = jax.ops.index_update(dummy_center_idx, None,
                                                jnp.arange(pos_jax.shape[0]).reshape(-1, 1))
        center_idx = dummy_center_idx.reshape(-1)
        center_idx_ = cupy.asarray(center_idx)
        center_idx_tsr = torch.as_tensor(center_idx_, device='cuda')

        neigh_idx = nbrs.idx.reshape(-1)

        # cast jax device array to cupy array so that it can be transferred to torch
        neigh_idx = cupy.asarray(neigh_idx)
        mask = cupy.asarray(mask)
        mask = torch.as_tensor(mask, device='cuda')
        flat_mask = mask.view(-1)
        neigh_idx_tsr = torch.as_tensor(neigh_idx, device='cuda')

        edge_idx_tsr = torch.cat((center_idx_tsr[flat_mask].view(1, -1), neigh_idx_tsr[flat_mask].view(1, -1)),
                                 dim=0)
        return edge_idx_tsr

    def search_for_neighbor(self, pos, nbr_searcher, masking_fn, type_name):
        pos_jax = jax.device_put(pos, jax.devices("gpu")[0])

        if not nbr_searcher.has_been_init:
            nbrs = nbr_searcher.init_new_neighbor_lst(pos_jax)
            self.nbr_cache[type_name] = nbrs
        else:
            nbrs = nbr_searcher.update_neighbor_lst(pos_jax, self.nbr_cache[type_name])
            self.nbr_cache[type_name] = nbrs

        edge_mask_all = masking_fn(pos_jax, nbrs.idx)
        edge_idx_tsr = self.get_edge_idx(nbrs, pos_jax, edge_mask_all)
        return edge_idx_tsr.long()

    def training_step(self, batch, batch_nb):
        pos_lst = batch['pos']
        gt_lst = batch['forces']
        edge_idx_lst = []
        for b in range(len(gt_lst)):
            pos, gt = pos_lst[b], gt_lst[b]

            if self.rotate_aug:
                pos = np.mod(pos, BOX_SIZE)
                pos, off = center_positions(pos)
                R = get_rotation_matrix()
                pos = np.matmul(pos, R)
                pos += off
                gt = np.matmul(gt, R)

            pos = np.mod(pos, BOX_SIZE)

            gt = self.scale_force(gt, self.train_data_scaler).cuda()
            pos_lst[b] = torch.from_numpy(pos).float().cuda()
            gt_lst[b] = gt

            edge_idx_tsr = self.search_for_neighbor(pos,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')
            edge_idx_lst += [edge_idx_tsr]
        gt = torch.cat(gt_lst, dim=0)
        pos_lst = [pos + torch.randn_like(pos) * 0.005 for pos in pos_lst]

        pred = self.pnet_model(pos_lst,
                               edge_idx_lst,
                               )

        if self.loss_fn == 'mae':
            loss = nn.L1Loss()(pred, gt)
        else:
            loss = nn.MSELoss()(pred, gt)

        conservative_loss = (torch.mean(pred)).abs()
        loss = loss + LAMBDA2 * conservative_loss

        self.training_mean = self.train_data_scaler.mean_
        self.training_var = self.train_data_scaler.var_

        self.log('total loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log(f'{self.loss_fn} loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('var', np.sqrt(self.training_var), on_step=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = StepLR(optim, step_size=5, gamma=0.001**(5/self.epoch_num))
        return [optim], [sched]

    def train_dataloader(self):
        dataset = LJDataNew(dataset_path=os.path.join(self.data_dir, 'lj_data'),
                               sample_num=1000,
                               case_prefix='data_',
                               seed_num=10,
                               m_num=NUM_OF_ATOMS,
                               mode='train')

        return DataLoader(dataset, num_workers=2, batch_size=self.batch_size, shuffle=True,
                          collate_fn=
                          lambda batches: {
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                          })

    def val_dataloader(self):
        dataset = LJDataNew(dataset_path=os.path.join(self.data_dir, 'lj_data'),
                               sample_num=1000,
                               case_prefix='data_',
                               seed_num=10,
                               m_num=NUM_OF_ATOMS,
                               mode='test')

        return DataLoader(dataset, num_workers=2, batch_size=16, shuffle=False,
                          collate_fn=
                          lambda batches: {
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                          })

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():

            pos_lst = batch['pos']
            gt_lst = batch['forces']
            edge_idx_lst = []
            for b in range(len(gt_lst)):
                pos, gt = pos_lst[b], gt_lst[b]
                pos = np.mod(pos, BOX_SIZE)

                gt = self.scale_force(gt, self.train_data_scaler).cuda()
                pos_lst[b] = torch.from_numpy(pos).float().cuda()
                gt_lst[b] = gt

                edge_idx_tsr = self.search_for_neighbor(pos,
                                                        self.nbr_searcher,
                                                        self.nbrlst_to_edge_mask,
                                                        'all')
                edge_idx_lst += [edge_idx_tsr]
            gt = torch.cat(gt_lst, dim=0)

            pred = self.pnet_model(pos_lst,
                                   edge_idx_lst,
                                   )
            ratio = torch.sqrt((pred.reshape(-1) - gt.reshape(-1)) ** 2) / (torch.abs(pred.reshape(-1)) + 1e-8)
            outlier_ratio = ratio[ratio > 10.].shape[0] / ratio.shape[0]
            mse = nn.MSELoss()(pred, gt)
            mae = nn.L1Loss()(pred, gt)

            self.log('val outlier', outlier_ratio, prog_bar=True, logger=True)
            self.log('val mse', mse, prog_bar=True, logger=True)
            self.log('val mae', mae, prog_bar=True, logger=True)


class ModelCheckpointAtEpochEnd(pl.Callback):
    """
       Save a checkpoint at epoch end
    """
    def __init__(
            self,
            filepath,
            save_step_frequency,
            prefix="checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
        """
        self.filepath = filepath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: ParticleNetLightning):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0 or epoch == pl_module.epoch_num -1:
            filename = os.path.join(self.filepath, f"{self.prefix}_{epoch}.ckpt")
            scaler_filename = os.path.join(self.filepath, f"scaler_{epoch}.npz")

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            np.savez(scaler_filename,
                     mean=pl_module.training_mean,
                     var=pl_module.training_var,
                     )
            # joblib.dump(pl_module.train_data_scaler, scaler_filename)


def train_model(args):
    lr = args.lr
    num_gpu = args.num_gpu
    check_point_dir = args.cp_dir
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    weight_ckpt = args.state_ckpt_dir
    batch_size = args.batch_size

    model = ParticleNetLightning(epoch_num=max_epoch,
                                 num_device=num_gpu if num_gpu != -1 else 1,
                                 learning_rate=lr,
                                 model_weights_ckpt=weight_ckpt,
                                 batch_size=batch_size,
                                 args=args)
    cwd = os.getcwd()
    model_check_point_dir = os.path.join(cwd, check_point_dir)
    os.makedirs(model_check_point_dir, exist_ok=True)
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    trainer = Trainer(gpus=num_gpu,
                      callbacks=[epoch_end_callback, checkpoint_callback],
                      min_epochs=min_epoch,
                      max_epochs=max_epoch,
                      amp_backend='apex',
                      amp_level='O1',
                      benchmark=True,
                      distributed_backend='ddp',
                      )
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default=11, type=int)
    parser.add_argument('--max_epoch', default=11, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--encoding_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--edge_embedding_dim', default=256, type=int)
    parser.add_argument('--drop_edge', action='store_true')
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./md_dataset')
    parser.add_argument('--loss', default='mae')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()

