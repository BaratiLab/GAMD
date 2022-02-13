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
import cupy

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import WaterMDDynamicBoxNet
from train_utils import WaterDataReal, WaterDataRealLarge
from graph_utils import NeighborSearcher, graph_network_nbr_fn
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# for water box
# CUTOFF_RADIUS = 9.5
# left_bound = 0.0
# right_bound = 12.4172
# BOX_SIZE = right_bound - left_bound
# NUM_OF_ATOMS = 258 * 3

# CUTOFF_RADIUS = 3.4

LAMBDA1 = 100.
LAMBDA2 = 0.5e-2


def create_water_bond(total_atom_num):
    bond = []
    for i in range(0, total_atom_num, 3):
        bond += [[i, i+1], [i, i+2]]
    return np.array(bond)


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
    # bond_info = create_water_bond(NUM_OF_ATOMS)
    # print(bond_info)
    param_dict = {'in_feats': 1,
                'encoding_size': args.encoding_size,
                'out_feats': 3,
                #'bond': bond_info,
                'hidden_dim': args.hidden_dim,
                'edge_embedding_dim': args.edge_embedding_dim,
                'conv_layer': args.conv_layer,
                'drop_edge': args.drop_edge,
                'use_layer_norm': args.use_layer_norm,
                'update_edge': args.update_edge,
                }
    # small model
    # param_dict = {'in_feats': 1,
    #               'encoding_size': 512,
    #               'out_feats': 3,
    #               'bond': bond_info,
    #               'hidden_dim': 256,
    #               'conv_layer': 5,
    #               }

    print("Using following set of hyper-parameters")
    print(args)

    # print(param_dict)
    model = WaterMDDynamicBoxNet(**param_dict, expand_edge=args.expand_edge)

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

        self.cutoff = args.cutoff
        self.rotate_aug = args.rotate_aug
        self.data_dir = args.data_dir
        self.loss_fn = args.loss
        self.use_part = args.use_part
        assert self.loss_fn in ['mae', 'mse']


    def load_training_stats(self, scaler_ckpt):
        if scaler_ckpt is not None:
            scaler_info = np.load(scaler_ckpt)
            self.training_mean = scaler_info['mean']
            self.training_var = scaler_info['var']

    def forward(self, pos, feat, edge_idx_tsr):
        return self.denormalize(self.pnet_model(pos, feat, edge_idx_tsr.long()), self.training_var, self.training_mean)

    def build_graph(self, edge_idx):
        return self.pnet_model.build_partial_graph(edge_idx)

    def denormalize(self, normalized_force, var, mean):
        return normalized_force * \
                np.sqrt(var) +\
                mean

    def predict_forces(self, feat: torch.Tensor, pos: np.ndarray, box_size):
        # enforce periodic boundary
        pos = np.mod(pos, box_size)
        pos = torch.from_numpy(pos).float().to(feat.device)
        pred = self.pnet_model([pos],
                               feat,
                               [box_size],
                               self.cutoff
                               )

        pred = pred.detach().cpu().numpy()

        pred = self.denormalize(pred, self.training_var, self.training_mean)

        return pred

    def scale_force(self, force, scaler):
        b_pnum, dims = force.shape
        force_flat = force.reshape((-1, 1))
        scaler.partial_fit(force_flat)
        force = torch.from_numpy(scaler.transform(force_flat)).float().view(b_pnum, dims)
        return force

    def training_step(self, batch, batch_nb):
        feat, pos_lst, box_size_lst = batch['feat'], batch['pos'], batch['box_size']
        gt_lst = batch['forces']

        for b in range(len(gt_lst)):
            pos, box_size, gt = pos_lst[b], box_size_lst[b], gt_lst[b]
            pos, off = center_positions(pos)
            R = get_rotation_matrix()
            pos = np.matmul(pos, R)
            pos += off
            box_size = np.matmul(box_size, R)
            box_size_lst[b] = box_size
            pos = np.mod(pos, box_size)
            gt = np.matmul(gt, R)
            gt = self.scale_force(gt, self.train_data_scaler).to(feat.device)
            pos_lst[b] = torch.from_numpy(pos).float().to(feat.device)
            gt_lst[b] = gt
        gt = torch.cat(gt_lst, dim=0)

        # enforce periodic boundary
        pos_lst = [pos + torch.randn_like(pos) * 0.00025 for pos in pos_lst]
        pred = self.pnet_model(pos_lst,
                               feat,
                               box_size_lst,
                               self.cutoff
                               )
        epoch = self.current_epoch
        # if epoch > 5:
        #     mae = nn.L1Loss()(pred, gt)
        # else:
        if self.loss_fn == 'mae':
            loss = nn.L1Loss()(pred, gt)
        else:
            loss = nn.MSELoss()(pred, gt)

        conservative_loss = (torch.mean(pred)).abs()
        loss = loss + LAMBDA2*conservative_loss

        self.training_mean = self.train_data_scaler.mean_
        self.training_var = self.train_data_scaler.var_

        self.log('total loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log(f'{self.loss_fn} loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('var', np.sqrt(self.training_var), on_step=True, prog_bar=True, logger=True)

        # self.log('regularization', conservative_loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        sched = StepLR(optim, step_size=100, gamma=0.001**(100/self.epoch_num))
        return [optim], [sched]

    def train_dataloader(self):
        dataset = WaterDataRealLarge(dataset_path=os.path.join(self.data_dir, 'RPBE-data-processed.npz'), use_part=self.use_part)
        return DataLoader(dataset, num_workers=2, batch_size=self.batch_size, shuffle=True,
                          collate_fn=
                          lambda batches: {
                              'feat': torch.cat([torch.from_numpy(batch['feat']).float() for batch in batches], dim=0),
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                              'box_size': [batch['box_size'] for batch in batches],
                          })

    def val_dataloader(self):
        dataset = WaterDataRealLarge(dataset_path=os.path.join(self.data_dir, 'RPBE-data-processed.npz'),
                                     mode='test')
        return DataLoader(dataset, num_workers=2, batch_size=self.batch_size*2, shuffle=False,
                          collate_fn=
                          lambda batches: {
                              'feat': torch.cat([torch.from_numpy(batch['feat']).float() for batch in batches], dim=0),
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                              'box_size': [batch['box_size'] for batch in batches],
                          })

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            feat, pos_lst, box_size_lst = batch['feat'], batch['pos'], batch['box_size']
            gt_lst = batch['forces']

            for b in range(len(gt_lst)):
                pos, box_size, gt = pos_lst[b], box_size_lst[b], gt_lst[b]

                pos = np.mod(pos, box_size)
                gt = self.scale_force(gt, self.train_data_scaler).to(feat.device)
                pos_lst[b] = torch.from_numpy(pos).float().to(feat.device)
                gt_lst[b] = gt
            gt = torch.cat(gt_lst, dim=0)
            # enforce periodic boundary
            pred = self.pnet_model(pos_lst,
                                   feat,
                                   box_size_lst,
                                   self.cutoff
                                   )
            mse = nn.MSELoss()(pred, gt)
            mae = nn.L1Loss()(pred, gt)

            self.training_mean = self.train_data_scaler.mean_
            self.training_var = self.train_data_scaler.var_

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
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=50)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    trainer = Trainer(gpus=num_gpu,
                      callbacks=[epoch_end_callback, checkpoint_callback],
                      min_epochs=min_epoch,
                      max_epochs=max_epoch,
                      amp_backend='apex',
                      amp_level='O2',
                      benchmark=True,
                      distributed_backend='ddp',
                      )
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default=11, type=int)
    parser.add_argument('--max_epoch', default=11, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--encoding_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--edge_embedding_dim', default=256, type=int)
    parser.add_argument('--cutoff', default=9.5, type=float)
    parser.add_argument('--conv_layer', default=5, type=int)
    parser.add_argument('--drop_edge', action='store_true')
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--update_edge', action='store_true')
    parser.add_argument('--disable_expand_edge', dest='expand_edge', default=True, action='store_false')

    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./md_dataset')
    parser.add_argument('--use_part', action='store_true')    # use only part of the training data?
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()

