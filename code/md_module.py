from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
import jax
from jax_md import space, partition
from jax import numpy as jnp
import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def pairwise_distance_norm(x, box_size, mask_self=False, cached_mask=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = \sqrt{||x[i,:]-y[j,:]||^2}
    '''
    dist_all = 0.
    for dim in range(x.shape[1]):
        x_norm = (x[:, 0] ** 2).view(-1, 1)
        y_t = x[:, 0].view(1, -1)
        y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x[:, 0].view(-1, 1), y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        # Ensure diagonal is zero if x=y
        del x_norm, y_norm, y_t
        dist -= torch.diag(dist.diag())
        dist = torch.clamp(dist, 0.0, np.inf)
        dist[dist != dist] = 0.
        dist = dist.view(-1)

        # dist = torch.remainder(dist + 0.5 * box_size, float(box_size)) - 0.5 * float(box_size)
        dist_mat_mask = dist > (box_size**2 / 4)
        dist[dist_mat_mask] = dist[dist_mat_mask] + box_size**2 -\
                              2.0 * box_size * torch.sqrt(dist[dist_mat_mask]) * torch.sign(dist[dist_mat_mask])

        del dist_mat_mask
        if dim != 2:
            x = x[:, 1:]
        dist_all += dist
        del dist
        torch.cuda.empty_cache()
    dist = torch.sqrt_(dist_all)
    if mask_self:
        # if cached_mask is None:
        #     self_mask = np.array([i*x.shape[0] + j for i in range(x.shape[0]) for j in range(x.shape[0]) if i == j])
        #     mask_array = np.ones(x.shape[0]*x.shape[0], dtype=bool)
        #     mask_array[self_mask] = False
        # else:
        #     mask_array = cached_mask
        dist = dist[dist > 0.]
    return dist, None


def pair_distance(pos: torch.Tensor, box_size, mask_self=False, return_norm=False, cached_mask=None):
    # [[0, 1, 2, ,3, 4 ...], [0, 1, ...],...]      [[0, 0, 0, 0, 0, ...], [1, 1, 1, ...],...]
    dist_mat = pos[None, :, :] - pos[:, None, :]
    dist_mat = torch.remainder(dist_mat + 0.5 * box_size, box_size) - 0.5 * box_size
    dist_mat = dist_mat.view(-1, pos.size(1))
    if mask_self:
        if cached_mask is None:
            self_mask = np.array([i*pos.shape[0] + j for i in range(pos.shape[0]) for j in range(pos.shape[0]) if i == j])
            mask_array = np.ones(pos.shape[0]*pos.shape[0], dtype=bool)
            mask_array[self_mask] = 0
        else:
            mask_array = cached_mask
        dist_mat = dist_mat[mask_array]
    if return_norm:
        return dist_mat.norm(dim=1), mask_array
    return dist_mat


def pair_distance_two_system(pos1: torch.Tensor, pos2: torch.Tensor, box_size):
    # pos1 and pos2 should in same shape
    # [[0, 1, 2, ,3, 4 ...], [0, 1, ...],...]      [[0, 0, 0, 0, 0, ...], [1, 1, 1, ...],...]
    dist_mat = pos1[None, :, :] - pos2[:, None, :]
    # dist_mat_mask_right = dist_mat > box_size / 2
    # dist_mat_mask_left = dist_mat < -box_size / 2
    #
    # dist_mat[dist_mat_mask_right] = dist_mat[dist_mat_mask_right] - box_size
    # dist_mat[dist_mat_mask_left] = dist_mat[dist_mat_mask_left] + box_size
    dist_mat = torch.remainder(dist_mat+0.5*box_size, box_size) - 0.5*box_size
    return dist_mat.view(-1, pos1.size(1))

def get_neighbor(pos: torch.Tensor, r_cutoff, box_size, return_dist=True,
                 predefined_mask=None, bond_type=None):

    if isinstance(pos, np.ndarray):
        if torch.cuda.is_available():
            pos = torch.from_numpy(pos).cuda()
            if bond_type is not None:
                bond_type = torch.from_numpy(bond_type).cuda()

    with torch.no_grad():
        distance = pair_distance(pos, box_size)
        distance_norm = torch.norm(distance, dim=1)  # [pos.size(0) * pos.size(0), 1]
        edge_idx_1 = torch.cat([torch.arange(pos.size(0)) for _ in range(pos.size(0))], dim=0).to(pos.device)
        edge_idx_2 = torch.cat([torch.LongTensor(pos.size(0)).fill_(i) for i in range(pos.size(0))], dim=0).to(pos.device)

        if predefined_mask is not None:
            mask = (distance_norm.view(-1) <= r_cutoff) & predefined_mask & ~(edge_idx_1 == edge_idx_2)
        else:
            mask = (distance_norm.view(-1) <= r_cutoff) & ~(edge_idx_1 == edge_idx_2)

        masked_bond_type = None
        if bond_type is not None:
            masked_bond_type = bond_type[mask]
        edge_idx_1 = edge_idx_1[mask].view(1, -1)
        edge_idx_2 = edge_idx_2[mask].view(1, -1)

        edge_idx = torch.cat((edge_idx_1, edge_idx_2), dim=0)
        distance = distance[mask]
        distance_norm = distance_norm[mask]

    if return_dist:
        return edge_idx, distance, distance_norm, masked_bond_type
    else:
        return edge_idx, masked_bond_type


@jax.jit
def edge_type_water(i, j):

    cond1 = jnp.logical_and(i % 3 == 0, 0 < j - i)
    cond1 = jnp.logical_and(cond1, j - i <= 2)

    cond2 = jnp.logical_and(i % 3 == 1, abs(j - i) <= 1)

    cond3 = jnp.logical_and(i % 3 == 2, 0 < i - j)
    cond3 = jnp.logical_and(cond3, i - j <= 2)

    in_same_molecule = jnp.logical_or(jnp.logical_or(cond1, cond2), cond3)
    bond_type = jnp.where(in_same_molecule, 0, 1)
    return bond_type


class NeighborSearcher(object):
    def __init__(self, box_size, cutoff):
        # define a displacement function under periodic condition
        self.displacement_fn, _ = space.periodic(box_size)
        self.disp = jax.vmap(self.displacement_fn)
        self.dist = jax.vmap(space.metric(self.displacement_fn))
        self.cutoff = cutoff
        self.neighbor_list_fn = partition.neighbor_list(self.displacement_fn,
                                                       box_size,
                                                       cutoff,
                                                       dr_threshold= cutoff / 5.)
        self.neighbor_list_fn_jit = jax.jit(self.neighbor_list_fn)
        self.box_size = box_size
        self.neighbor_dist_jit = self.displacement_fn

    def init_new_neighbor_lst(self, pos):
        # Create a new neighbor list.
        pos = jnp.mod(pos, self.box_size)
        nbr = self.neighbor_list_fn(pos)

        return nbr

    def update_neighbor_lst(self, pos, nbr):
        update_idx = True
        pos = jnp.mod(pos, self.box_size)
        # update_idx = np.any(self.dist(pos, nbr.reference_position) > (self.cutoff / 10.))

        nbr = self.neighbor_list_fn_jit(pos, nbr)
        nbr_lst_updated = False
        if nbr.did_buffer_overflow:
            nbr = self.neighbor_list_fn(pos)
            nbr_lst_updated = True

        return nbr, update_idx, nbr_lst_updated


def graph_network_nbr_fn(displacement_fn,
                         cutoff,
                         edge_type_fn,
                         N):

    def nbrlst_to_edge(pos: jnp.ndarray, neigh_idx: jnp.ndarray):
        # notice here, pos must be jax numpy array, otherwise fancy indexing will fail
        d = jax.partial(displacement_fn)
        d = space.map_neighbor(d)
        pos_neigh = pos[neigh_idx]
        dR = d(pos, pos_neigh)

        dr_2 = space.square_distance(dR)
        mask = jnp.logical_and(neigh_idx != N, dr_2 < cutoff ** 2)

        edge_dist = dR
        edge_norm = jnp.sqrt(dr_2)
        return edge_dist, edge_norm, mask

    def filter_edge_idx(center_idx: jnp.ndarray, neigh_idx: jnp.ndarray):
        edge_type_fv = jax.vmap(edge_type_fn, in_axes=(0, 0), out_axes=0)
        I = center_idx
        J = neigh_idx
        edge_type = edge_type_fv(I, J)
        return edge_type

    return nbrlst_to_edge, filter_edge_idx


def get_water_box_neighbor_fn(box_size, cutoff):
    displacement_fn, _ = space.periodic(box_size)
    neighbor_list_fn = partition.neighbor_list(displacement_fn,
                                                box_size,
                                                cutoff,
                                                dr_threshold=cutoff/5.,
                                                capacity_multiplier=1.0
                                               )
    return neighbor_list_fn

