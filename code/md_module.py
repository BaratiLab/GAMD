from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_
import jax
from jax_md import space, partition
from jax import numpy as jnp
from jax import device_put, ops
import numba as nb
from numba import njit, cuda
from matplotlib import pyplot as plt
from parameters import *
import warnings
warnings.filterwarnings("ignore")

zeros_initializer = partial(constant_, val=0.)


def periodic_boundary(pos: torch.tensor):

    for dim in range(3):
        left_mask = pos[:, dim] < left_bound
        while torch.sum(left_mask)>0:
            pos[left_mask, dim] = pos[left_mask, dim] + (right_bound - left_bound)
            left_mask = pos[:, dim] < left_bound

        right_mask = pos[:, dim] > right_bound
        while torch.sum(right_mask)>0:
            pos[right_mask, dim] = pos[right_mask, dim] - (right_bound - left_bound)
            right_mask = pos[:, dim] > right_bound
    return pos


def get_kinetic_energy(vn, batch_mode=False):
    if not batch_mode:
        return 0.5*(vn**2).sum()*MASS
    else:
        # first dimension is batch
        # [batch, particle_num, 3]
        return 0.5*(vn**2).sum(dim=2).sum(dim=1)*MASS


def get_temperature(kinetic):
    temp = (2 * kinetic) / (3*(N-1))         #  * (e_arg / kb)
    return temp


def step(model: nn.Module, pos, vel, eta):
    with torch.no_grad():
        k_eng = get_kinetic_energy(vel)
        temp = get_temperature(k_eng)
        # verlet scheme
        forces = model.forward(vel, [pos])

        vel += ((forces / MASS) - eta * vel) * (DT/2)
        pos += vel * DT
        pos = periodic_boundary(pos)

        eta += (DT / (taw ** 2)) * ((temp / T_set) - 1)
        forces_ = model.forward(vel, [pos])
        vel = (vel + (forces_ / MASS) * (DT/2)) / (1 + eta * DT / 2)
    return pos, vel, temp, eta


def batch_rdf_loss(forces, pos_lst, batch_vel, batch_eta):
    batch_vel = batch_vel.clone()
    batch_pos = torch.cat(pos_lst, dim=0)
    batch_vel += ((forces / MASS) - batch_eta * batch_vel) * (DT / 2)
    batch_pos += batch_vel * DT
    batch_pos = periodic_boundary(batch_pos)
    rdf_loss = get_rdf_loss(batch_pos, pos_lst)
    return rdf_loss


def get_rdf_loss(batch_pos, target_pos_lst):
    rdf = RDF()
    if torch.cuda.is_available():
        rdf.to(batch_pos.device)
    rdf_loss = 0.
    system_size = target_pos_lst[0].size(0)
    for b in range(len(target_pos_lst)):
        _, _, g = rdf.forward(batch_pos[b*system_size:(b+1)*system_size, :])
        _, _, g_obs = rdf.forward(target_pos_lst[b])
        rdf_loss += JS_rdf(g_obs, g)
    return rdf_loss / len(target_pos_lst)


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
    #print(dist)
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


def JS_rdf(g_obs, g):
    """Jensen–Shannon divergence of two rdf"""
    eps = 1e-8
    g_m = 0.5 * (g_obs + g)
    loss_js = (-(g_obs + eps) * (torch.log(g_m + eps) - torch.log(g_obs + eps))).mean()
    loss_js += (-(g + eps) * (torch.log(g_m + eps) - torch.log(g + eps))).mean()

    return loss_js


def gaussian_smearing_(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances
    del distances
    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    del diff
    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.
    sample struct dictionary:
        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}
    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """
    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.
        Returns:
            torch.Tensor: Tensor of convolved distances.
        """
        result = gaussian_smearing_(distances,
                                   self.offsets,
                                   self.width,
                                   centered=self.centered)

        return result


class RDF(nn.Module):
    def __init__(self, nbins=400, r_range=(0, BOX_SIZE/2), width=None):
        super(RDF, self).__init__()
        PI = np.pi

        start = r_range[0]
        end = r_range[1]
        self.bins = torch.linspace(start, end, nbins + 1)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        )
        self.cut_off = end - start
        self.box_size = self.cut_off * 2
        # compute volume differential
        self.vol_bins = 4 * PI / 3 * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)
        self.nbins = nbins
        self.cached_mask = None

    def forward(self, pos: torch.Tensor, divided=False, batch_size=int(1e6), box_size=None):
        if box_size is None:
            box_size = self.box_size

        if self.cached_mask is None:
            pair_dist, self_mask = pairwise_distance_norm(pos, box_size, mask_self=True)
            self.cached_mask = self_mask
        else:
            pair_dist, _ = pairwise_distance_norm(pos, box_size, mask_self=True, cached_mask=self.cached_mask)
        pair_dist = pair_dist.detach()
        if not divided:
            count = self.smear(pair_dist.view(-1).squeeze()[..., None]).sum(0)
        else:
            count = torch.zeros((self.nbins)).cuda()
            for b in range(pair_dist.shape[0] // batch_size + 1):
                end = b*batch_size + batch_size
                if b*batch_size + batch_size >= pair_dist.shape[0]:
                    end = -1
                count += self.smear(pair_dist[b*batch_size:end].view(-1).squeeze()[..., None]).sum(0)
            del pair_dist
            count = count
        norm = count.sum()  # normalization factor for histogram
        count = count / norm  # normalize

        V = (4 / 3) * np.pi * (self.cut_off ** 3)
        rdf = count.to(self.vol_bins.device) / (self.vol_bins / V)

        return count, self.bins, rdf

class RDF2Sys(nn.Module):
    def __init__(self, nbins=400, r_range=(0, BOX_SIZE/2), width=None):
        super(RDF2Sys, self).__init__()
        PI = np.pi

        start = r_range[0]
        end = r_range[1]
        self.bins = torch.linspace(start, end, nbins + 1)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        )
        self.cut_off = end - start
        self.box_size = self.cut_off * 2
        # compute volume differential
        self.vol_bins = 4 * PI / 3 * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)
        self.nbins = nbins

    def forward(self, pos1: torch.Tensor, pos2: torch.Tensor, box_size=None):
        if box_size is None:
            box_size = self.box_size
        pair_dist = torch.norm(pair_distance_two_system(pos1, pos2, box_size), dim=1)

        pair_dist = pair_dist.detach()
        pair_dist = pair_dist[pair_dist > 1.]
        count = self.smear(pair_dist.view(-1).squeeze()[..., None]).sum(0)
        norm = count.sum()  # normalization factor for histogram
        count = count / norm  # normalize

        V = (4 / 3) * np.pi * (self.cut_off ** 3)
        rdf = count.to(self.vol_bins.device) / (self.vol_bins / V)

        return count, self.bins, rdf


def enforce_water_pbc(pos, left_bound, right_bound):
    for dim in range(3):
        # only oxygen
        out_of_left_idx = np.argwhere(pos[0::3, dim] < left_bound)
        for i in range(3):
            pos[out_of_left_idx*3+i, dim] = pos[out_of_left_idx*3+i, dim] + (right_bound - left_bound)

        out_of_right_idx = np.argwhere(pos[0::3, dim] > right_bound)
        for i in range(3):
            pos[out_of_right_idx*3+i, dim] = pos[out_of_right_idx*3+i, dim] - (right_bound - left_bound)
    return pos

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

@njit
def periodic_displacement(pos1, pos2, box_size):
    disp = pos1 - pos2
    dist = 0.
    for dim in range(3):
        # if np.abs(disp[dim]) > box_size/2:
        #     dist += (np.abs(disp[dim]) - box_size/2)**2
        # else:
        dist += disp[dim] ** 2
    dist = np.sqrt(dist)
    if dist > box_size/2:
        dist = dist - box_size/2
    return dist


@njit
def stack_edge_idx(idx_arr):
    edge_idx = []
    count = 0
    for row in range(idx_arr.shape[0]):
        for col in range(idx_arr.shape[1]):
            j = idx_arr[row, col]
            if j >= idx_arr.shape[0]:
                break
            i = row
            edge_idx.append([i, j])
            count += 1

    return edge_idx, count

# when stacking edge idx
# also indicates, what edge type it is
# for now the type decision rule is hardcoded
@njit
def stack_water_edge_idx(idx_arr):

    edge_idx = np.zeros((3, idx_arr.shape[0]*64), dtype=np.int32)
    count = 0
    for row in range(idx_arr.shape[0]):
        for col in range(idx_arr.shape[1]):
            j = idx_arr[row, col]
            if j == idx_arr.shape[0]:
                break
            i = row
            # if periodic_displacement(pos[i], pos[j], box_size) > cutoff:
            #     continue

            if i % 3 == 0:  # oxygen
                bond_type = 0 if 0 < j - i <= 2 else 1
            elif i % 3 == 1:  # middle hydrogen
                bond_type = 0 if abs(j - i) <= 1 else 1
            elif i % 3 == 2:  # last hydrogen
                bond_type = 0 if 0 < i - j <= 2 else 1
            edge_idx[:3, count] = np.array([i, j, bond_type])

            count += 1
    # prune edge idx
    edge_idx = edge_idx[:, :count]
    return edge_idx


if __name__ == '__main__':
    for i in range(1, 10):
        pos = np.load(f'./md_dataset/XVF_NOT_EQ/X_1{i}.npy')[1000]

        pos_jax = device_put(jnp.array(pos[:]))
        test_searcher = NeighborSearcher(20.0, 2.7*1.5)
        nbrs = test_searcher.init_new_neighbor_lst(pos_jax)
        idx = device_put(jnp.sort(nbrs.idx), jax.devices("cpu")[0])
        edge_idx, _ = stack_water_edge_idx(np.array(idx))
        edge_idx = np.array(edge_idx).T
        bond_type_all = []
        p_num = pos.shape[0] // 3
        for i in range(p_num * 3):
            for j in range(p_num * 3):
                if i % 3 == 0:  # oxygen
                    bond_type_all.append(0 if 0 < j - i <= 2 else 1)
                elif i % 3 == 1:  # middle hydrogen
                    bond_type_all.append(0 if abs(j - i) <= 1 else 1)
                elif i % 3 == 2:  # last hydrogen
                    bond_type_all.append(0 if 0 < i - j <= 2 else 1)
        bond_type_all = np.array(bond_type_all)
        edge_idx_, masked_bond_type = get_neighbor(pos.astype(np.float32),  2.7*1.5, 20.0, return_dist=False, bond_type=bond_type_all)
        edge_idx_ = edge_idx_.cpu().numpy()
        print(edge_idx_.shape[1])
        displacement_fn, _ = space.periodic(20.0)
        metric = space.metric(displacement_fn)
        metric = jax.vmap(metric)
        print(edge_idx.shape[1])
        out_mask = metric(pos[edge_idx[0, :]], pos[edge_idx[1, :]])<= 2.7*1.5
        edge_idx = edge_idx[:, out_mask]
        print(edge_idx.shape[1])
        print(np.any(metric(pos[edge_idx[0, :]], pos[edge_idx[1, :]]) > 2.7 * 1.5))
        print(np.any(metric(pos[edge_idx_[0, :]], pos[edge_idx_[1, :]]) > 2.7 * 1.5))

        # for i in range(edge_idx_.shape[1]):
        #     find = False
        #     for j in range(edge_idx.shape[1]):
        #         if (edge_idx[0, j] == edge_idx_[0, i] and edge_idx[1, j] == edge_idx_[1, i]) or \
        #             (edge_idx[0, j] == edge_idx_[1, i] and edge_idx[1, j] == edge_idx_[0, i]):
        #             find = True
        #             break
        #
        #     if not find:
        #         print(edge_idx_[:2, i])
        #         print(pos[edge_idx_[0, i]])
        #         print(pos[edge_idx_[1, i]])

    # r1 = np.random.uniform(0, 10.0, (10000, 3))
    # r2 = np.random.uniform(0, 20.0, (10000, 3))
    # torch_res = torch.norm(torch.from_numpy(r1) - torch.from_numpy(r2), dim=1)
    # jax_res = me
    # print(metric(pos[0], pos[1]))

    # pos2 = np.random.normal(0.0, 20.0, (774, 3)).astype(np.float32)
    # pos_jax2 = device_put(jnp.array(pos2[:]))
    # nbrs = test_searcher.update_neighbor_lst(pos_jax2, nbrs)
    # idx = device_put(jnp.sort(nbrs.idx), jax.devices("cpu")[0])
    # edge_idx, _ = stack_water_edge_idx(np.array(idx))
    # edge_idx = np.array(edge_idx).T
    # print(edge_idx.shape[1])
    # edge_idx_, masked_bond_type = get_neighbor(pos2, 2.7 * 1.5, 20.0, return_dist=False, bond_type=bond_type_all)
    # print(edge_idx_.shape[1])
