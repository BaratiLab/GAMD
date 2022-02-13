import numpy as np
import torch
import torch.nn as nn
import dgl.nn
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import time
from md_module import get_neighbor
from sklearn.preprocessing import StandardScaler

from typing import List, Set, Dict, Tuple, Optional


def cubic_kernel(r, re):
    eps = 1e-3
    r = torch.threshold(r, eps, re)
    return nn.ReLU()((1. - (r/re)**2)**3)


class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='relu',
                 init_param=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh, elu, as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if hidden_layer == 1:
                    if activation_first:
                        mlp_layer += [act_fn, nn.Linear(in_feats, out_feats)]
                    else:
                        print('Using MLP with no hidden layer and activations! Fall back to nn.Linear()')
                        mlp_layer += [nn.Linear(in_feats, out_feats)]
                elif not activation_first:
                    mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:   # l == hidden_layer-1
                mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)
        if init_param:
            self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat):
        return self.mlp_layer(feat)


class SmoothConvLayerNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 hidden_dim=128,
                 activation='relu',
                 drop_edge=True,
                 update_edge_emb=False):

        super(SmoothConvLayerNew, self).__init__()
        self.drop_edge = drop_edge
        self.update_edge_emb = update_edge_emb
        if self.update_edge_emb:
            self.edge_layer_norm = nn.LayerNorm(in_edge_feats)

        # self.theta_src = nn.Linear(in_node_feats, hidden_dim)
        self.edge_affine = MLP(in_edge_feats, hidden_dim, activation=activation, hidden_layer=2)
        self.src_affine = nn.Linear(in_node_feats, hidden_dim)
        self.dst_affine = nn.Linear(in_node_feats, hidden_dim)
        self.theta_edge = MLP(hidden_dim, in_node_feats,
                              hidden_dim=hidden_dim, activation=activation, activation_first=True,
                              hidden_layer=2)
        # self.theta = MLP(hidden_dim, hidden_dim, activation_first=True, hidden_layer=2)

        self.phi_dst = nn.Linear(in_node_feats, hidden_dim)
        self.phi_edge = nn.Linear(in_node_feats, hidden_dim)
        self.phi = MLP(hidden_dim, out_node_feats,
                       activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation=activation)

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        h = node_feat.clone()
        with g.local_scope():
            if self.drop_edge and self.training:
                src_idx, dst_idx = g.edges()
                e_feat = g.edata['e'].clone()
                dropout_ratio = 0.2
                idx = np.arange(dst_idx.shape[0])
                np.random.shuffle(idx)
                keep_idx = idx[:-int(idx.shape[0] * dropout_ratio)]
                src_idx = src_idx[keep_idx]
                dst_idx = dst_idx[keep_idx]
                e_feat = e_feat[keep_idx]
                g = dgl.graph((src_idx, dst_idx))
                g.edata['e'] = e_feat
            # for multi batch training
            if g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h

            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            edge_idx = g.edges()
            src_idx = edge_idx[0]
            dst_idx = edge_idx[1]
            edge_code = self.edge_affine(g.edata['e'])
            src_code = self.src_affine(h_src[src_idx])
            dst_code = self.dst_affine(h_dst[dst_idx])
            g.edata['e_emb'] = self.theta_edge(edge_code+src_code+dst_code)

            if self.update_edge_emb:
                normalized_e_emb = self.edge_layer_norm(g.edata['e_emb'])
            g.update_all(fn.src_mul_edge('h', 'e_emb', 'm'), fn.sum('m', 'h'))
            edge_emb = g.ndata['h']

        if self.update_edge_emb:
            g.edata['e'] = normalized_e_emb
        node_feat = self.phi(self.phi_dst(h) + self.phi_edge(edge_emb))
        return node_feat


class SmoothConvBlockNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 hidden_dim=128,
                 conv_layer=3,
                 edge_emb_dim=64,
                 use_layer_norm=False,
                 use_batch_norm=True,
                 drop_edge=False,
                 activation='relu',
                 update_egde_emb=False,
                 ):
        super(SmoothConvBlockNew, self).__init__()
        self.conv = nn.ModuleList()
        self.edge_emb_dim = edge_emb_dim
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        self.drop_edge = drop_edge
        if use_batch_norm == use_layer_norm and use_batch_norm:
            raise Exception('Only one type of normalization at a time')
        if use_layer_norm or use_batch_norm:
            self.norm_layers = nn.ModuleList()

        for layer in range(conv_layer):
            if layer == 0:
                self.conv.append(SmoothConvLayerNew(in_node_feats=in_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            else:
                self.conv.append(SmoothConvLayerNew(in_node_feats=out_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(out_node_feats))
            elif use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(out_node_feats))

    def forward(self, h: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:

        for l, conv_layer in enumerate(self.conv):
            if self.use_layer_norm or self.use_batch_norm:
                h = conv_layer.forward(graph, self.norm_layers[l](h)) + h
            else:
                h = conv_layer.forward(graph, h) + h

        return h


# code from DGL documents
class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))


class WaterMDDynamicBoxNet(nn.Module):
    def __init__(self,
                 in_feats,
                 encoding_size,
                 out_feats,
                 bond=None,       #
                 hidden_dim=128,
                 conv_layer=4,
                 edge_embedding_dim=128,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False,
                 update_edge=False,
                 expand_edge=True):
        super(WaterMDDynamicBoxNet, self).__init__()
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                              out_node_feats=encoding_size,
                                              hidden_dim=hidden_dim,
                                              conv_layer=conv_layer,
                                              edge_emb_dim=edge_embedding_dim,
                                              use_layer_norm=use_layer_norm,
                                              use_batch_norm=not use_layer_norm,
                                              drop_edge=drop_edge,
                                              activation='silu',
                                              update_egde_emb=update_edge)

        self.edge_emb_dim = edge_embedding_dim
        self.expand_edge = expand_edge
        if self.expand_edge:
            self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(dropout)
        self.use_bond = not bond is None

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        self.node_encoder = nn.Linear(in_feats, encoding_size)
        if bond is not None:
            if self.expand_edge:
                self.edge_encoder = MLP(4 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                        activation='gelu')
            else:
                self.edge_encoder = MLP(4 + 1, self.edge_emb_dim, hidden_dim=hidden_dim,
                                        activation='gelu')
            self.bond_graph = self.build_bond_graph(bond)
        else:
            if self.expand_edge:
                self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                        activation='gelu')
            else:
                self.edge_encoder = MLP(3 + 1, self.edge_emb_dim, hidden_dim=hidden_dim,
                                        activation='gelu')
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')

    def calc_edge_feat(self, rel_pos_periodic, rel_pos_norm):

        if self.training:
            self.fit_length(rel_pos_norm)
            self._update_length_stat(self.length_scaler.mean_, np.sqrt(self.length_scaler.var_))
        rel_pos_periodic = -rel_pos_periodic / (rel_pos_norm + 1e-8)
        rel_pos_norm = (rel_pos_norm - self.length_mean) / self.length_std
        if self.expand_edge:
            edge_feat = torch.cat((rel_pos_periodic,
                                   rel_pos_norm,
                                   self.edge_expand(rel_pos_norm)), dim=1)
        else:
            edge_feat = torch.cat((rel_pos_periodic,
                                   rel_pos_norm), dim=1)
        return edge_feat

    def build_graph(self, fluid_pos, cutoff, box_size, self_loop=True):
        if isinstance(box_size, torch.Tensor):
            box_size = box_size.to(fluid_pos.device)
        elif isinstance(box_size, np.ndarray):
            box_size = torch.from_numpy(box_size).to(fluid_pos.device)

        with torch.no_grad():
            edge_idx, distance, distance_norm, _ = get_neighbor(fluid_pos,
                                                                cutoff, box_size)
        center_idx = edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        fluid_edge_feat = self.calc_edge_feat(distance, distance_norm.view(-1, 1))
        if not self.use_bond:
            fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat)) # [edge_num, 64]
            fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
            fluid_graph.edata['e'] = fluid_edge_emb
        else:
            edge_type = self.bond_graph.has_edges_between(center_idx, neigh_idx)
            fluid_edge_feat = torch.cat((fluid_edge_feat, edge_type.view(-1, 1)), dim=1)
            fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
            fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
            fluid_graph.edata['e'] = fluid_edge_emb

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
        return fluid_graph

    def build_graph_batches(self, pos_lst, box_size_lst, cutoff):
        graph_lst = []
        for pos, box_size in zip(pos_lst, box_size_lst):
            graph = self.build_graph(pos, cutoff, box_size)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph

    def build_bond_graph(self, bond):
        if isinstance(bond, np.ndarray):
            bond = torch.from_numpy(bond).cuda()
        bond_graph = dgl.graph((bond[:, 0], bond[:, 1]))
        bond_graph = dgl.add_reverse_edges(bond_graph)  # undirectional and symmetry
        return bond_graph

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1,1)
        self.length_scaler.partial_fit(length)

    def forward(self,
                fluid_pos_lst,  #   list of [N, 3]
                x,  # node feature    # [b*N, 3]
                box_size_lst,   #   list of scalar
                cutoff          # a scalar
                ):
        # fluid_graph = self.build_graph(fluid_pos, cutoff, box_size)
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, box_size_lst, cutoff)
        else:
            fluid_graph = self.build_graph(fluid_pos_lst[0], cutoff, box_size_lst[0])

        x = self.node_encoder(x)
        x = self.graph_conv(x, fluid_graph)

        x = self.graph_decoder(x)
        return x


class WaterMDNetNew(nn.Module):
    def __init__(self,
                 in_feats,
                 encoding_size,
                 out_feats,
                 box_size,   # can also be array
                 bond=None,       #
                 hidden_dim=128,
                 conv_layer=4,
                 edge_embedding_dim=128,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False):
        super(WaterMDNetNew, self).__init__()
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=edge_embedding_dim,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')

        self.edge_emb_dim = edge_embedding_dim
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(dropout)
        self.use_bond = not bond is None

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        if isinstance(box_size, np.ndarray):
            self.box_size = torch.from_numpy(box_size).float()
        else:
            self.box_size = box_size
        self.box_size = self.box_size

        self.node_encoder = nn.Linear(in_feats, encoding_size)
        if bond is not None:
            self.edge_encoder = MLP(4 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                    activation='gelu')
            self.use_bond = True
            self.bond_graph = self.build_bond_graph(bond)
        else:
            self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                    activation='gelu')
            self.use_bond = False
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')

    def calc_edge_feat(self,
                       src_idx: torch.Tensor,
                       dst_idx: torch.Tensor,
                       pos_src: torch.Tensor,
                       pos_dst=None) -> torch.Tensor:
        # this is the raw input feature

        # to enhance computation performance, dont track their calculation on graph
        if pos_dst is None:
            pos_dst = pos_src

        with torch.no_grad():
            rel_pos = pos_dst[dst_idx.long()] - pos_src[src_idx.long()]
            if isinstance(self.box_size, torch.Tensor):
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size.to(rel_pos.device),
                                                   self.box_size.to(rel_pos.device)) - 0.5 * self.box_size.to(rel_pos.device)
            else:
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size,
                                                   self.box_size) - 0.5 * self.box_size

            rel_pos_norm = rel_pos_periodic.norm(dim=1).view(-1, 1)  # [edge_num, 1]
            rel_pos_periodic /= rel_pos_norm + 1e-8   # normalized

        if self.training:
            self.fit_length(rel_pos_norm)
            self._update_length_stat(self.length_scaler.mean_, np.sqrt(self.length_scaler.var_))

        rel_pos_norm = (rel_pos_norm - self.length_mean) / self.length_std
        edge_feat = torch.cat((rel_pos_periodic,
                               rel_pos_norm,
                               self.edge_expand(rel_pos_norm)), dim=1)
        return edge_feat

    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)

        if not self.use_bond:
            fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
            fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
            fluid_graph.edata['e'] = fluid_edge_emb
        else:
            edge_type = self.bond_graph.has_edges_between(center_idx, neigh_idx)
            fluid_edge_feat = torch.cat((fluid_edge_feat, edge_type.view(-1, 1)), dim=1)
            fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
            fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
            fluid_graph.edata['e'] = fluid_edge_emb

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
        return fluid_graph

    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(edge_idx, pos)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph

    def build_bond_graph(self, bond) -> dgl.DGLGraph:
        if isinstance(bond, np.ndarray):
            bond = torch.from_numpy(bond).cuda()
        bond_graph = dgl.graph((bond[:, 0], bond[:, 1]))
        bond_graph = dgl.add_reverse_edges(bond_graph)  # undirectional and symmetry
        return bond_graph

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1, 1)
        self.length_scaler.partial_fit(length)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                x: torch.Tensor,  # node feature    # [b*N, 3]
                fluid_edge_lst: List[torch.Tensor]
                ) -> torch.Tensor:
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, fluid_edge_lst)
        else:
            fluid_graph = self.build_graph(fluid_edge_lst[0], fluid_pos_lst[0])
        x = self.node_encoder(x)
        x = self.graph_conv(x, fluid_graph)

        x = self.graph_decoder(x)
        return x


class SimpleMDNetNew(nn.Module):  # no bond, no learnable node encoder
    def __init__(self,
                 encoding_size,
                 out_feats,
                 box_size,   # can also be array
                 hidden_dim=128,
                 conv_layer=4,
                 edge_embedding_dim=128,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False):
        super(SimpleMDNetNew, self).__init__()
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=edge_embedding_dim,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')

        self.edge_emb_dim = edge_embedding_dim
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(dropout)

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        if isinstance(box_size, np.ndarray):
            self.box_size = torch.from_numpy(box_size).float()
        else:
            self.box_size = box_size
        self.box_size = self.box_size

        self.node_emb = nn.Parameter(torch.randn((1, encoding_size)), requires_grad=True)
        self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                activation='gelu')
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')

    def calc_edge_feat(self,
                       src_idx: torch.Tensor,
                       dst_idx: torch.Tensor,
                       pos_src: torch.Tensor,
                       pos_dst=None) -> torch.Tensor:
        # this is the raw input feature

        # to enhance computation performance, dont track their calculation on graph
        if pos_dst is None:
            pos_dst = pos_src

        with torch.no_grad():
            rel_pos = pos_dst[dst_idx.long()] - pos_src[src_idx.long()]
            if isinstance(self.box_size, torch.Tensor):
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size.to(rel_pos.device),
                                                   self.box_size.to(rel_pos.device)) - 0.5 * self.box_size.to(rel_pos.device)
            else:
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size,
                                                   self.box_size) - 0.5 * self.box_size

            rel_pos_norm = rel_pos_periodic.norm(dim=1).view(-1, 1)  # [edge_num, 1]
            rel_pos_periodic /= rel_pos_norm + 1e-8   # normalized

        if self.training:
            self.fit_length(rel_pos_norm)
            self._update_length_stat(self.length_scaler.mean_, np.sqrt(self.length_scaler.var_))

        rel_pos_norm = (rel_pos_norm - self.length_mean) / self.length_std
        edge_feat = torch.cat((rel_pos_periodic,
                               rel_pos_norm,
                               self.edge_expand(rel_pos_norm)), dim=1)
        return edge_feat

    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)

        fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        fluid_graph.edata['e'] = fluid_edge_emb

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
        return fluid_graph

    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(edge_idx, pos)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1, 1)
        self.length_scaler.partial_fit(length)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor]
                ) -> torch.Tensor:
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, fluid_edge_lst)
        else:
            fluid_graph = self.build_graph(fluid_edge_lst[0], fluid_pos_lst[0])
        num = np.sum([pos.shape[0] for pos in fluid_pos_lst])
        x = self.node_emb.repeat((num, 1))
        x = self.graph_conv(x, fluid_graph)

        x = self.graph_decoder(x)
        return x


