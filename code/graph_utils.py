
import jax
from jax_md import space, partition
from jax_md.space import pairwise_displacement
from jax import numpy as jnp

import warnings
warnings.filterwarnings("ignore")


class NeighborSearcher(object):
    def __init__(self, box_size, cutoff):
        # define a displacement function under periodic condition
        self.box_size = jnp.array(box_size)

        self.displacement_fn, _ = space.periodic(self.box_size)
        self.disp = jax.vmap(self.displacement_fn)
        self.dist = jax.vmap(space.metric(self.displacement_fn))
        self.cutoff = cutoff
        self.has_been_init = False
        self.neighbor_list_fn = partition.neighbor_list(self.displacement_fn,
                                                       self.box_size,
                                                       cutoff,
                                                       dr_threshold= cutoff / 6.,
                                                       mask_self=False)
        self.neighbor_list_fn_jit = jax.jit(self.neighbor_list_fn)
        self.neighbor_dist_jit = self.displacement_fn

    def init_new_neighbor_lst(self, pos):
        # Create a new neighbor list.
        pos = jnp.mod(pos, self.box_size)
        nbr = self.neighbor_list_fn(pos)
        self.has_been_init = True
        return nbr

    def update_neighbor_lst(self, pos, nbr):
        pos = jnp.mod(pos, self.box_size)
        # update_idx = np.any(self.dist(pos, nbr.reference_position) > (self.cutoff / 10.))

        nbr = self.neighbor_list_fn_jit(pos, nbr)
        if nbr.did_buffer_overflow:
            nbr = self.neighbor_list_fn(pos)

        return nbr


def graph_network_nbr_fn(displacement_fn,
                         cutoff,
                         N):

    def nbrlst_to_edge_mask(pos: jnp.ndarray, neigh_idx: jnp.ndarray):
        # notice here, pos must be jax numpy array, otherwise fancy indexing will fail
        d = jax.partial(displacement_fn)
        d = space.map_neighbor(d)
        pos_neigh = pos[neigh_idx]
        dR = d(pos, pos_neigh)

        dr_2 = space.square_distance(dR)
        mask = jnp.logical_and(neigh_idx != N, dr_2 < cutoff ** 2)

        return mask

    return nbrlst_to_edge_mask


# def graph_network_nbr_fn_with_type_mask(displacement_fn,
#                                          cutoff,
#                                          N):
#
#     def nbrlst_to_edge_mask(pos: jnp.ndarray, neigh_idx: jnp.ndarray, type_mask: jnp.ndarray):
#         # notice here, pos must be jax numpy array, otherwise fancy indexing will fail
#         d = jax.partial(displacement_fn)
#         d = space.map_neighbor(d)
#         pos_neigh = pos[neigh_idx]
#         dR = d(pos, pos_neigh)
#
#         dr_2 = space.square_distance(dR)
#         mask = jnp.logical_and(neigh_idx != N, dr_2 < cutoff ** 2)
#         mask = jnp.logical_and(type_mask, mask)
#         return mask
#
#     return nbrlst_to_edge_mask
