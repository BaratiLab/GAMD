import torch
from openmmtools import testsystems
from simtk.openmm import *
from simtk.openmm.app import *
import simtk.unit as unit

import logging

import numpy as np

from openmmtools.constants import kB
from openmmtools import respa, utils

logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole
from openmmtools import states, integrators
import time
import numpy as np
import sys
import os
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from hack_integrator import HackNoseHooverIntegrator, HackHalfNoseHooverIntegrator
BOX_SCALE = 2.0
DT = 2.0

platform = Platform.getPlatformByName('CPU')

waterbox = testsystems.WaterBox(
                                box_edge=BOX_SCALE*unit.nanometers,
                                cutoff=min(10.0, BOX_SCALE*5.0-0.5)*unit.angstrom,
                                model='tip3p',
                                constrained=True)
[topology, system, positions] = [waterbox.topology, waterbox.system, waterbox.positions]

p_num = positions.shape[0] // 3
timestep = DT * unit.femtoseconds
temperature = 300 * unit.kelvin

GAMMA = 25. / unit.picosecond
dummy_integrator = CompoundIntegrator()
integrator1 = HackNoseHooverIntegrator(system, temperature,
                                       collision_frequency=GAMMA,
                                       chain_length=10,
                                       timestep=timestep)
integrator2 = HackHalfNoseHooverIntegrator(system, temperature,
                                           collision_frequency=GAMMA,
                                           chain_length=10,
                                           timestep=timestep)
dummy_integrator.addIntegrator(integrator1)
dummy_integrator.addIntegrator(integrator2)

dummy_simulator = Simulation(topology, system, dummy_integrator, platform=platform)

dummy_simulator.context.setPositions(positions)
dummy_simulator.context.setVelocitiesToTemperature(temperature)

# ===========================================================================
from types import SimpleNamespace
from train_network_real_large import ParticleNetLightning
NUM_OF_ATOMS = positions.shape[0]                  # (258*3)
print(f'Simulating {NUM_OF_ATOMS} number of atoms')
PATH = '../model_ckpt_dft/checkpoint.ckpt'
SCALER_CKPT = '../model_ckpt_dft/scaler.npz'
args = SimpleNamespace(use_layer_norm=True,
                       encoding_size=256,
                       hidden_dim=128,
                       edge_embedding_dim=256,
                       conv_layer=5,
                      drop_edge=False,
                      cutoff=9.5,
                      rotate_aug=False,
                       update_edge=False,
                       use_part=False,
                       expand_edge=True,
                      data_dir='',
                      loss='mse')
model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model.load_training_stats(SCALER_CKPT)
model.cuda()
model.eval()

particle_type = []
for i in range(NUM_OF_ATOMS):
    particle_type.append(1 if i % 3 == 0 else 0)   # O: 1, H: 0
particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)
# transform into one hot encoding
particle_type_one_hot = np.zeros((particle_type.size, 1), dtype=np.float32)
particle_type_one_hot[particle_type.reshape(-1) == 1] = 1
feat = torch.from_numpy(particle_type_one_hot).float().cuda()

dataReporter = StateDataReporter(f'./log_nvt_gnn_nosehoover.txt',
                                 250,
                                 totalSteps=int(100000//DT),
                                step=True, time=True,
                                 kineticEnergy=True,
                                 temperature=True, separator='\t')
dummy_simulator.reporters.append(dataReporter)
dummy_simulator.minimizeEnergy()

dummy_state = dummy_simulator.context.getState(getPositions=True,
                                               getVelocities=True,
                                               getForces=True)
pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.bohrs)
vel = dummy_state.getVelocities(asNumpy=True)
box_size = (np.array([BOX_SCALE*10., BOX_SCALE*10., BOX_SCALE*10.], dtype=np.float32) * unit.angstrom).value_in_unit(unit.bohrs)

force = model.predict_forces(feat, pos, box_size)
force = (force * 2625.5 / 0.0529177)*(unit.kilojoules_per_mole/unit.nanometers)


print(f'Using collision frequency: {GAMMA}')
for t in range(int(50000//DT)):
    if (t+1)%500 == 0:
        print(f'Finished {(t+1)} steps')

    dummy_integrator.setCurrentIntegrator(0)
    if t != 0:
        integrator1.copy_state_from_integrator(integrator2)
    integrator1.setPerDofVariableByName('force_last', force)
    dummy_simulator.step(1)

    dummy_state = dummy_simulator.context.getState(getPositions=True,
                                                   enforcePeriodicBox=True)
    pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.bohrs)
    force = model.predict_forces(feat, pos, box_size)
    force = (force * 2625.5 / 0.0529177)*(unit.kilojoules_per_mole/unit.nanometers)
    dummy_integrator.setCurrentIntegrator(1)
    integrator2.copy_state_from_integrator(integrator1)
    integrator2.setPerDofVariableByName('gnn_force', force)
    dummy_simulator.step(1)

