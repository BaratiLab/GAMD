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

from hack_integrator import HackLangevinIntegrator, HackDummyIntegrator, HackHalfVelocityIntegrator
BOX_SCALE = 2.0
DT = 2.0

platform = Platform.getPlatformByName('CPU')
P_NUM = 258

fluid = testsystems.LennardJonesFluid(nparticles=P_NUM, reduced_density=0.50, shift=True)
[topology, system, positions] = fluid.topology, fluid.system, fluid.positions
pos = np.load('../init_pos.npy')
p_num = positions.shape[0]
timestep = DT * unit.femtoseconds
temperature = 100 * unit.kelvin

GAMMA = 25.0 / unit.picosecond
dummy_integrator = CompoundIntegrator()
integrator1 = HackLangevinIntegrator(temperature,
                                     collision_rate=GAMMA,
                                     timestep=timestep)
integrator2 = HackHalfVelocityIntegrator(timestep=timestep)
dummy_integrator.addIntegrator(integrator1)
dummy_integrator.addIntegrator(integrator2)

dummy_simulator = Simulation(topology, system, dummy_integrator, platform=platform)

dummy_simulator.context.setPositions(pos*unit.angstrom)
dummy_simulator.context.setVelocitiesToTemperature(temperature)
# mass = np.ones((p_num*3, 1), dtype=np.float32)*1.008
# mass[::3] = 15.9994
# mass = mass*unit.amu
print(system.getForces())

def remove_force_offset(force):
    offset = np.mean(force)
    force = force - offset
    return force

# ===========================================================================
# ===========================================================================
from types import SimpleNamespace
from train_network_lj import ParticleNetLightning
NUM_OF_ATOMS = positions.shape[0]
print(f'Simulating {NUM_OF_ATOMS} number of atoms')
PATH = '../model_ckpt_lj/checkpoint.ckpt'
SCALER_CKPT = '../model_ckpt_lj/scaler.npz'
args = SimpleNamespace(use_layer_norm=True,
                       encoding_size=128,
                       hidden_dim=128,
                       edge_embedding_dim=128,
                      drop_edge=False,
                       conv_layer=4,
                      rotate_aug=False,
                       update_edge=False,
                       use_part=False,
                      data_dir='',
                      loss='mae')
model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model.load_training_stats(SCALER_CKPT)
model.cuda()
model.eval()

dataReporter = StateDataReporter('log_nvt_gnn_langevin_lj.txt', 100,
                                 totalSteps=int(100000//DT),
                                step=True, time=True,
                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                 temperature=True, separator='\t')
dummy_simulator.reporters.append(dataReporter)
dummy_simulator.minimizeEnergy(1e-6)

dummy_state = dummy_simulator.context.getState(getPositions=True,
                                               getVelocities=True,
                                               getForces=True)
pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
vel = dummy_state.getVelocities(asNumpy=True)

force = model.predict_forces(pos)
force = force*(unit.kilojoules_per_mole/unit.nanometers)

print(f'Using collision frequency: {GAMMA}')
for t in range(int(50000//DT)):
    if (t+1)%500 == 0:
        print(f'Finished {(t+1)} steps')

    dummy_integrator.setCurrentIntegrator(0)
    integrator1.setPerDofVariableByName('force_last', force)
    dummy_simulator.step(1)
    dummy_state = dummy_simulator.context.getState(getPositions=True,
                                                   getForces=True,
                                                   enforcePeriodicBox=True)
    pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    gt_force = dummy_state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometers)

    force = model.predict_forces(pos)

    force = force * (unit.kilojoules_per_mole / unit.nanometers)
    dummy_integrator.setCurrentIntegrator(1)
    integrator2.setPerDofVariableByName('gnn_force', force)
    dummy_simulator.step(1)

