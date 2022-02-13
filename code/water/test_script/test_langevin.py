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
BOX_SCALE = 2.0   # to match the scale in experimental data
DT = 2.0

platform = Platform.getPlatformByName('CPU')

waterbox = testsystems.WaterBox(
                                box_edge=BOX_SCALE*unit.nanometers,
                                cutoff=min(10.0, BOX_SCALE*5.0-0.5)*unit.angstrom,
                                model='tip3p',
                                constrained=True)
[topology, system, _] = [waterbox.topology, waterbox.system, waterbox.positions]
positions = np.load('../init_pos.npy')*unit.angstrom   # determined init of position
p_num = positions.shape[0] // 3
timestep = DT * unit.femtoseconds
temperature = 300 * unit.kelvin
kBT = kB * temperature

GAMMA = 25.0 / unit.picosecond
dummy_integrator = CompoundIntegrator()
integrator1 = HackLangevinIntegrator(temperature,
                                     collision_rate=GAMMA,
                                     timestep=timestep)
integrator2 = HackHalfVelocityIntegrator(timestep=timestep)
dummy_integrator.addIntegrator(integrator1)
dummy_integrator.addIntegrator(integrator2)

dummy_simulator = Simulation(topology, system, dummy_integrator, platform=platform)

dummy_simulator.context.setPositions(positions)
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
from train_network_tip3p import ParticleNetLightning
NUM_OF_ATOMS = positions.shape[0]                  # (258*3)
print(f'Simulating {NUM_OF_ATOMS} number of atoms')
PATH = '../model_ckpt_tip3p/checkpoint.ckpt'
SCALER_CKPT = '../model_ckpt_tip3p/scaler.npz'
args = SimpleNamespace(use_layer_norm=True,
                       encoding_size=128,
                       hidden_dim=128,
                       edge_embedding_dim=128,
                      drop_edge=False,
                      rotate_aug=False,
                      data_dir='',
                      loss='mae')
model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model.load_training_stats(SCALER_CKPT)
model.cuda()
model.eval()

particle_type = []
for i in range(NUM_OF_ATOMS):
    particle_type.append(1 if i % 3 == 0 else 0)
particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)
# transform into one hot encoding
particle_type_one_hot = np.zeros((particle_type.size, 1), dtype=np.float32)
particle_type_one_hot[particle_type.reshape(-1) == 1] = 1
feat = torch.from_numpy(particle_type_one_hot).float().cuda()

# dataReporter = StateDataReporter('log_nvt_gnn_langevin.txt', 100,
#                                  totalSteps=int(100000//DT),
#                                 step=True, time=True,
#                                 potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
#                                  temperature=True, separator='\t')
# dummy_simulator.reporters.append(dataReporter)

dummy_simulator.minimizeEnergy()

dummy_state = dummy_simulator.context.getState(getPositions=True,
                                               getVelocities=True,
                                               getForces=True)
pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
vel = dummy_state.getVelocities(asNumpy=True)

force = model.predict_forces(feat, pos)
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

    force = model.predict_forces(feat, pos, verbose=True)
    mae = np.mean(np.abs(gt_force-force))*0.0010364   # to ev/A
    force = force * (unit.kilojoules_per_mole / unit.nanometers)
    dummy_integrator.setCurrentIntegrator(1)
    integrator2.setPerDofVariableByName('gnn_force', force)
    dummy_simulator.step(1)


