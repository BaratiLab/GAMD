from openmmtools import testsystems
from simtk.openmm.app import *
import simtk.unit as unit

import logging

import numpy as np

from openmmtools.constants import kB
from openmmtools import respa, utils

logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
from openmmtools import states, integrators
import time
import numpy as np
import sys
import os
from matplotlib import pyplot as plt


def get_rotation_matrix():
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(-1.0, 1.0, size=(3,)) * np.pi
    print(f'Using angle: {angles}')
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


BOX_SCALE = 2
DT = 2
for seed in range(10):
    print(f'Running seed: {seed}')
    P_NUM = 258
    fluid = testsystems.LennardJonesFluid(nparticles=P_NUM, reduced_density=0.50, shift=True)
    [topology, system, positions] = fluid.topology, fluid.system, fluid.positions

    R = get_rotation_matrix()
    positions = positions.value_in_unit(unit.angstrom)
    positions, off = center_positions(positions)
    positions = np.matmul(positions, R)
    positions += off
    positions += np.random.randn(positions.shape[0], positions.shape[1]) * 0.005
    positions *= unit.angstrom

    timestep = DT * unit.femtoseconds
    temperature = 100 * unit.kelvin
    chain_length = 10
    friction = 25. / unit.picosecond
    num_mts = 5
    num_yoshidasuzuki = 5

    integrator1 = integrators.NoseHooverChainVelocityVerletIntegrator(system,
                                                                      temperature,
                                                                      friction,
                                                                      timestep, chain_length, num_mts, num_yoshidasuzuki)

    simulation = Simulation(topology, system, integrator1)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.minimizeEnergy(tolerance=1*unit.kilojoule/unit.mole)
    simulation.step(1)

    os.makedirs(f'./lj_data/', exist_ok=True)
    dataReporter_gt = StateDataReporter(f'./log_nvt_lj_{seed}.txt', 50, totalSteps=50000,
        step=True, time=True, speed=True, progress=True, elapsedTime=True, remainingTime=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                                     separator='\t')
    simulation.reporters.append(dataReporter_gt)

    for t in range(1000):
        if (t+1)%100 == 0:
            print(f'Finished {(t+1)*50} steps')
        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True,
                                             enforcePeriodicBox=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        np.savez(f'../md_dataset/lj_data/data_{seed}_{t}.npz',
                 pos=pos,
                 vel=vel,
                 forces=force)
        simulation.step(50)

