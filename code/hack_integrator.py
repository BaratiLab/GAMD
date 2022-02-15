import simtk.openmm as mm
from sys import stdout
from openmmtools import integrators
from openmmtools.integrators import ThermostatedIntegrator, NoseHooverChainVelocityVerletIntegrator
from openmmtools.constants import kB
from openmmtools import respa, utils
from simtk.openmm.app import *
import simtk.unit as unit
import logging
# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole
from openmmtools import states
logger = logging.getLogger(__name__)
import numpy as np


class HackAndersenVVIntegrator(ThermostatedIntegrator):

    """Velocity Verlet integrator with Andersen thermostat using per-particle collisions (rather than massive collisions).

    References
    ----------
    Hans C. Andersen "Molecular dynamics simulations at constant pressure and/or temperature", Journal of Chemical Physics 72, 2384-2393 (1980)
    http://dx.doi.org/10.1063/1.439486

    Examples
    --------

    Create a velocity Verlet integrator with Andersen thermostat.

    >>> timestep = 1.0 * unit.femtoseconds
    >>> collision_rate = 91.0 / unit.picoseconds
    >>> temperature = 298.0 * unit.kelvin
    >>> integrator = AndersenVelocityVerletIntegrator(temperature, collision_rate, timestep)

    Notes
    ------
    The velocity Verlet integrator is taken verbatim from Peter Eastman's example in the CustomIntegrator header file documentation.
    The efficiency could be improved by avoiding recomputation of sigma_v every timestep.

    """

    def __init__(self, temperature=298 * unit.kelvin, collision_rate=91.0 / unit.picoseconds, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator with Andersen thermostat, implemented as per-particle collisions (rather than massive collisions).

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default=298*unit.kelvin
           The temperature of the fictitious bath.
        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default=91/unit.picoseconds
           The collision rate with fictitious bath particles.
        timestep : np.unit.Quantity compatible with femtoseconds, default=1*unit.femtoseconds
           The integration timestep.

        """
        super(HackAndersenVVIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("p_collision", timestep * collision_rate)  # per-particle collision probability per timestep
        self.addPerDofVariable("sigma_v", 0)  # velocity distribution stddev for Maxwell-Boltzmann (computed later)
        self.addPerDofVariable("collision", 0)  # 1 if collision has occured this timestep, 0 otherwise
        self.addPerDofVariable("x1", 0)  # for constraints

        #
        # Update velocities from Maxwell-Boltzmann distribution for particles that collide.
        #
        self.addComputeTemperatureDependentConstants({"sigma_v": "sqrt(kT/m)"})
        self.addComputePerDof("collision", "step(p_collision-uniform)")  # if collision has occured this timestep, 0 otherwise
        self.addComputePerDof("v", "(1-collision)*v + collision*sigma_v*gaussian")  # randomize velocities of particles that have collided

        #
        # Velocity Verlet step

        # hacky force 1
        self.addPerDofVariable("test1", 0.)
        self.addPerDofVariable("test2", 0.)
        #
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*test1/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*test2/m+(x-x1)/dt")
        self.addConstrainVelocities()


# first half of BAOAB langevin integrator
class HackLangevinIntegrator(ThermostatedIntegrator):
    _kinetic_energy = "0.5 * m * v * v"

    def __init__(self,
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 ):
        """
        One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt

        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass

        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal
        """

        # Compute constants
        gamma = collision_rate
        self._gamma = gamma
        # Create a new CustomIntegrator
        super(HackLangevinIntegrator, self).__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)
        h = timestep
        self.addGlobalVariable("a", np.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)
        self.addPerDofVariable("force_last", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        # update velocities first, B
        self.addComputePerDof("v", "v + (dt / 2) * force_last / m")
        self.addConstrainVelocities()

        # update positions (and velocities, if there are constraints)  A
        self.addComputePerDof("x", "x + ((dt / 2) * v)")
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
        self.addConstrainVelocities()

        # stochastic step, O
        # update velocities
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        # update positions (and velocities, if there are constraints)  A
        self.addComputePerDof("x", "x + ((dt / 2) * v)")
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
        self.addConstrainVelocities()

        # update velocities the second time, B
        # self.addComputePerDof("v", "v + (dt / 2) * f / m")
        # self.addConstrainVelocities()

class HackHalfVelocityIntegrator(mm.CustomIntegrator):
    def __init__(self, timestep):
        super(HackHalfVelocityIntegrator, self).__init__(timestep)
        # same as in openmm.app.StateDataReporter._initializeConstants
        self.addPerDofVariable('gnn_force', 0)
        # update velocities the second time, B
        self.addComputePerDof("v", "v + (dt / 2) * gnn_force / m")
        self.addConstrainVelocities()


# first half of nose hoover integrator
class HackNoseHooverIntegrator(ThermostatedIntegrator):
    YSWeights = {
        1: [1.0000000000000000],
        3: [0.8289815435887510, -0.6579630871775020, 0.8289815435887510],
        5: [0.2967324292201065, 0.2967324292201065, -0.1869297168804260, 0.2967324292201065, 0.2967324292201065]
    }

    def __init__(self, system=None, temperature=298 * unit.kelvin, collision_frequency=50 / unit.picoseconds,
                 timestep=0.001 * unit.picoseconds, chain_length=5, num_mts=5, num_yoshidasuzuki=5):

        super(HackNoseHooverIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.n_c = num_mts
        self.n_ys = num_yoshidasuzuki
        try:
            self.weights = self.YSWeights[self.n_ys]
        except KeyError:
            raise Exception("Invalid Yoshida-Suzuki value. Allowed values are: %s" %
                            ",".join(map(str, self.YSWeights.keys())))
        if chain_length < 0:
            raise Exception("Nosé-Hoover chain length must be at least 0")
        if chain_length == 0:
            logger.warning('Nosé-Hoover chain length is 0; falling back to regular velocity verlet algorithm.')
        self.M = chain_length

        # Define the "mass" of the thermostat particles (multiply by ndf for particle 0)
        kT = self.getGlobalVariableByName('kT')
        frequency = collision_frequency.value_in_unit(unit.picoseconds ** -1)
        Q = kT / frequency ** 2

        #
        # Compute the number of degrees of freedom.
        #
        if system is None:
            logger.warning('The system was not passed to the NoseHooverChainVelocityVerletIntegrator. '
                           'For systems with constraints, the simulation will run at the wrong temperature.')
            # Fall back to old scheme, which only works for unconstrained systems
            self.addGlobalVariable("ndf", 0)
            self.addPerDofVariable("ones", 1.0)
            self.addComputeSum("ndf", "ones")
        else:
            # same as in openmm.app.StateDataReporter._initializeConstants
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3

            self.addGlobalVariable("ndf", dof)  # number of degrees of freedom

        #
        # Define global variables
        #
        self.addGlobalVariable("bathKE", 0.0)  # Thermostat bath kinetic energy
        self.addGlobalVariable("bathPE", 0.0)  # Thermostat bath potential energy
        self.addGlobalVariable("KE2", 0.0)  # Twice the kinetic energy
        self.addGlobalVariable("Q", Q)  # Thermostat particle "mass"
        self.addGlobalVariable("scale", 1.0)
        self.addGlobalVariable("aa", 0.0)
        self.addGlobalVariable("wdt", 0.0)
        for w in range(self.n_ys):
            self.addGlobalVariable("w{}".format(w), self.weights[w])

        #
        # Initialize thermostat parameters
        #
        for i in range(self.M):
            self.addGlobalVariable("xi{}".format(i), 0)  # Thermostat particle
            self.addGlobalVariable("vxi{}".format(i), 0)  # Thermostat particle velocities in ps^-1
            self.addGlobalVariable("G{}".format(i), -frequency ** 2)  # Forces on thermostat particles in ps^-2
            self.addGlobalVariable("Q{}".format(i), 0)  # Thermostat "masses" in ps^2 kJ/mol
        # The masses need the number of degrees of freedom, which is approximated here.  Need a
        # better solution eventually, to properly account for constraints, translations, etc.
        self.addPerDofVariable("x1", 0)
        if self.M:
            self.addComputeGlobal("Q0", "ndf*Q")
            for i in range(1, self.M):
                self.addComputeGlobal("Q{}".format(i), "Q")

        # hacky force
        self.addPerDofVariable("force_last", 0.)
        #
        # Take a velocity verlet step, with propagation of thermostat before and after
        #
        if self.M: self.propagateNHC()
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*force_last/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+(x-x1)/dt")

    def reset(self, new_frequency):
        kT = self.getGlobalVariableByName('kT')
        frequency = new_frequency.value_in_unit(unit.picoseconds ** -1)
        Q = kT / frequency ** 2
        self.setGlobalVariableByName('Q', Q)
        self.addComputeGlobal("Q0", "ndf*Q")
        for i in range(1, self.M):
            self.addComputeGlobal("Q{}".format(i), "Q")


    def propagateNHC(self):
        """ Propagate the Nosé-Hoover chain """
        self.addComputeGlobal("scale", "1.0")
        self.addComputeSum("KE2", "m*v^2")
        self.addComputeGlobal("G0", "(KE2 - ndf*kT)/Q0")
        for ncval in range(self.n_c):
            for nysval in range(self.n_ys):
                self.addComputeGlobal("wdt", "w{}*dt/{}".format(nysval, self.n_c))
                self.addComputeGlobal("vxi{}".format(self.M-1), "vxi{} + 0.25*wdt*G{}".format(self.M-1, self.M-1))
                for j in range(self.M-2, -1, -1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j+1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j,j))
                # update particle velocities
                self.addComputeGlobal("aa", "exp(-0.5*wdt*vxi0)")
                self.addComputeGlobal("scale", "scale*aa")
                # update the thermostat positions
                for j in range(self.M):
                    self.addComputeGlobal("xi{}".format(j), "xi{} + 0.5*wdt*vxi{}".format(j,j))
                # update the forces
                self.addComputeGlobal("G0", "(scale*scale*KE2 - ndf*kT)/Q0")
                # update thermostat velocities
                for j in range(self.M-1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j+1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j,j))
                    self.addComputeGlobal("G{}".format(j+1), "(Q{}*vxi{}*vxi{} - kT)/Q{}".format(j,j,j,j+1))
                self.addComputeGlobal("vxi{}".format(self.M-1), "vxi{} + 0.25*wdt*G{}".format(self.M-1, self.M-1))
        # update particle velocities
        self.addComputePerDof("v", "scale*v")

    def get_global_variable_from_integrator(self, name: str, integrator):
        value = integrator.getGlobalVariableByName(name)
        self.setGlobalVariableByName(name, value)

    def copy_state_from_integrator(self, integrator):
        # copy from another integrator
        for i in range(self.M):
            self.get_global_variable_from_integrator("xi{}".format(i), integrator)  # Thermostat particle
            self.get_global_variable_from_integrator("vxi{}".format(i),
                                                     integrator)  # Thermostat particle velocities in ps^-1
            self.get_global_variable_from_integrator("G{}".format(i),
                                                     integrator)  # Forces on thermostat particles in ps^-2
            self.get_global_variable_from_integrator("Q{}".format(i), integrator)  # Thermostat "masses" in ps^2 kJ/mol


# second half of nose hoover integrator
class HackHalfNoseHooverIntegrator(ThermostatedIntegrator):
    YSWeights = {
        1: [1.0000000000000000],
        3: [0.8289815435887510, -0.6579630871775020, 0.8289815435887510],
        5: [0.2967324292201065, 0.2967324292201065, -0.1869297168804260, 0.2967324292201065, 0.2967324292201065]
    }

    def __init__(self, system=None, temperature=298 * unit.kelvin, collision_frequency=50 / unit.picoseconds,
                 timestep=0.001 * unit.picoseconds, chain_length=5, num_mts=5, num_yoshidasuzuki=5):

        super(HackHalfNoseHooverIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.n_c = num_mts
        self.n_ys = num_yoshidasuzuki
        try:
            self.weights = self.YSWeights[self.n_ys]
        except KeyError:
            raise Exception("Invalid Yoshida-Suzuki value. Allowed values are: %s" %
                            ",".join(map(str, self.YSWeights.keys())))
        if chain_length < 0:
            raise Exception("Nosé-Hoover chain length must be at least 0")
        if chain_length == 0:
            logger.warning('Nosé-Hoover chain length is 0; falling back to regular velocity verlet algorithm.')
        self.M = chain_length

        # Define the "mass" of the thermostat particles (multiply by ndf for particle 0)
        kT = self.getGlobalVariableByName('kT')
        frequency = collision_frequency.value_in_unit(unit.picoseconds ** -1)
        Q = kT / frequency ** 2

        #
        # Compute the number of degrees of freedom.
        #
        if system is None:
            logger.warning('The system was not passed to the NoseHooverChainVelocityVerletIntegrator. '
                           'For systems with constraints, the simulation will run at the wrong temperature.')
            # Fall back to old scheme, which only works for unconstrained systems
            self.addGlobalVariable("ndf", 0)
            self.addPerDofVariable("ones", 1.0)
            self.addComputeSum("ndf", "ones")
        else:
            # same as in openmm.app.StateDataReporter._initializeConstants
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3

            self.addGlobalVariable("ndf", dof)  # number of degrees of freedom

        #
        # Define global variables
        #
        self.addGlobalVariable("bathKE", 0.0)  # Thermostat bath kinetic energy
        self.addGlobalVariable("bathPE", 0.0)  # Thermostat bath potential energy
        self.addGlobalVariable("KE2", 0.0)  # Twice the kinetic energy
        self.addGlobalVariable("Q", Q)  # Thermostat particle "mass"
        self.addGlobalVariable("scale", 1.0)
        self.addGlobalVariable("aa", 0.0)
        self.addGlobalVariable("wdt", 0.0)
        for w in range(self.n_ys):
            self.addGlobalVariable("w{}".format(w), self.weights[w])

        #
        # Initialize thermostat parameters
        #
        for i in range(self.M):
            self.addGlobalVariable("xi{}".format(i), 0)  # Thermostat particle
            self.addGlobalVariable("vxi{}".format(i), 0)  # Thermostat particle velocities in ps^-1
            self.addGlobalVariable("G{}".format(i), -frequency ** 2)  # Forces on thermostat particles in ps^-2
            self.addGlobalVariable("Q{}".format(i), 0)  # Thermostat "masses" in ps^2 kJ/mol
        # The masses need the number of degrees of freedom, which is approximated here.  Need a
        # better solution eventually, to properly account for constraints, translations, etc.
        self.addPerDofVariable("x1", 0)
        if self.M:
            self.addComputeGlobal("Q0", "ndf*Q")
            for i in range(1, self.M):
                self.addComputeGlobal("Q{}".format(i), "Q")

        # hacky force
        self.addPerDofVariable("gnn_force", 0.)

        self.addComputePerDof("v", "v+0.5*dt*gnn_force/m")
        self.addConstrainVelocities()
        if self.M: self.propagateNHC()
        # Compute heat bath energies
        self.computeEnergies()

    def reset(self, new_frequency):
        kT = self.getGlobalVariableByName('kT')
        frequency = new_frequency.value_in_unit(unit.picoseconds ** -1)
        Q = kT / frequency ** 2
        self.setGlobalVariableByName('Q', Q)
        self.addComputeGlobal("Q0", "ndf*Q")
        for i in range(1, self.M):
            self.addComputeGlobal("Q{}".format(i), "Q")

    def get_global_variable_from_integrator(self, name: str, integrator):
        value = integrator.getGlobalVariableByName(name)
        self.setGlobalVariableByName(name, value)

    def copy_state_from_integrator(self, integrator):
        self.get_global_variable_from_integrator("bathKE", integrator)  # Thermostat bath kinetic energy
        self.get_global_variable_from_integrator("bathPE", integrator)  # Thermostat bath potential energy
        #
        # copy from another integrator
        #
        for i in range(self.M):
            self.get_global_variable_from_integrator("xi{}".format(i), integrator)  # Thermostat particle
            self.get_global_variable_from_integrator("vxi{}".format(i),
                                                     integrator)  # Thermostat particle velocities in ps^-1
            self.get_global_variable_from_integrator("G{}".format(i),
                                                     integrator)  # Forces on thermostat particles in ps^-2
            self.get_global_variable_from_integrator("Q{}".format(i), integrator)  # Thermostat "masses" in ps^2 kJ/mol

    def propagateNHC(self):
        """ Propagate the Nosé-Hoover chain """
        self.addComputeGlobal("scale", "1.0")
        self.addComputeSum("KE2", "m*v^2")
        self.addComputeGlobal("G0", "(KE2 - ndf*kT)/Q0")
        for ncval in range(self.n_c):
            for nysval in range(self.n_ys):
                self.addComputeGlobal("wdt", "w{}*dt/{}".format(nysval, self.n_c))
                self.addComputeGlobal("vxi{}".format(self.M - 1), "vxi{} + 0.25*wdt*G{}".format(self.M - 1, self.M - 1))
                for j in range(self.M - 2, -1, -1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j + 1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j, j))
                # update particle velocities
                self.addComputeGlobal("aa", "exp(-0.5*wdt*vxi0)")
                self.addComputeGlobal("scale", "scale*aa")
                # update the thermostat positions
                for j in range(self.M):
                    self.addComputeGlobal("xi{}".format(j), "xi{} + 0.5*wdt*vxi{}".format(j, j))
                # update the forces
                self.addComputeGlobal("G0", "(scale*scale*KE2 - ndf*kT)/Q0")
                # update thermostat velocities
                for j in range(self.M - 1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j + 1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j, j))
                    self.addComputeGlobal("G{}".format(j + 1), "(Q{}*vxi{}*vxi{} - kT)/Q{}".format(j, j, j, j + 1))
                self.addComputeGlobal("vxi{}".format(self.M - 1), "vxi{} + 0.25*wdt*G{}".format(self.M - 1, self.M - 1))
        # update particle velocities
        self.addComputePerDof("v", "scale*v")

    def computeEnergies(self):
        """ Computes kinetic and potential energies for the heat bath """
        # Bath kinetic energy
        self.addComputeGlobal("bathKE", "0.0")
        for i in range(self.M):
            self.addComputeGlobal("bathKE", "bathKE + 0.5*Q{}*vxi{}^2".format(i, i))
        # Bath potential energy
        self.addComputeGlobal("bathPE", "ndf*xi0")
        for i in range(1, self.M):
            self.addComputeGlobal("bathPE", "bathPE + xi{}".format(i))
        self.addComputeGlobal("bathPE", "kT*bathPE")

