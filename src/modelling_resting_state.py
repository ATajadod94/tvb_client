_author_ = 'Alireza Tajadod'
_project_ = 'The virtual brain'

## imports
import os
from pathlib import PurePosixPath
import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator import models, simulator, coupling, integrators, monitors
from tvb.datatypes import connectivity

from tvb.datatypes.time_series import TimeSeriesRegion
import tvb.analyzers.correlation_coefficient as corr_coeff

## Load and Prepare data
#

class structural():
    def __init__(self, path, speed=np.inf):
        ## numeric
        self.path = PurePosixPath(path)
        self.connectivity = connectivity.Connectivity.from_file(path)
        self.num_regions = len(self.connectivity.region_labels)
        self.speed = speed
        self.tract_lengths = self.connectivity.tract_lengths

        ## Visual
        self.labels = self.connectivity.region_labels

        ## configure
        self.connectivity.configure()
        self.SC = self.connectivity.weights

    def upper_triangle(self):
        return np.triu(self.connectivity.weights)

    def visualize(self):
        fig = plt.figure(figsize = (15,15))
        fig.suptitle('Hag SC', fontsize=20)

        self.plotweights()
        self.plottracts()
        fig.tight_layout()
        fig.subplots_adjust(top=1.5)
        plt.show()

    def plotweights(self, interpolation='nearest', cmap='jet'):
        plt.subplot(121)
        plt.imshow(self.SC, interpolation=interpolation, cmap=cmap, aspect='equal')
        plt.xticks(range(0, self.num_regions), self.labels, fontsize=7, rotation=90)
        plt.yticks(range(0, self.num_regions), self.labels, fontsize=7)
        cb = plt.colorbar(shrink=0.25)
        cb.set_label('weights', fontsize=14)

    def plottracts(self, interpolation='nearest', cmap='jet'):
        plt.subplot(122)
        plt.imshow(self.tract_lengths, interpolation=interpolation, cmap=cmap, aspect='equal')
        plt.xticks(range(0, self.num_regions), self.labels, fontsize=7, rotation=90)
        plt.yticks(range(0, self.num_regions), self.labels, fontsize=7)
        cb = plt.colorbar(shrink=0.25)
        cb.set_label('tracts', fontsize=14)

class functional():
    def __init__(self, path):
        self.path = PurePosixPath(path)
        self.weights = np.load(path)

    def upper_triangle(self):
        return np.triu(self.weights, 1)

    def plotweights(self, interpolation='nearest', cmap='jet'):
        plt.subplot(121)
        plt.imshow(self.weights, interpolation = interpolation, cmap = cmap, aspect = 'equal')
        plt.xticks(range(0, self.num_regions), self.labels, fontsize=7, rotation=90)
        plt.yticks(range(0, self.num_regions), self.labels, fontsize=7)
        cb = plt.colorbar(shrink=0.25)
        cb.set_label('weights', fontsize=14)
        plt.show()
class resting_state():
    def __init__(self, structural_path, functional_path):
        self.sc = structural(structural_path)
        self.fc = functional(functional_path)
        self.pcc = np.corrcoef(self.sc.upper_triangle().ravel(),
                               self.fc.upper_triangle().ravel())[0, 1]

    def run_rww_sim_bif(self, con, G, regime, D, dt, simlen, initconds):
        # Initialise Simulator.
        sim = simulator.Simulator(
            model=MyRWW(**regime),
            connectivity=con,
            coupling=coupling.Scaling(a=np.float(G)),
            integrator=integrators.HeunDeterministic(dt=dt),
            monitors=monitors.TemporalAverage(period=1.)
        )

        # Set initial conditions.
        if initconds:
            if initconds == 'low':
                sim.initial_conditions = np.random.uniform(low=0., high=0.2, size=((1, 1, self.sc.num_regions, 1)))
            elif initconds == 'high':
                sim.initial_conditions = np.random.uniform(low=0.8, high=1.0, size=(1, 1, self.sc.num_regions, 1))

        sim.configure()
        # Lunch simulation
        H = []
        for (t, y), in sim(simulation_length=simlen):
            H.append(sim.model.H.copy())

        H = np.array(H)
        Hmax = np.max(H[14999, :])
        return Hmax

    def setup_sim(self):
        regime = {'a': 270., 'b': 108., 'd': 0.154, 'gamma': 0.641 / 1000, 'w': 1., 'I_o': 0.3}

        # Run G sweep with short runs
        self.Gs = np.arange(0., 3.1, 0.5)

        Hmax_low = np.zeros((len(self.Gs)))
        Hmax_high = np.zeros((len(self.Gs)))
        for iG, G in enumerate(self.Gs):
            Hmax_low[iG] = self.run_rww_sim_bif(self.sc.connectivity, np.float(self.Gs[iG]), regime, 0.001, 0.1, 15000, 'low')
            Hmax_high[iG] = self.run_rww_sim_bif(self.sc.connectivity, np.float(self.Gs[iG]), regime, 0.001, 0.1, 15000, 'high')

    def plot_bifuriction(self):
        # Visualize

        plt.figure(figsize=(10, 8))

        # FC
        plt.plot(self.pcc, '-*', label='FC - FC')
        plt.xlabel('$G_{coupl}$', fontsize=20);
        plt.xticks(np.arange(len(self.Gs)), self.Gs)
        plt.ylabel('PCC', fontsize=20)

        # SC
        plt.plot(self.pcc, '-*g', label='SC - FC')
        plt.xlabel('$G_{coupl}$', fontsize=20);
        plt.xticks(np.arange(len(self.Gs)), self.Gs)
        plt.ylabel('PCC', fontsize=20)
        plt.title('Correlation Diagram', fontsize=20)

        plt.axvline(21, color='k', linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

class mymodel():
    def __init__(self):
        self.rww = models.ReducedWongWang()
        self.W = np.linspace(0.6, 1.05, num=50)


    def local_excitatory(self):
        # Initialize the state-variable S
        self.S = np.linspace(0., 1., num=1000).reshape((1, -1, 1))
        self.W = np.linspace(0.6, 1.05, num=50)
        # Remember: the phase-flow only represents the dynamic behaviour of a disconnected node => SC = 0.
        self.C = self.S * 0.
        # Parameter sweep
        # Fixed Io value
        self.rww.I_o = 0.33

    def phase_flow(self):
        # Visualize phase-flow for different w values
        fig = plt.figure(figsize=(10, 10))
        for iw, w in enumerate(self.W):
            self.rww.w = w
            dS = self.rww.dfun(self.S, self.C)
            plt.plot(self.S.flat, dS.flat, 'k', alpha=0.1)
        plt.plot([0, 0], 'r')
        plt.title('Phase flow for different values of $w$', fontsize=20)
        plt.xlabel('S', fontsize=20);
        plt.xticks(fontsize=20)
        plt.ylabel('dS', fontsize=20);
        plt.yticks(fontsize=20)
        plt.show()

    def phase_plot(self):
        # Visualize phase-flow for different w values
        fig = plt.figure(figsize=(10, 10))
        self.rww.w = self.W
        dS = self.rww.dfun(self.S, self.C)
        plt.plot(self.S.flat, dS.flat, 'k', alpha=0.1)
        plt.plot([0, 0], 'r')
        plt.title('Phase flow for different values of $w$', fontsize=20)
        plt.xlabel('S', fontsize=20);
        plt.xticks(fontsize=20)
        plt.ylabel('dS', fontsize=20);
        plt.yticks(fontsize=20)
        plt.show()

    def external_input(self):
        Io = np.linspace(0.00, 0.42, num=50)
        self.rww.w = 1
        # Plot phase-flow for different Io values
        fig = plt.figure(figsize=(10, 10))
        for i, io in enumerate(Io):
            self.rww.I_o = io
            dS = self.rww.dfun(self.S, self.C)
            plt.plot(self.S.flat, dS.flat, 'k', alpha=0.1)
        plt.plot([0, 0], 'r')
        plt.title('Phase flow for different values of $I_o$', fontsize=20)
        plt.xlabel('S', fontsize=20);
        plt.ylabel('dS', fontsize=20);
        plt.show()



class MyRWW(models.ReducedWongWang):
    def dfun(self, state, coupling, local_coupling=0.0):
        # save the x and H value as attribute on object
        S, = state
        c_0, = coupling
        lc_0 = local_coupling * S
        self.x = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
        self.H = (self.a*self.x - self.b) / (1 - np.exp(-self.d*(self.a*self.x - self.b)))
        # call the default implementation
        return super(MyRWW, self).dfun(state, coupling, local_coupling=local_coupling)

