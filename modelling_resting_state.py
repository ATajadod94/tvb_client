_author_ = 'Alireza Tajadod'
_project_ = 'The virtual brain'

## imports
import os
from pathlib import PurePosixPath
import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator import models
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

class resting_state():
    def __init__(self, structural_path, functional_path):
        self.sc = structural(structural_path)
        self.fc = functional(functional_path)
        self.pcc = np.corrcoef(self.sc.upper_triangle().ravel(),
                               self.fc.upper_triangle().ravel())[0, 1]


        #@property
        #def SC(self):
        #    return self._sc.connectivity
        #
        #@property
        #def FC(self):
        #    return self._fc.connectivity

        #@property
        #def pcc(self):
        #    return np.corrcoef(self.__sc.upper_triangle.ravel(),
        #                       self.__fc.upper_triangle.ravel())[0,1]
class mymodel():
    def __init__(self):
        self.rww = models.ReducedWongWang()

    def local_excitatory(self):
        # Initialize the state-variable S
        self.S = np.linspace(0., 1., num=1000).reshape((1, -1, 1))

        # Remember: the phase-flow only represents the dynamic behaviour of a disconnected node => SC = 0.
        self.C = self.S * 0.
        # Parameter sweep
        self.W = np.linspace(0.6, 1.05, num=50)
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

