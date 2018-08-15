import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from modelling_resting_state import *

myrestingstate = resting_state('resources/Node7[Public]/resting_state_hands_on/connectivity_HagmannDeco66.zip', 'resources/Node7[Public]/resting_state_hands_on/Hagmann_empFC_avg.npy')
myrestingstate.sc.plotweights()
myrestingstate.sc.plottracts()


model = mymodel()
model.local_excitatory()
model.phase_flow()
model.external_input()

myrestingstate.setup_sim()
myrestingstate.plot_bifuriction()


