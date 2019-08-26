import sys
sys.path.append('../')
from datasets.datasets import make_hyperbolic_paraboloid_tsom
from KSE.lib.datasets.artificial.curves_topo_changes import create_data

print(create_data(100, 100).shape)