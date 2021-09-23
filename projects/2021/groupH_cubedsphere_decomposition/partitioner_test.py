"Test file to execute the Partitioner"

from fv3gfs.util import Quantity, CubedSphereCommunicator, CubedSpherePartitioner, TilePartitioner, X_DIM, Y_DIM
from fv3gfs.util import boundary as bd
import numpy as np
from fv3gfs.util.testing import DummyComm
layout = (5,3) #Put any layout
total_ranks = layout[0] * layout[1] * 6
rank = 5 #Put any rank on layout

print("layout =",layout)
print("rank = ",rank)
print("Calculations for the left edge")
boundary_list = CubedSpherePartitioner(TilePartitioner(layout))
test_list = boundary_list._left_edge(rank)
print("List of Boundaries:")
print(test_list)
