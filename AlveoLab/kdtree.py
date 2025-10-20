import os

import numpy as np
from pykdtree.kdtree import KDTree as _pyKDTree

# pykdtree uses OpenMP multicore processing to speed itself up. But the tree queries used here
# are too small to significantly do much good. Disable OpenMP by setting this environment
# variable.
os.environ["OMP_NUM_THREADS"] = "1"

class KDTree(_pyKDTree):
    def __init__(self, data, leafsize=16):
        super().__init__(data, leafsize)
        self.dtype = data.dtype

    def query(self, x, k=None, eps=0.0, distance_upper_bound=np.inf):
        x = np.asarray(x, self.dtype)
        target_shape = x.shape[:-1]
        if k is None:
            k = 1
        else:
            target_shape += (k,)

        distances, args = super().query(
            x.reshape((-1, self.ndim)),
            k=k,
            eps=eps,
            distance_upper_bound=distance_upper_bound,
        )
        return distances.reshape(target_shape), args.reshape(target_shape)

    # Everything below is to make this thing pickle safe.
    def __getstate__(self):
        if hasattr(self, "ndim"):
            # pykdtree flattens the data - reshape it.
            data = self.data.reshape((self.n, self.ndim))
        else:
            data = self.data
        return {"data": data, "leafsize": self.leafsize}

    def __setstate__(self, dic):
        self.__init__(**dic)

    __reduce__ = object.__reduce__
    __reduce_ex__ = object.__reduce_ex__