import numpy as np

a = np.load("unet.npy")
b = np.load("misa.npy")
# import pdb;pdb.set_trace()
np.testing.assert_allclose(a,b,rtol=1e-2)
