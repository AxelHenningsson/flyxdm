"""In this demo script we bench the RAM usage of an - on the fly - difffraction strain operator and
show how the explicit storage of the sparse system matrix is unfeasible for high-resolution
reconstructions. The low RAM usage achieved with the - on the fly - implementation is possible
thanks to a matrix factorisation of the system matrix that reveals an underlying block partitioned
structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from flyxdm import operators, utils
import os

DATA_PATH = os.path.abspath( os.path.join( __file__, '..', 'data') )


if __name__ == "__main__":

    # We start by loading our previous computed noisy simulation data
    gsino      = np.load(os.path.join( DATA_PATH, 'gsino.npy' ))
    omega      = np.load(os.path.join( DATA_PATH, 'omega.npy' ))
    grain      = np.load(os.path.join( DATA_PATH, 'grain.npy' ))

    factorised_ram = []
    assem_ram = []
    npix = []

    for i in range(1, 5):

        # Upsample the grain resolution and measurements
        gsino_upsampled = np.repeat(gsino, i, axis=2)
        grain_upsampled = np.repeat(np.repeat(grain, i, axis=0), i, axis=1)

        npix.append( np.sum(grain_upsampled) )
        print('number of nonzero pixels in grain is : ', npix[-1])

        # Compute RAM of on the fly operator
        S = operators.StrainSampler( gsino_upsampled )
        P = operators.MeanProjector(grain, 6, grain.shape[1]*2,  omega, gpu=True)
        factorised_ram.append( operators.join( (S, P) ).nbytes / 1e6 )

        # Compute RAM of assembled sparse matrix (floating point values + unsigned integer indices)
        # Note that the assembly of the projection matrix as an explicit sparse matrix is very slow...
        P = utils.single_astra_proj_mat( grain_upsampled, omega )
        number_of_nonzeros = len(P.data)
        tensor_components = 6
        bytes_per_float = 8
        bytes_per_int = 4
        matrix_bytes = tensor_components*( number_of_nonzeros*bytes_per_float + 2*number_of_nonzeros*bytes_per_int )
        assem_ram.append( matrix_bytes / 1e6  )

    # We visualise the results of our benchmark
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rcParams.update({'font.size': 15})

    plt.figure(figsize=(8,4))
    plt.loglog(npix, factorised_ram, 'ko-', label='On The Fly Operator (GPU)', markersize=6)
    plt.loglog(npix, assem_ram, 'k^--', label='Assembled Matrix Operator', markersize=6)
    plt.xlabel('Number of Nonzero Pixels in Image')
    plt.ylabel('Allocated Memory of Operator [Mb]')
    plt.legend()
    plt.grid(which='both', linewidth=0.5)

    plt.show()