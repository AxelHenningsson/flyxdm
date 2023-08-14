"""This is a demo strain reconstruction in which simulated far-field diffraction data is used.
Diffraction vectors were computed in the simulate_diffraction.py script. This demo implements
a fast GPU accelerated solver for intragranular strain, which is made possible by using a sparse
matrix factorisation of the system matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from xfab import tools
from flyxdm import operators, utils
from scipy.optimize import minimize
import os

DATA_PATH = os.path.abspath( os.path.join( __file__, '..', 'data') )

if __name__ == "__main__":

    # We start by loading our previous computed noisy simulation data
    strain     = np.load(os.path.join( DATA_PATH, 'strain.npy' ) )
    strainsino = np.load(os.path.join( DATA_PATH, 'strainsino.npy' ) )
    gsino      = np.load(os.path.join( DATA_PATH, 'gsino.npy' ))
    g0sino     = np.load(os.path.join( DATA_PATH, 'g0sino.npy' ))
    omega      = np.load(os.path.join( DATA_PATH, 'omega.npy' ))
    noise_std  = np.load(os.path.join( DATA_PATH, 'strain_noise_std.npy' ))
    grain      = np.load(os.path.join( DATA_PATH, 'grain.npy' ))
    X          = np.load( os.path.join(DATA_PATH , 'X.npy') )
    Y          = np.load( os.path.join(DATA_PATH , 'Y.npy') )
    print(np.min(X), np.max(X), np.min(Y), np.max(Y))

    # We define a projector (forming averages on ray paths)
    P  = operators.MeanProjector(grain, 6, grain.shape[1]*2,  omega, gpu=True)

    # We define a basis renderer (rendering a basis coefficent set unto a pixel grid)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing='ij')
    loc    = np.array((xx.flatten(), yy.flatten())).T
    B = operators.RadialBasisRenderer(loc, scale=0.16, pixel_support=grain, number_of_channels=6)

    # We define the weights, to handle noise, in a classical WLSQ sense.
    W = operators.DiagonalPrecision( noise_std )

    # We define the strain sampler, mapping strain tensors to directional strains.
    S = operators.StrainSampler( gsino )

    # Let us now join all operators to form the equation system
    #   A @ x = W @ S @ P @ K @ x = b
    # where b = W @ strainsino are noisy weighted measurements.
    A = operators.join( (W, S, P, B) )

    # Now we may reconstruct the strain field by iterative methods, never representing the
    # operator O as a sparse matrix.
    b = W.forward(strainsino)
    inital_guess=np.zeros( (B.number_of_channels, B.number_of_basis_functions) )
    x = utils.lsq( A, b, inital_guess, tol=1e-4, maxiter=6*B.number_of_basis_functions, disp=True )
    pixel_reconstruction = B.forward( x )

    # Finally we visualise the reconstructed field and compare to the ground truth strain field.
    fields = [strain, pixel_reconstruction, pixel_reconstruction - strain]
    fig, ax = plt.subplots( len(fields), 6, figsize=(15, 7), sharex=True, sharey=True)
    for j, f in enumerate(fields):
        for i, a in enumerate(ax[j,:]):
            s = f[i,:,:].copy()
            s[~grain] = np.nan
            if j==2:
                rmse = np.sqrt( np.sum(f[i,:,:]**2) / np.sum(grain) )
                rmse = np.round(rmse*1e5).astype(int)
                if i==0: a.set_xlabel('RMSE :    ' + str(rmse) + '$ \\times 10^{-5}$', fontsize=18)
                else: a.set_xlabel('           ' + str(rmse) + '$ \\times 10^{-5}$', fontsize=18)
            a.imshow(s, cmap='RdBu_r', vmin=-0.0025, vmax=0.0025)
            t = ['$\\varepsilon_{xx}$','$\\varepsilon_{yy}$','$\\varepsilon_{zz}$','$\\varepsilon_{xy}$','$\\varepsilon_{xz}$','$\\varepsilon_{yz}$'][i]
            if j==0: a.annotate(t, (0,0), fontsize=22)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_xaxis().set_ticks([])
            a.get_yaxis().set_ticks([])

    ax[0,0].set_ylabel('GT', fontsize=18)
    ax[1,0].set_ylabel('REC', fontsize=18)
    ax[2,0].set_ylabel('RES', fontsize=18)

    plt.tight_layout()
    plt.show()
