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
    ub_field   = np.load(os.path.join( DATA_PATH, 'ub_field.npy' ) )
    gsino      = np.load(os.path.join( DATA_PATH, 'gsino.npy' ))
    g0sino     = np.load(os.path.join( DATA_PATH, 'g0sino.npy' ))
    omega      = np.load(os.path.join( DATA_PATH, 'omega.npy' ))
    noise_std  = np.load(os.path.join( DATA_PATH, 'gnoise_std.npy' ))
    grain      = np.load(os.path.join( DATA_PATH, 'grain.npy' ))
    X          = np.load( os.path.join(DATA_PATH , 'X.npy') )
    Y          = np.load( os.path.join(DATA_PATH , 'Y.npy') )
    hkls       = np.load( os.path.join(DATA_PATH , 'hkls.npy') )
    UB0        = np.load( os.path.join(DATA_PATH , 'UB0.npy') )

    # We define a projector (forming averages on ray paths)
    P  = operators.MeanProjector(grain, 9, grain.shape[1]*2,  omega, gpu=True)

    # We define a basis renderer (rendering a basis coefficent set unto a pixel grid)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing='ij')
    loc    = np.array((xx.flatten(), yy.flatten())).T
    B = operators.RadialBasisRenderer(loc, scale=0.20, pixel_support=grain, number_of_channels=9)

    # We define the weights, to handle noise, in a classical WLSQ sense.
    W = operators.DiagonalPrecision( noise_std )

    # We define the strain sampler, mapping strain tensors to directional strains.
    M = operators.Laue( hkls )

    # Let us now join all operators to form the equation system
    #   A @ x = W @ M @ P @ K @ x = b
    # where b = W @ gsino are noisy weighted measurements.
    A = operators.join( (W, M, P, B) )

    # Now we may reconstruct the ub field by iterative methods, never representing the
    # operator O as a sparse matrix
    b = W.forward(gsino)
    inital_guess=np.zeros( (B.number_of_channels, B.number_of_basis_functions) )
    for i in range(9): inital_guess[i,-1] = UB0.T.flatten()[i] # For faster convergence we can use our initial guess to set the bias.
    x = utils.lsq( A, b, inital_guess, tol=1e-4, maxiter=9*B.number_of_basis_functions, disp=True )
    pixel_reconstruction = B.forward( x )

    # Finally we visualise the reconstructed field and compare to the ground truth strain field.
    euler, strain = utils.to_euler_and_strain( ub_field, [4.926, 4.926, 5.4189, 90., 90., 120.], grain )
    rec_euler, rec_strain = utils.to_euler_and_strain( pixel_reconstruction, [4.926, 4.926, 5.4189, 90., 90., 120.], grain )

    fields = [euler, rec_euler, rec_euler - euler]
    fig, ax = plt.subplots( len(fields), 9, figsize=(19, 7), sharex=True, sharey=True)
    for j, f in enumerate(fields):
        for i in range(3):
            a = ax[j,i]
            s = f[i,:,:].copy()
            s[~grain] = np.nan
            s -= np.nanmean(s)
            if j==2:
                vmin, vmax = -0.8, 0.8
                a.imshow( s, cmap='jet', vmin=vmin, vmax=vmax)
                rmse = np.sqrt( np.sum(f[i,:,:]**2) / np.sum(grain) )
                rmse = np.round(rmse*1e3).astype(int)
                if i==0: a.set_xlabel('RMSE :    ' + str(rmse) + '$ \\times 10^{-3}$', fontsize=18)
                else: a.set_xlabel('           ' + str(rmse) + '$ \\times 10^{-3}$', fontsize=18)
            else:
                a.imshow( s, cmap='jet', vmin=-0.8, vmax=0.8)
            if j==0: a.annotate('$\\Delta$ $\\varphi_1$,$\\Delta$ $\\phi$,$\\Delta$ $\\varphi_2$'.split(',')[i],  (0,0), fontsize=18)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_xaxis().set_ticks([])
            a.get_yaxis().set_ticks([])
    fields = [strain, rec_strain, rec_strain - strain]
    for j, f in enumerate(fields):
        for i in range(3, 9):
            a = ax[j,i]
            s = f[i-3,:,:].copy()
            s[~grain] = np.nan
            if j==2:
                a.imshow( s, cmap='RdBu_r', vmin=-0.0025, vmax=0.0025)
                rmse = np.sqrt( np.sum(f[i-3,:,:]**2) / np.sum(grain) )
                rmse = np.round(rmse*1e5).astype(int)
                if i==0: a.set_xlabel('RMSE :    ' + str(rmse) + '$ \\times 10^{-5}$', fontsize=18)
                else: a.set_xlabel('           ' + str(rmse) + '$ \\times 10^{-5}$', fontsize=18)
            else:
                a.imshow( s, cmap='RdBu_r', vmin=-0.0025, vmax=0.0025)
            t = '$\\varepsilon_{11}$ $\\varepsilon_{22}$ $\\varepsilon_{33}$ $\\varepsilon_{12}$ $\\varepsilon_{13}$ $\\varepsilon_{23}$'
            if j==0: a.annotate(t.split(' ')[i-3],  (0,0), fontsize=20)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_xaxis().set_ticks([])
            a.get_yaxis().set_ticks([])
    ax[0,0].set_ylabel('GT', fontsize=18)
    ax[1,0].set_ylabel('REC', fontsize=18)
    ax[2,0].set_ylabel('RES', fontsize=18)

    plt.show()
