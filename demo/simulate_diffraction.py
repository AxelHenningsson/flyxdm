"""This is a demo simulation in which far-field diffraction is simulated from a single 2D
slice of alpha-quartz (space group P3223). Diffraction vectors are generated in a
scanning-3DXRD context were each X-ray line integral corresponds to a single measurement of
the underlying strain-orientation field in the crystal slice.
"""

import numpy as np
import matplotlib.pyplot as plt
from xfab import tools
from flyxdm import operators, utils
import os

DATA_PATH = os.path.abspath( os.path.join( __file__, '..', 'data') )

if __name__=='__main__':

    # We consider an alpha-quartz crystal and define a reference crystal
    # strain-orientation state as:
    unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
    U0  = np.load( os.path.join(DATA_PATH, 'U0.npy'))
    B0  = tools.form_b_mat(unit_cell)
    UB0 = U0 @ B0

    # Diffraction will be considered in the Bragg angle range [4,13] degrees,
    # a range which is typically observable in scanning-3DXRD.
    wavelength =  0.2845704100778472
    sthmin     =  np.sin(np.radians(4)) / wavelength
    sthmax     =  np.sin(np.radians(13)) / wavelength
    all_hkls   =  tools.genhkl_all(unit_cell, sthmin, sthmax, sgname='P3221').T

    # Let us define a strain-orientation field representing the grain deformation
    # state and grain support/shape. These fields will later be used to generate
    # per-ray diffraction vectors that encode the crystal state.
    strain    = np.load( os.path.join(DATA_PATH , 'strain.npy') )
    ub_field  = np.load( os.path.join(DATA_PATH , 'ub_field.npy') )
    grain     = np.load( os.path.join(DATA_PATH , 'grain.npy') )

    # We now compute the turntable angles, omega, at which each Miller index
    # h,k,l will result in a diffraction vector, G. Here we select to sample
    # 500 of the resulting diffraction events.
    omega, G, hkls = utils.get_omega_and_Gs(UB0, wavelength, all_hkls)
    index = np.random.permutation( len(omega) )[0:500]
    omega, G, hkls = omega[index], G[:, index], hkls[:, index]

    # To model the per-ray diffraction vector we project the crystal UB-field into a sinogram of
    # average UB states, one for each ray integral.
    ub_projector = operators.MeanProjector(grain, ub_field.shape[0], grain.shape[1]*2,  omega, gpu=True)
    ub_sino      = ub_projector.forward( ub_field )

    # We may now form a sinogram over average diffraction vectors by simply multiplying
    # though the ub-sinogram with the Miller, h,k,l indices, following the Laue equations.
    h, k, l      = [m.reshape(len(m),1) for m in hkls]
    gsino        = ub_sino[0:3,:,:]*h  + ub_sino[3:6,:,:]*k + ub_sino[6:,:,:]*l

    # The corresponding reference sinogram over average diffraction vectors can be formed as
    g0sino = np.zeros( gsino.shape )
    for i,gvec in enumerate( (UB0@hkls).T ): g0sino[:,i,:] = gvec.reshape(3,1)
    g0sino *= ub_projector.sinomask

    # To simulate that the measured data is non-perfect we add Gaussian Noise to the diffraction
    # vectors. We select 10% of the measurements for additional noising, representing outliers.
    gnoise_std = np.zeros_like( gsino, dtype=float ).flatten()
    n = len(gnoise_std)//10
    gnoise_std[0:n] = 1e-2 # outliers get x10 noise standard deviation
    gnoise_std[n:]  = 1e-3
    np.random.shuffle(gnoise_std)
    gnoise_std = ub_projector.sinomask * gnoise_std.reshape(gsino.shape)
    gsino += np.random.normal(loc=0, scale=gnoise_std)

    # Following the ideas of Henningsson, A. & Hendriks, J. (2021). J. Appl. Cryst. 54, 1057-1070.
    # we can form a sinogram over directional scalar strains using the diffraction vectors as
    gTg0  = np.sum(gsino*g0sino, axis=0)
    g0Tg0 = np.sum(g0sino*g0sino, axis=0)
    strainsino  = 1 - np.divide(gTg0, g0Tg0, out=np.ones_like(gTg0), where=g0Tg0!=0)

    # The noise in the directional strain now propagates according to the linear transform
    a  = np.sum(g0sino*(gnoise_std**2)*g0sino, axis=0)
    strain_noise_std = np.sqrt( np.divide(a, g0Tg0**2, out=np.ones_like(gTg0), where=g0Tg0!=0) )

    # We save the resulting data to our data path. Checkout the demo reconstruction for more
    # tutorials on how to utilise matrix decompositions to implement fast diffraction routines
    # in far-field X-ray diffraction.
    np.save(os.path.join( DATA_PATH, 'strainsino.npy' ), strainsino)
    np.save(os.path.join( DATA_PATH, 'gsino.npy' ), gsino)
    np.save(os.path.join( DATA_PATH, 'g0sino.npy' ), g0sino)
    np.save(os.path.join( DATA_PATH, 'omega.npy' ), omega)
    np.save(os.path.join( DATA_PATH, 'gnoise_std.npy' ), gnoise_std)
    np.save(os.path.join( DATA_PATH, 'strain_noise_std.npy' ), strain_noise_std)
    np.save( os.path.join(DATA_PATH, 'hkls.npy'), hkls)
    np.save( os.path.join(DATA_PATH, 'UB0.npy'), UB0)


    # Finally we visualise some of our simulation results
    fig, ax = plt.subplots(3, 3, figsize=(12,6), sharex=True, sharey=True)
    fig.suptitle('Grain UB-field')
    k=0
    for i in range(3):
        for j in range(3):
            s = ub_field[k].copy()
            s[~grain]= np.nan
            im = ax[i,j].imshow(s, cmap='viridis')
            ax[i,j].axis('off')
            plt.colorbar(im, ax=ax[i,j])
            t = '11 21 31 12 22 23 13 23 33'.split(' ')[k]
            ax[i,j].annotate('$UB_{' + t + '}$', (0, 0), fontsize=18)
            k+=1

    fig, ax = plt.subplots(1, 1, figsize=(12,6), sharex=True, sharey=True)
    fig.suptitle('Sinogram over Directional Average Strain')
    s = strainsino.copy()
    s[~ub_projector.sinomask]= np.nan
    im = ax.imshow(s, aspect='auto', cmap='RdBu_r', vmin=-0.0045, vmax=0.0045)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    fig, ax = plt.subplots(1, 3, figsize=(12,6), sharex=True, sharey=True)
    fig.suptitle('Sinogram over Average Diffraction Vectors')
    for i in range(3):
        g = gsino[i].copy()
        g[~ub_projector.sinomask]= np.nan
        im = ax[i].imshow(g, aspect='auto')
        ax[i].axis('off')
        plt.colorbar(im, ax=ax[i])

    fig, ax = plt.subplots(1, 3, figsize=(12,6), sharex=True, sharey=True)
    fig.suptitle('Sinogram over Average Diffraction Vectors')
    for i in range(3):
        g = gsino[i].copy()
        g[~ub_projector.sinomask]= np.nan
        im = ax[i].imshow(g, aspect='auto')
        ax[i].axis('off')
        plt.colorbar(im, ax=ax[i])

    fig,ax = plt.subplots(3, 3, figsize=(8,7), sharey=True )
    fig.suptitle('Euler angle deviation from mean (dgr) and strain tensor (x1e-4)')
    euler, strain = utils.to_euler_and_strain( ub_field, unit_cell, grain )
    for i in range(3):
        s = euler[i].copy()
        s[~grain] = np.nan
        delta = s - np.nanmean(s)
        im = ax[0,i].imshow(delta,  vmin=-0.8, vmax=0.8, cmap='jet')
        ax[0,i].annotate('$\\Delta$ $\\varphi_1$,$\\Delta$ $\\phi$,$\\Delta$ $\\varphi_2$'.split(',')[i],  (0,0), fontsize=20)
        ax[0,i].axis('off')
    fig.colorbar(im, ax=ax.ravel().tolist(), location='top', shrink=0.95, pad=0.05, aspect=60)
    for i in range(3):
        s = strain[i].copy()
        s[~grain] = np.nan
        ax[1,i].imshow(s*1e4, vmin=-25, vmax=25, cmap='RdBu_r')
        ax[1,i].annotate('$\\varepsilon_{11}$ $\\varepsilon_{22}$ $\\varepsilon_{33}$'.split(' ')[i],  (0,0), fontsize=20)
        ax[1,i].axis('off')
    for i in range(3):
        s = strain[i+3].copy()
        s[~grain] = np.nan
        im =ax[2,i].imshow(s*1e4, vmin=-25, vmax=25, cmap='RdBu_r')
        ax[2,i].annotate('$\\varepsilon_{12}$ $\\varepsilon_{13}$ $\\varepsilon_{23}$'.split(' ')[i],  (0,0), fontsize=20)
        ax[2,i].axis('off')
    fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom', shrink=0.95, pad=0.01, aspect=60)

    plt.show()
