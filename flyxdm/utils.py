import numpy as np
from scipy.optimize import minimize
import astra
from xfab import tools

def to_euler_and_strain( ub_field, cell, mask ):
    """Convert a pixelated field of crystal UB matrices to a field of Bunge Euler angles and strain tensors.

    Args:
        ub_field (:obj:`numpy array`): field of ub matrices of shape=(9,m,n). the UB matrix at pixel i,j is
            as UB_ij = ub_field[:, i, j].reshape(3,3).T.
        cell (`list`): unit reference cell of the crystal used in strain computation.
        mask (:obj:`numpy array`): Binary array marking the support of the field with unity.

    Returns:
        :obj:`numpy array`: euler, strain of shape=(3,m,n) and shape=(6,m,n) euler[:,i,j]= phi_1, PHI, phi_2 in
            Bunge notation (degrees) and strain[:,i,j] = e11, e22, e33, e12, e13, e23.

    """
    euler  = np.zeros( (3, ub_field.shape[1], ub_field.shape[2]), dtype=float )
    strain = np.zeros( (6, ub_field.shape[1], ub_field.shape[2]), dtype=float )
    for i in range(ub_field.shape[1]):
        for j in range(ub_field.shape[2]):
            if mask[i,j]!=0:
                UB = ub_field[:, i, j].reshape(3,3).T
                U,B = tools.ub_to_u_b(UB)
                phi_1, PHI, phi_2 = tools.u_to_euler(U)
                e11, e12, e13, e22, e23, e33 = tools.b_to_epsilon(B, cell)
                s = np.array([[e11, e12, e13],[e12, e22, e23],[e13, e23, e33]])
                s = U @ s @ U.T
                euler[:,i,j]  = np.degrees( tools.u_to_euler(U) )
                strain[:,i,j] = (s[0,0], s[1,1], s[2,2], s[0,1], s[0,2], s[1,2])
    return euler, strain

def lsq(linear_operator, measurements, initial_guess, tol=1e-8, maxiter=999, disp=False):
    """Solve a linear system of equations by least squares (LSQ).

    Args:
        linear_operator (:obj:`flyxdm.operators.LinearOperator`): The linear operator representing the system matrix.
        measurements (:obj:`numpy array`): Second member of the equation system with shape matching linear_operator.forward()
            output.
        initial_guess (:obj:`numpy array`): Initial solution guess with shape matching linear_operator.backward() input.
        tol (:obj:`float`): Tolerance for termination in scipy.optimize.minimise L-BFGS-B implementation. Defaults to 1e-8.
        maxiter (:obj:`int`): Maximum number of L-BFGS-B iterations. Defaults to 999.
        disp (:obj:`bool`): Print cost function reduction progress. Defaults to False.

    Returns:
        :obj:`numpy array` The LSQ solution to the linear system with shape=initial_guess.shape.

    """
    def func(x):
        res  = linear_operator.forward(x.reshape(*initial_guess.shape)) - measurements
        cost = np.sum(res*res)
        jac  = 2*linear_operator.backward( res )
        return cost, jac.flatten()

    opts = {'maxiter':maxiter, 'disp':disp}
    res = minimize(func, initial_guess.flatten(), tol=1e-8, method='L-BFGS-B', jac=True, options=opts)

    return res.x.reshape(*initial_guess.shape)

def get_omega_and_Gs( UB, wavelength, hkls):
    """Compute rotation angles settings that will lead to diffraction for a crystal and experimental setup.

    The X-ray beam is assumed to be aligned with the np.array([1,0,0]) direction.

    Args:
        UB (:obj:`numpy array`): The crystal UB matrix shape=(3,3)
        wavelength (:obj:`float`): The X-ray beam wavelength in units of angstom.
        hkls (:obj:`numpy array`): Considered Miller indices shape=(3,m)

    Returns:
        `tuple` of :obj:`numpy array` Rotation angles of the crystal at which diffraction will occur shape=(n,), Diffraction
            vectors shape=(3,n) and corresponding Miller indices shape=(3,n). The output is sorted according to rotation
            angle.

    """
    Gs = UB.dot(hkls)
    omega = _laue(Gs, rotation_axis=np.array([0,0,1]), wave_vector=2*np.pi*np.array([1,0,0])/wavelength)
    m  = (omega<180)*(omega>0)
    Gs_selected = np.concatenate( (Gs[:, m[0,:]], Gs[:, m[1,:]]), axis=1)
    omega_selected = np.concatenate( (omega[0, m[0,:]], omega[1, m[1,:]]), axis=0)
    hkls_selected = np.concatenate( (hkls[:, m[0,:]], hkls[:, m[1,:]]), axis=1)

    index = np.argsort(omega_selected)
    omega_selected = omega_selected[index]
    Gs_selected = Gs_selected[:, index]
    hkls_selected = hkls_selected[:, index]

    return omega_selected, Gs_selected, hkls_selected

def _laue(G_0, rotation_axis, wave_vector):
    """Compute rotation angles settings that will lead to diffraction for a crystal and experimental setup.

    Each diffraction vector, G_0, is associated to a set of lattice planes that may diffract once or twice in a [0,180]
    degree rotation around rotation_axis.

    Args:
        G_0 (:obj:`numpy array`):
        rotation_axis (:obj:`numpy array`):
        wave_vector (:obj:`float`):

    Returns:
        :obj:`numpy array` Rotation angles of the crystal at which diffraction will occur shape=(2,m) where np.inf
            represents unfeasible lattice planes.

    """
    rx, ry, rz = rotation_axis
    K = np.array([[0, -rz, ry],
                  [rz, 0, -rx],
                  [-ry, rx, 0]])
    K2 = K.dot(K)
    rho_0_factor = -wave_vector.dot(K2)
    rho_1_factor =  wave_vector.dot(K)
    rho_2_factor =  wave_vector.dot(np.eye(3, 3) + K2)
    rho_0 = rho_0_factor.dot(G_0)
    rho_1 = rho_1_factor.dot(G_0)
    rho_2 = rho_2_factor.dot(G_0) + np.sum((G_0 * G_0), axis=0) / 2.
    denominator = rho_2 - rho_0
    a = np.divide(rho_1, denominator, out=np.full_like(rho_0, np.nan), where=denominator!=0)
    b = np.divide(rho_0 + rho_2, denominator, out=np.full_like(rho_0, np.nan), where=denominator!=0)
    rootval = a**2 - b
    leadingterm = -a
    rootval[rootval<0] = np.inf
    s1 = leadingterm + np.sqrt(rootval)
    s2 = leadingterm - np.sqrt(rootval)
    t1 = 2 * np.arctan(s1)
    t2 = 2 * np.arctan(s2)
    omega = np.vstack((t1,t2))

    return np.degrees( omega )

def single_astra_proj_mat(grain_map, omega):
    """Create a sparse (scalar) projection matrix.

    Args:
        grain_map (:obj:`numpy array`): 2D binary image where the object support is masked.
        omega (:obj:`numpy array`): angles (in degrees) at which to project.

    Returns:
        :obj:`scipy.sparse.csr_matrix` The projection matrix.

    """
    nbr_det_pix = int(2*grain_map.shape[0])
    vol_geom = astra.create_vol_geom( grain_map.shape[0] , grain_map.shape[1]  )
    proj_geom = astra.create_proj_geom( 'parallel', 1.0, nbr_det_pix, np.radians(omega))
    proj_id = astra.create_projector( 'strip', proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    mat = astra.matrix.get(matrix_id)
    astra.projector.delete(proj_id)
    astra.matrix.delete(matrix_id)
    m = grain_map.flatten() > 0
    return mat[:,m]

def _check_operator(operator, x0, samples=50):
    """Check that the forward and backwards pass of an operator are consistent by use of numerical gradients.

    Args:
        operators (`tuple` of :obj:`flyxdm.operators.LinearOperator`): The linear operators to join.
        x0 (:obj:`numpy array`): Input array that matches the forward pass of operator.
        samples (`int`): NUmber of samples. For each sample a random index in x0 is selected for gradient
            evaluation.
    Returns:
        False if the operator.forward call is inconsistent with the operator.backward call otherwise True.

    """
    y = operator.forward( x0 )
    x = (np.random.rand(*x0.shape)-0.5)
    res = operator.forward( x ) - y
    g = 2 * operator.backward( res )
    eps = 2 * 1e-6
    for _ in range(samples):
        index = tuple( ( np.random.rand( len(x0.shape) ) * np.array(x0.shape) ).astype(int) )

        x[index] -= eps
        res = operator.forward( x.copy() ) - y.copy()
        c0 = np.sum(res*res)
        x[index] += 2*eps
        res = operator.forward( x.copy() ) - y.copy()
        c1 = np.sum(res*res)
        x[index] -= eps

        numjac = (c1-c0)/(2*eps)

        if np.abs(g[index]) > 1e-8:
            relative_error = ( numjac - g[index] ) / g[index]
            if relative_error > 0.01: return False
        else:
            if  np.abs(numjac - g[index]) > 1e-5: return False
    return True

