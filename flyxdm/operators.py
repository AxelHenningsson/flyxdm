"""A collection of (linear) operators that can be combined to yield multiple far-field
X-ray diffraction forward models. Each operator must implement a forward and backwards
(adjoint) pass but is not required to be explicitly represented as a sparse (or dense)
matrix in RAM. This module is meant to be used in conjunction with algebraic results in
scanning-3DXRD which imply that the forward model can be factorized into a sequence of
block diagonal matrices. With these new results it is possible to rewrite existing
reconstruction algorithms in a very RAM efficient and fast way. The paper describing the
factorization method is currently under peer-review.

"""

import astra
import scipy
import numpy as np
import matplotlib.pyplot as plt

class LinearOperator(object):
    """Template for linear operators.
    """

    def __init__(self) -> None:
        """Constructor for the LinearOperator class.

        This constructor initializes an instance of the LinearOperator class.

        Args:
            None

        Returns:
            None

        """
        pass

    def forward(self, x):
        """Forward operation of the linear operator.

        This method performs the forward operation of the linear operator on the input 'x'.

        Args:
            x (:obj:`numpy array`): Input data on which the forward operation is applied.

        Returns:
            :obj:`numpy array` corresponding to the matrix vector products; A @ x.

        """
        pass

    def backward(self, y):
        """Backward operation of the linear operator.

        This method performs the backward (adjoint) operation of the linear operator on the input 'y'.

        Args:
            y (:obj:`numpy array`): Input data on which the backward operation is applied.

        Returns:
            :obj:`numpy array` corresponding to the matrix vector product; A.T @ y.

        """
        pass

    @property
    def nbytes(self):
        """Property to calculate the memory size of the linear operator.

        This property calculates and returns the memory size of the linear operator in bytes.

        Returns:
            int: Memory size of the linear operator in bytes.

        """
        pass

    def print_mem(self):
        """
        Print memory usage information.

        This method prints the memory usage information for the linear operator object.

        Returns:
            None

        """
        print(' '+self.__class__.__name__.ljust(30)  + str( self.nbytes/1e6 ) + ' Mb')


def join( operators: tuple) -> object:
    """Joint a sequence of linear operators - A, B, C...

    The joint operator assumes action on a target object x to yield y as

        y <-- A @ B @ C @ x.

    I.e te last operator in the `operators` tuple (C in this case) is the first to act on
    the target, x.

    Args:
        operators (`tuple` of :obj:`flyxdm.operators.LinearOperator`): The linear operators to join.

    Attributes:
        operators (`tuple` of :obj:`flyxdm.operators.LinearOperator`): The linear operators to join.

    """

    class operator(LinearOperator):

        def __init__(self, operators: tuple) -> None:
            """Construct a joint operator from a sequence of linear operators.

            Args:
                operators (`tuple`): sequence of `flyxdm.operators.LinearOperator`.

            """
            self.operators = operators

        # Override
        def forward(self, x: np.array) -> np.array:
            out = x.copy()
            for O in reversed(operators):
                out = O.forward( out )
            return out

        # Override
        def backward(self, y: np.array) -> np.array:
            out = y.copy()
            for O in operators:
                out = O.backward( out )
            return out

        # Override
        @property
        def nbytes(self):
            out = 0
            for O in operators:
                out += O.nbytes
            return out

        # Override
        def print_mem(self):
            print('--------------------------------------------')
            print('   Operator Name      |      Memory Usage')
            print('--------------------------------------------')
            for O in operators:
                O.print_mem()
            print('--------------------------------------------')
            print(' TOTAL :  '.ljust(31) + str(self.nbytes/1e6) + ' Mb')
            print('--------------------------------------------')

    return operator(operators)


class StrainSampler(LinearOperator):

    def __init__(self, gsino: np.array) -> None:
        """Sample projected strain tensor components to form directional strains.

        This operation corresponds to a multiplication with a hexa-block-diagonal matrix
        S x = [S1 S2 S3 S4 S5 S6] x, where Si is diagonal and x a column-vector. This operation
        is thus compressing the input from 6 to 1 channels.

        Args:
            gsino (:obj:`numpy array`): A sinogram of measured diffraction vectors, shape=(3,m,n).

        """
        kappa_sino  = gsino / (np.linalg.norm(gsino, axis=0)+1e-12)
        kx, ky, kz  = kappa_sino
        (_, sm, sn) = kappa_sino.shape
        self._S = np.zeros( (6, sm, sn) )
        self._S[0] = kx*kx
        self._S[1] = ky*ky
        self._S[2] = kz*kz
        self._S[3] = 2*kx*ky
        self._S[4] = 2*kx*kz
        self._S[5] = 2*ky*kz

    # Override
    def forward(self, hexa_channel_sinogram: np.array) -> np.array:
        """Execute the inner product forward operator that samples projected strain.

        Args:
            hexa_channel_sinogram (:obj:`numpy array`: A hexa-channel sinogram shape=(6,m,n).

        Returns:
            :obj:`numpy array` A mono channel sinogram shape=(1,m,n).
        """
        return np.sum( self._S * hexa_channel_sinogram, axis=0 )

    # Override
    def backward(self, mono_channel_sinogram: np.array) -> np.array:
        """

        Args:
            image (:obj:`numpy array`): A mono channel sinogram shape=(1,m,n).

        Returns:
            :obj:`numpy array` A hexa-channel sinogram shape=(6,m,n).
        """
        return self._S * mono_channel_sinogram

    # Override
    @property
    def nbytes(self):
        return self._S.nbytes


class DiagonalPrecision(LinearOperator):

    def __init__(self, sino_noise: np.array) -> None:
        """Per-equation weights to be used to handle noise in regression.

        This operation corresponds to a diagonal weight matrix. It is useful when the data
        in the equation system is believed to be corrupted with Gaussian independent noise.

        Args:
            sino_noise (:obj:`numpy array`): A sinogram of  per-ray noise standard deviations shape=(m,n).

        Attributes:
            sino_noise_weights (:obj:`numpy array`): The normalized diagonal weights (1 / std). I.e a larger
                weight means better precision. NOTE: the weights are normalized by np.max(sino_noise_weights).

        """
        self.sino_noise_weights = np.divide( 1, sino_noise, out=np.zeros_like(sino_noise), where=sino_noise!=0)
        self.sino_noise_weights = self.sino_noise_weights / np.max(self.sino_noise_weights)

    # Override
    def forward(self, sinogram: np.array) -> np.array:
        """Multiply a multi-channel sinogram by the per-ray precision.

        Args:
            sinogram (:obj:`numpy array`): sinogram over tensor field shape=(k,m,n), where the number of channels, k, is arbitrary.

        Returns:
            :obj:`numpy array`: weighted multi-channel sinogram.

        """
        return self.sino_noise_weights * sinogram

    # Override
    def backward(self, sinogram: np.array) -> np.array:
        """Multiply a multi-channel sinogram by the per-ray precision.

        Args:
            sinogram (:obj:`numpy array`): sinogram over tensor field shape=(k,m,n), where the number of channels, k, is arbitrary.

        Returns:
            :obj:`numpy array`: weighted multi-channel sinogram.

        """
        return self.forward(sinogram)

    # Override
    @property
    def nbytes(self):
        return self.sino_noise_weights.nbytes


class RadialBasisRenderer(LinearOperator):

    def __init__(self, loc: np.array, scale: float, pixel_support: np.array, number_of_channels: np.array, bias: bool=True) -> None:
        """Render a set of equidistant Gaussian basis functions unto a pixel grid.

        This object represents a - change of basis operator - in the sense that it maps a set of Gaussian basis
        coefficients (plus and optional bias) to the set of pixel coefficients used in the piecewise constant
        image grid representation as defined by the pixel_support.

        While the location and scale of the Guassians included in the expansion is arbitrary the RadialBasisRenderer
        will only render these Gaussians unto pixels that have support, as defined by the binary image pixel_support.

        When the number of Gaussians used in the basis expansion is very sparse, a bias term can help to alleviate
        some of the function representation limitations induced by the inherent oscillations present in such Gaussian
        basis expansions. By setting bias=True the last basis function coefficient set refers to the spatially constant
        unity basis function.

        Args:
            loc (:obj:`numpy array`): relative (x,y) coordinates of the basis functions (mode locations), shape=(N, 2).
                The normalized coordinates of the image grid are taken to go from [-1,1] in both dimensions.
            scale (`float`): Standard deviation of Gaussian basis functions. (same for all basis functions)
            pixel_support (:obj:`numpy array`): 2D binary image where 1 indicates that the function is supported,  shape=(m,n).
            number_of_channels (:obj:`numpy array`): Number of tensor components/channels to use. Each basis function renders
                number_of_channels times to each pixel using a different set of basis coefficients for each render.
            bias (bool, optional): Add an additional spatially constant unity basis function to the expansion. The last column
                basis coefficient array refers to this bias term. For instance if x.shape=(9,n) is the basis coefficient array
                then x[:,-1] defines the constant bias of each channel. Defaults to True.

        Attributes:
            loc (:obj:`numpy array`): relative (x,y) coordinates of the basis functions (mode locations).
            scale (`float`): Standard deviation of Gaussian basis functions.
            number_of_channels (:obj:`numpy array`): Number of tensor components/channels to use.
            has_bias (bool, optional): If bias term is present.

        """
        self.loc   = loc
        self.scale = scale
        self.m, self.n = pixel_support.shape
        self.number_of_channels = number_of_channels
        self._set_rbf_rendering_matrix(bias, pixel_support)
        self.has_bias = bias

    def _set_rbf_rendering_matrix(self, bias, pixel_support):
        """Assemble a sparse matrix that renders a set of Gaussian (and an optional bias) to a pixel grid.

        Args:
            bias (bool): If true; an additional basis column is appended to the rendering matrix and taken as the
                spatially constant unity function.
            pixel_support (:obj:`numpy array`): _description_

        Returns:
            scipy.sparse.csr_matrix: The rendering matrix.
        """
        xg, yg = np.linspace(-1, 1, self.m), np.linspace(-1, 1, self.n)
        grid_x, grid_y = np.meshgrid(xg, yg, indexing='ij')  # 2D grid for interpolation
        self.number_of_basis_functions = self.loc.shape[0] + int(bias)
        self._B = np.zeros(( self.m*self.n, self.number_of_basis_functions))
        for col in range(self.loc.shape[0]):
            rbf = np.exp( -0.5*( (grid_x - self.loc[col,0])**2 + (grid_y - self.loc[col,1])**2)/(2 * self.scale * self.scale) ).flatten()
            rbf[rbf < np.exp( -0.5*( 5.0 /(2 * self.scale) ) ) ] = 0 # cutoff for sparsity at 5 * std
            self._B[:,col]  = rbf
            self._B[:,col] *= pixel_support.flatten()
        if bias: self._B[:,-1] = pixel_support.flatten()
        self._B = scipy.sparse.csr_matrix(self._B)

    # Override
    def forward(self, basis_coeffs: np.array) -> np.array:
        """Render the radial basis unto the pixel grid.

        Args:
            basis_coeffs (:obj:`numpy array`): Basis coefficients, shape = (number of channels, number of basis functions)

        Returns:
            :obj:`numpy array`: Pixel image shape=(m,n).

        """
        pixel_image = np.zeros( (self.number_of_channels, self.m, self.n) )
        for i in range(self.number_of_channels):
            pixel_image[i,:,:] = self._B.dot( basis_coeffs[i] ).reshape(self.m, self.n)
        return pixel_image

    # Override
    def backward(self, image: np.array) -> np.array:
        """Back-propagate the pixel grid values unto the coefficients of the radial basis.

        Args:
            image (:obj:`numpy array`): Pixel image shape=(m,n).

        Returns:
            :obj:`numpy array`: Basis coefficients, shape = (number of channels, number of basis functions)

        """
        assert image.shape[1]==self.m
        assert image.shape[2]==self.n
        basis_coeffs = np.zeros( (self.number_of_channels, self.number_of_basis_functions) )
        for i in range(image.shape[0]):
            basis_coeffs[i,:] = self._B.T.dot( image[i,:,:].flatten() )
        return basis_coeffs

    # Override
    @property
    def nbytes(self):
        return self._B.data.nbytes + self._B.indptr.nbytes + self._B.indices.nbytes


class Laue(LinearOperator):

    def __init__(self, hkls: np.array) -> None:
        """Computes diffraction vectors, g, from a sinogram over UB-matrices and a set of integer Miller (h,k,l) indices.

        Args:
            hkls (:obj:`numpy array`): integer Miller (h,k,l) indices, shape=(3,m). Each column in hkls corresponds
                to a single projection view.

        Attr:
            h,k,l (:obj:`numpy array`): integer Miller (h,k,l) indices, shape=(m,1).

        """
        self.h, self.k, self.l = [m.reshape(len(m),1) for m in hkls]

    # Override
    def forward(self, ub_sino: np.array) -> np.array:
        """Construct a sinogram over diffraction vectors.

        Args:
            ub_sino (:obj:`numpy array`): Sinogram over ub matrices shape=(9,m,n). where ub_sino[i*3 : i*3 + 3] is
                corresponds to the i:th column of a 3x3 ub matrix. (The ub matrix represents the reciprocal lattice
                of a crystal such that np.linalg.inv(ub) holds the crystal cell axes as rows.)

        Returns:
            :obj:`numpy array`: A sinogram over diffraction vectors. shape=(3,m,n)

        """
        return ub_sino[0:3,:,:]*self.h  + ub_sino[3:6,:,:]*self.k + ub_sino[6:,:,:]*self.l

    # Override
    def backward(self, gsino: np.array) -> np.array:
        """Backwards maps the 3-channel diffraction vector sinogram to a 9-channel ub sinogram.

        Args:
            gsino (:obj:`numpy array`): A sinogram of measured diffraction vectors, shape=(3,m,n).

        Returns:
            :obj:`numpy array`: A back-projected sinogram over "ub matrices". shape=(9,m,n)

        """
        out = np.zeros((9, gsino.shape[1], gsino.shape[2]))

        out[0,:] = self.h * gsino[0,:,:]
        out[1,:] = self.h * gsino[1,:,:]
        out[2,:] = self.h * gsino[2,:,:]

        out[3,:] = self.k * gsino[0,:,:]
        out[4,:] = self.k * gsino[1,:,:]
        out[5,:] = self.k * gsino[2,:,:]

        out[6,:] = self.l * gsino[0,:,:]
        out[7,:] = self.l * gsino[1,:,:]
        out[8,:] = self.l * gsino[2,:,:]

        return out

class MeanProjector(LinearOperator):

    def __init__(self, density: tuple, number_of_channels: int, det_count: int, angles: np.array, gpu: bool=True) -> None:
        """Projection operator for multi-channel of 2D pixel images (i.e 2D tensor fields or scalar 3D volumes).

        This operator projects volume averaged spatial vector fields such that the formed sinograms hold
        average properties over the ray domains.

        This is a wrapper implementation using the astra lib: https://www.astra-toolbox.com/

        Args:
            image_shape (`tuple`): 2D image shape (number_of_channels, no_x_pixels, no_y_pixels).
            det_count (`int`): Number of detector pixels/scan-positions per projection.
            angles (:obj:`numpy array`): Angles in degrees at which to project the input field at.
            gpu (`bool`, optional): Use GPU unit to accelerate computation with astra. Defaults to True.

        Attr:
            number_of_channels (`int`): Number of expected channels in input images.
            det_count (`int`): Number of detector pixels/scan-positions per projection.
            m, n (`int`): Expected image shape.
            angles (:obj:`numpy array`): Angles in degrees at which to project the input field at.
            gpu (`bool`, optional): Use GPU unit to accelerate computation with astra. Defaults to True.

        """
        assert np.all(density>=0), "density must be >= 0 "
        assert len(density.shape)==2, "density must be a 2D array "
        self.number_of_channels, self.m, self.n = number_of_channels, density.shape[0], density.shape[1]
        assert self.m%2==0 and self.n%2==0, "Please pad the image to make the shape even"
        self.det_count = det_count
        self.angles = angles
        self.gpu = gpu

        if gpu:
            self._vol_geom = astra.creators.create_vol_geom( (self.m, self.n, self.number_of_channels) )
        else:
            self._vol_geom = astra.creators.create_vol_geom( (self.m, self.n) )

        self._rsum, self.sinomask = self._get_mean_weights(density)

    def _get_mean_weights(self, density):
        if self.gpu:
            vol_geom = astra.creators.create_vol_geom( (self.m, self.n, 1) )
            proj_geom = astra.creators.create_proj_geom("parallel3d", 1., 1., 1, self.det_count, np.radians(self.angles))
            im = density.copy().reshape(1, self.m, self.n).astype(float)
            idn, density_sinogram = astra.creators.create_sino3d_gpu( im , proj_geom, vol_geom )
            density_sinogram = density_sinogram[0]
            astra.data3d.delete(idn)
        else:
            vol_geom = astra.creators.create_vol_geom( density.shape )
            proj_geom = astra.creators.create_proj_geom('parallel', 1., self.det_count, np.radians(self.angles))
            pid = astra.creators.create_projector('strip', proj_geom, vol_geom)
            idn, density_sinogram = astra.creators.create_sino( density.astype(float), pid )
            astra.data3d.delete(idn)
        sinomask = density_sinogram > 1e-8*np.min(density)
        _rsum = 1. / ( density_sinogram[sinomask] )
        return _rsum, sinomask

    # Override
    def forward(self, image: np.array) -> np.array:
        """Forward project the multi-channel image into a multi-channel sinogram.

        Args:
            image (:obj:`numpy array`): multi-channel image, shape=(number_of_channels, m, n).

        Returns:
            :obj:`numpy array`: multi-channel sinogram, shape=(number_of_channels, k, l).

        """
        if self.gpu:
            proj_geom = astra.creators.create_proj_geom("parallel3d", 1., 1., self.number_of_channels, self.det_count, np.radians(self.angles))
            idn, sinogram = astra.creators.create_sino3d_gpu( image , proj_geom, self._vol_geom )
            astra.data3d.delete(idn)
        else:
            proj_geom = astra.creators.create_proj_geom('parallel', 1., self.det_count, np.radians(self.angles))
            pid = astra.creators.create_projector('strip', proj_geom, self._vol_geom)
            sinogram = np.zeros((self.number_of_channels, len(self.angles), self.det_count))
            for i in range(self.number_of_channels):
                idn, sinogram[i,:,:] = astra.creators.create_sino( image[i,:,:] , pid )
                astra.data3d.delete(idn)
        sinogram[:,self.sinomask] *= self._rsum
        sinogram[:,~self.sinomask] = 0
        return sinogram

    # Override
    def backward(self, sinogram: np.array) -> np.array:
        """Backwards project a multi-channel sinogram into a multi-channel image.

        Args:
            sinogram (:obj:`numpy array`): multi-channel sinogram, shape=(number_of_channels, k, l).

        Returns:
            :obj:`numpy array`: multi-channel image, shape=(number_of_channels, m, n).

        """
        avg_sino = sinogram.copy()
        avg_sino[:,self.sinomask] *= self._rsum
        avg_sino[:,~self.sinomask] = 0
        if self.gpu:
            proj_geom = astra.creators.create_proj_geom("parallel3d", 1., 1., self.number_of_channels, self.det_count, np.radians(self.angles))
            idn, image = astra.creators.create_backprojection3d_gpu( avg_sino , proj_geom, self._vol_geom )
            astra.data3d.delete(idn)
        else:
            proj_geom = astra.creators.create_proj_geom('parallel', 1., self.det_count, np.radians(self.angles))
            pid = astra.creators.create_projector('strip', proj_geom, self._vol_geom)
            image = np.zeros((self.number_of_channels, self.m, self.n))
            for i in range(self.number_of_channels):
                idn, image[i,:,:] = astra.creators.create_backprojection( avg_sino[i,:,:] , pid )
                astra.data3d.delete(idn)
        return image

    # Override
    @property
    def nbytes(self):
        return self._rsum.nbytes + self.angles.nbytes + self.sinomask.nbytes

    def __del__(self):
        astra.functions.clear()