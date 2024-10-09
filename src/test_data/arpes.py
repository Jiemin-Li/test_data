import math
import numpy as np
from scipy.interpolate import CubicHermiteSpline, RegularGridInterpolator
from scipy.ndimage import zoom
import scipy.constants as sci_const
import time
import xarray as xr

# Generic helper functions

def generate1Dpoly(zeros):
    """Generates a 1D polynomial with zero gradient at each point in zeros.

    This is a convenience function that wraps `CubicHermiteSpline` from
    `scipy.interpolates` using `zeroes` as the zero gradient points and
    returns the fitted polynomial.

    Parameters
    ----------
    zeros : ((float, float),(float, float), ....)
        Lists/Tuple of 2 element lists/tuples containing the ('x','y')
        coordinates where the gradient should be zero.

    Returns
    -------
    poly : scipy.interpolate._cubic.CubicHermiteSpline
        A CubicHermiteSpline object that is the calculated polynomial.
    """
    coords = tuple(zip(*zeros))
    poly = CubicHermiteSpline(x=coords[0], y=coords[1],
                              dydx=np.zeros_like(coords[1]))
    return poly


def reduce_to_firstBZ(coords, lattice_constants=(2.5, 3.4)):
    """ Reduces coordinates to the minimal unique hexagonal BZ region.

    Designed to return equivalent $k_{x}$, $k_{y}$ coordinates to the input
    coordinates that are within the minimal unique hexagonal BZ region.
    Assumes that the axis-parallel in-plane primitive translation vector is
    in the +ve $k_{x}$ direction and the +ve out-of-plane axis is $k_{z}$.
    The $k_{z}$ co-ordinate is optional but will be reduced. If the $k_{z}$
    is used then an optional $E_{b}$ axis can also be added that will be
    passed through unchanged.

    Parameters
    ---------
    coords : [float or numpy.array, float or numpy.array
              (, float or numpy.array, float or numpy.array)].
        The $k_{x}$, $k_{y}$ coordinates to reduce to the minimal unique
        hexagonal BZ region. Optionally the $k_{z}$ coordinate (which is
        reduced) and the binding energy (which is passed through unchanged)
        can be added in the order kx, ky, kz, Eb. If inputting numpy.array's
        then all arrays must be of equal length.
    lattice_constants : float or (float, float), optional.
            The in-plane lattice constant or both lattice constants (in-plane,
            out-of-plane) for the hexagonal brillouin zone in Angstroms,
            default is approximately equal to that for graphene (2.5, 3.4).

    Returns
    -------
    reduced : [float or numpy.array, float or numpy.array
               (,float or numpy.array, float or numpy.array)],
        The reduced $k_{x}$, $k_{y}$ coordinates within the minimal unique
        hexagonal BZ region.

    """

    if isinstance(lattice_constants, (float, int)):
        if len(coords) > 2:
            raise ValueError("If the z-coordinate is given the perpendicular"
                             "lattice constant is also required")

        lattice_constants = (lattice_constants,)

    BZ_size = (2 * math.pi / lattice_constants[0]) * (1 / 2)

    # primitive translation vectors for hexagonal BZ
    translation_vectors = [[2 * BZ_size, 0], [BZ_size, BZ_size * np.sqrt(3)]]
    # rotation matrix for -60 deg
    rotation_matrix = np.array([[1 / 2, np.sqrt(3) / 2],
                                [-1 * np.sqrt(3) / 2, 1 / 2]])

    # move to the +ve x, +ve y quadrant of the BZ.
    reduced = [abs(point) for point in coords]

    # move the origin to the bottom left corner of the BZ
    reduced = [abs(reduced[0]) + BZ_size,
               abs(reduced[1]) + BZ_size / np.sqrt(3)]

    # translate to the first BZ
    translation_times = np.ceil(abs(reduced[1]) / translation_vectors[1][1]) - 1
    reduced[1] = reduced[1] % translation_vectors[1][1]
    reduced[0] = reduced[0] - translation_times * translation_vectors[1][0]
    reduced[0] = reduced[0] % translation_vectors[0][0]

    # move origin to the BZ centre again.
    reduced[0] = reduced[0] - BZ_size
    reduced[1] = reduced[1] - BZ_size / np.sqrt(3)

    # move to the +ve x, +ve y quadrant of the BZ.
    reduced = [abs(point) for point in reduced]

    # if outside the minimal unique hexagonal region rotate into it.
    if reduced[0] < np.sqrt(3) * reduced[1]:  # y=mx+c->x=y/m: c=0, m=1/sqrt(3)
        reduced = abs(np.dot(rotation_matrix, reduced))
        reduced = reduced.tolist()
    # mirror coord around the vertical BZ boundary if necessary
    if reduced[0] > BZ_size:
        reduced[0] = 2 * BZ_size - reduced[0]

    if len(coords) > 2:  # if the z coord was given
        reciprocal_constant = (2 * math.pi / lattice_constants[1]) * (1 / 2)
        # shift the z 'origin' to the BZ boundary and into +ve half
        kz = (abs(coords[2]) + reciprocal_constant / 2)
        # translate to the first BZ
        kz = kz % reciprocal_constant
        # shift the z 'origin' back to the BZ centre
        kz = abs(kz - reciprocal_constant / 2)
        reduced.append(kz)
    if len(coords) > 3:  # if Eb coord was given
        reduced.append(coords[3])

    return reduced


def generate_symmetry_lines(symmetry_point_energies, lattice_constant=2.5):
    """Symmetry line polynomials for a set of symmetry point binding energies.

    This function generates a set of polynomials describing the variation in
    energy between the high symmetry points of a hexagonal lattice in 2D
    ($k_{x}$ vs $k_{y}$).

    Parameters
    ----------
    symmetry_point_energies : [float, float, float], in the unit of eV.
        Binding energies of the [0,0], [lattice_constant,
        lattice_constant/sqrt(3)] and [lattice_constant,0] high symmetry
        points.

    lattice_constant : float, optional.
        The lattice constant (in-plane) for the hexagonal brillouin zone in
        Angstroms, default is approximately equal to that for graphene (2.5).

    Returns
    -------
    symmetry_lines : [{'kx':kx(ky) function, 'E':E(ky) function}, {...}].
        A list of two dictionaries (one for each non x-axis parallel high
        symmetry direction) that holds functions mapping $k_{x}$ (under the
        'kx' key) and Binding Energy (under the 'Eb' key) to $k_{y}$
        respectively.

    """

    BZ_size = (2 * math.pi / lattice_constant) * (1 / 2)
    symmetry_points = [[0., 0.],
                       [BZ_size, BZ_size / np.sqrt(3)],
                       [BZ_size, 0.]]

    def distance_function(x_coefficients, kx1, ky1):
        """Generates a function that maps y to distance along symmetry line

        Parameters
        ----------
        x_coefficients : numpy.ndarray
            numpy array containing the polynomial coefficients to map y to x.
        kx1, ky1 : float
            initial x and y components

        Returns
        -------
        return_function : func.
            The function that returns distance along the high symmetry
            direction as a function of ky.
        """

        def return_function(ky):
            """The returned function that maps ky to distance.
            Parameters
            ----------
            ky : float
                The y value for which the distance along the symmetry line needs
                to be computed.

            Returns
            -------
            output : float
                The distance along the symmetry line for the given ky value.
            """

            output = np.sqrt((np.poly1d(x_coefficients)(ky) - kx1) ** 2 +
                             (ky - ky1) ** 2)
            return output

        return return_function

    def energy_function(polynomial_function, distance_func):
        """Binding energy as a function of ky along a high symmetry direction.

        Parameters
        ----------
        polynomial_function : scipy.interpolate._cubic.CubicHermiteSpline
            The 1D polynomial that varies the Energy along the high symmetry
            direction.
        distance_func : func .
            A function generated by `distance_function` that returns distance
            along the high symmetry direction for the given ky value.

        Returns
        -------
        return_function
        """

        def return_function(ky):
            """
            The returned function that maps ky to Energy.

            Parameters
            ----------
            ky : float
                The ky value for which the distance along the symmetry line
                needs to be computed.

            Returns
            -------
            output : float
                The Binding Energy for the given ky value.
            """
            output = polynomial_function(distance_func(ky))

            return output

        return return_function

    # Generate the polynomials along the high symmetry lines.
    symmetry_lines = []
    for i in range(1, 3):
        coef = np.polyfit(*zip((symmetry_points[i - 1][1],
                                symmetry_points[i - 1][0]),
                               (symmetry_points[i][1],
                                symmetry_points[i][0])),
                          1)
        # x as a function of y
        symmetry_lines.append({'x': np.poly1d(coef)})
        # distance along symmetry line as a function of y
        distance = distance_function(coef, symmetry_points[i - 1][1],
                                     symmetry_points[i - 1][0])
        point_i = (0, symmetry_point_energies[i - 1],)
        point_f = (distance(symmetry_points[i][1]), symmetry_point_energies[i])
        # E value as a function of distance along symmetry line
        polynomial = generate1Dpoly([point_i, point_f])
        symmetry_lines[i - 1]['Eb'] = energy_function(polynomial, distance)

    return symmetry_lines


def gaussian(x, centre, width):
    """1D Gaussian function that returns intensity at the x value(s).

    Returns the Intensity of the Gaussian function with a centre at
    'centre' and a width of 'width' at the x value(s)

    Parameters
    ----------
    x : float, int or numpy.ndarray.
        The x value (or values) for which the Gaussian intensity is
        required.
    centre : float or int.
        The centre, or peak position, of the gaussian distribution
    width : float or int.
        The width of the gaussian distribution

    Returns
    -------
    intensity : float, int or numpy.ndarray.
        The Gaussian intensity calculated at the x value (or values).

    """
    norm_amplitude = 1. / (np.sqrt(2. * np.pi) * width)

    return norm_amplitude * np.exp(-(x - centre) ** 2 / (2 * width ** 2))


def lorentzian(x, centre, width):
    """1D Lorentzian function that returns intensity at the x value(s).

    Returns the Intensity of the Lorentzian function with a centre at
    'centre' and a width of 'width' at the x value(s)

    Parameters
    ----------
    x : float, int or numpy.ndarray.
        The x value (or values) for which the Lorentzian intensity is
        required.
    centre : float or int.
        The centre, or peak position, of the gaussian distribution
    width : float or int.
        The width of the gaussian distribution


    Returns
    -------
    intensity : float, int or numpy.ndarray.
        The Lorentzian intensity calculated at the x value (or values).

    """

    return (1 / (2 * np.pi)) * (width / ((x - centre) ** 2 + width ** 2))


def fermi(binding_energy, temperature=300, fermi_energy=0, zero_offset=0):
    """1D Fermi function that returns the intensity at the energy value(s).

    Returns the Fermi function for the given temperature (in K) and Fermi
    energy (in eV) for a given binding energy (in eV). Note: I assume the
    value of 1 for +ve binding energy and shift the zero by
    zero_offset/(1+zero_offset) to leave some detector noise above the
    Fermi level.

    Parameters
    ----------
    binding_energy : float or numpy.ndarray,
        The binding energy (or energies) in eV for which the Fermi function
        intensity should be returned.
    temperature : float,
        The temperature of the sample in K at which the Fermi function should
        be calculated.
    fermi_energy : float,
        The binding energy of the Fermi level (in eV), default is 0.
    zero_offset : float,
        The 'zero' offset to shift the zero values by (in eV), default is 0.

    Returns
    -------
    amplitude : float or numpy.ndarray,
        The Fermi function amplitude at the given binding energy (or energies).

    """

    kb = sci_const.Boltzmann/sci_const.elementary_charge  # Boltzmans constant in eV/K
    amplitude = 1 / (np.exp(-(binding_energy - fermi_energy) / (kb *
                                                                temperature))+1)
    amplitude = (amplitude + zero_offset) / (1 + zero_offset)

    return amplitude


def perpendicular_momentum(photon_energy, parallel_momentum,
                           binding_energy=0.0, inner_potential=15.0,
                           work_function=5.0):
    """Converts photon energies to perpendicular momentum.

    This function converts photon energy (energies) to perpendicular
    momentum(s) using the relationships:

        $A_{\hbar}=\sqrt{2m_{e}}/\hbar$
        $E_{k}= (E_{ph}-E_{b}-\Phi)$
        $\theta=asin(A_{\hbar}*|k_{\parallel}|/E_{k})$
        $k_{\perpendicular}=\sqrt{E_{k}*cos(\theta)+V0}/A_{\hbar}^{2}$

    where: $\hbar$ is the reduced Planck constant, $m_{e}$ is the electron
    mass, $\theta$ is the electron emission angle, $\k_{\parallel}$ and
    $\k_{perpendicular}$ are the surface parallel and perpendicular momentum,
    $E_{ph}$ is the photon energy, $E_{b}$ is the electron binding energy and
    $\Phi$ is the work function.
    Parameters
    ----------
    photon_energy : float or 1D numpy.ndarray,
        The photon energy (or energies) in eV.
    parallel_momentum : float or 1D numpy.ndarray,
        The parallel momentum in the unit of the inverse of angstrom.
    binding_energy : float or 1D numpy.ndarray,
        The binding energy (or energies) in eV.

    inner_potential: float or 1D numpy.ndarray
        The inner potential in eV.
    work_function: float or 1D numpy.ndarray
        The work function in eV.

    Returns
    -------
    $k_z$ : float or 1D numpy.ndarray,
        The perpendicular momentum in the unit of the inverse of angstrom.
    """
    check_data_len = [len(value)
                      if (isinstance(value, (np.ndarray)) and
                          len(value.shape) == 1)
                      else False
                      for value in [inner_value for inner_value in
                                    [parallel_momentum, photon_energy,
                                     binding_energy]
                                    if not isinstance(inner_value,
                                                      (float, int))]]

    if not np.all(check_data_len):
        raise ValueError(f'The input must be an int, float or 1D numpy array!')
    else:
        check_data_len = list(filter(lambda a: a!=1,check_data_len))
        if check_data_len[1:] != check_data_len[:-1]:
            raise ValueError(f'The length of inputs must be the same if '
                             f'their length is larger than 1!')

    # h_bar = sci_const.hbar  # in J.s
    # m_e = sci_const.m_e  # in Kg or J.s^2/m^2 (E=mc^2)
    A_hbar = np.sqrt(2 * sci_const.m_e) / sci_const.hbar  # in J^(-1/2).m^(-1)

    Eph = photon_energy * 1.6E-19  # convert from eV to J
    k_para = parallel_momentum*1E10  # convert from Ang to m
    Eb = binding_energy * 1.6E-19  # convert from eV to J
    V0 = inner_potential * 1.6E-19  # convert from eV to J
    WF = work_function * 1.6E-19  # convert from eV to J

    E_k = (Eph - Eb - WF)  # in J
    theta = np.arcsin(np.where(1 > np.abs(np.abs(k_para) / (A_hbar*np.sqrt(E_k))),
                               (np.abs(k_para) / (A_hbar*np.sqrt(E_k))), 1))
    kz = A_hbar * np.sqrt(E_k * np.cos(theta)**2 + V0)  # in m^(-1)
    kz *= 1E-10  # convert from m^(-1) to Ang^(-1)

    return kz


units_map = {'kx': '$\AA^{-1}$', 'ky': '$\AA^{-1}$', 'kz': '$\AA^{-1}$',
             'Eb': 'eV', 'Eph': 'eV'}


name_map = {'kx': '$k_x$', 'ky': '$k_y$', 'kz': '$k_z$',
            'Eb': '$E_b$', 'Eph': '$E_{ph}$'}


def to_xarray(data, coords):
    """Converts the spectra from numpy array/dict to an xarray
    This converts an ND data array and dictionary containing axes coordinates
    to an xarray.

    Parameters
    ----------
    data : numpy.ndarray.
        A numpy.ndarray holding the given N-D spectra.
    coords : {'kx': numpy.ndarray, 'ky': numpy.ndarray,
              'kz': numpy.ndarray, 'Eb': numpy.ndarray}.
        The constant value, or range of values, for each of the potential
        spectral axes ($k_{x}$,$k_{y}$,$k_{z}$ and $E_{b}$)

    Returns
    -------
    output : xarray.DataArray, default.
        An xarray.DataArray containing the nd array as well as coordinate
        information, and long-names and unit information for the dataarray
        and coordinates.
    """
    new_axes = tuple(i for i, v in enumerate(coords.values()) if len(v) == 1)
    extended_data = np.expand_dims(data, new_axes)
    output = xr.DataArray(data=extended_data,
                          dims=[k for k in coords.keys()],
                          coords=coords,
                          attrs={'long_name': 'Intensity',
                                 'units': 'detector counts'})

    for coord in coords.keys():
        output[coord].attrs['units'] = units_map[coord]
        output[coord].attrs['long_name'] = name_map[coord]

    return output


class Band:
    """Holds information associated with generating specific electron bands.

    This class holds attributes that allow for calculation of a simulated
    electron band. It assumes a hexagonal crystal structure with the shorter
    primitive translation vector along the $k_{x}$ axis.

    NOTE: These bands take a while to initialize (~ 1 minute) as we generate
        a 4D intensity spectra and then fit a 4D function to this. This is
        done at initialization so that calls to self.spectra() can return
        large, high resolution N-D spectra quickly during use.

    Attributes
    ----------
    symmetry_point_energies : [[float,float,float], [float,float,float]]
            A two element list of 3 element lists providing the binding
            energies for each of the $k_{z}$ direction high symmetry planes.
            Each list has binding energies (in eV) for each of the high
            symmetry points in each $k_{z}$ high symmetry plane. These
            positions (in-plane coordinates) are:
                [0,0],
                [(in-plane lattice constant),
                 (in_plane lattice constant)/sqrt(3)]
                [(in-plane lattice constant),0]
    lattice_constants : (float, float), optional.
        The in-plane lattice constants (in-plane, out-of-plane) for the
        hexagonal brillouin zone in Angstroms, default is approximately
        equal to that for graphene (2.5, 3.4).
    symmetry_lines : [{'kx':kx(ky) function, 'E':E(ky) function}, {...}].
        A list of two dictionaries (one for each non x-axis parallel high
        symmetry direction) that holds functions mapping $k_{x}$ (under the
        'kx' key) and Binding Energy (under the 'Eb' key) to $k_{y}$
        respectively.

    Methods
    -------
    energy : self.energy(kx, ky, kz)
        Returns the binding energy of the band for the inputted kx, ky, and kz
        values.
    spectra : self.spectra(ranges, noise=0.1, temperature=300)
        Returns an N-D ARPES spectra of the band.

    """

    def __init__(self, symmetry_point_energies,
                 lattice_constants=(2.5, 3.4),
                 g_width=0.4, l_width=0.3):
        """Initializes the Band class.

        Parameters
        ----------
        symmetry_point_energies : [[float,float,float], [float,float,float]]
            A two element list of 3 element lists providing the binding
            energies for each of the $k_{z}$ direction high symmetry planes.
            Each list has binding energies (in eV) for each of the high
            symmetry points in each $k_{z}$ high symmetry plane. These
            positions (in-plane coordinates) are:
                [0,0],
                [(in-plane lattice constant),
                 (in_plane lattice constant)/sqrt(3)]
                [(in-plane lattice constant),0]
        lattice_constants : (float, float), optional.
            The in-plane lattice constants (in-plane, out-of-plane) for the
            hexagonal brillouin zone in Angstroms, default is approximately
            equal to that for graphene (2.5, 3.4).
        g_width, l_width : float, optional.
            The widths of the gaussian (g_width) and lorentzian(l_width)
            broadening of the spectra (in eV) returned by self.spectra(...).
        """

        self.symmetry_point_energies = symmetry_point_energies
        self.lattice_constants = lattice_constants
        symmetry_lines = [
            generate_symmetry_lines(energies,
                                    lattice_constant=lattice_constants[0])
            for energies in symmetry_point_energies]
        self.symmetry_lines = symmetry_lines

        BZ_x = (2 * math.pi / lattice_constants[0]) * (1 / 2)
        BZ_y = 2 * BZ_x / np.sqrt(3)
        BZ_z = (2 * math.pi / lattice_constants[1]) * (1 / 2)
        ranges = {'kx': [0, BZ_x + 0.3, 25], 'ky': [0, BZ_y + 0.3, 25],
                  'kz': [0, BZ_z + 0.3, 25], 'Eb': [12, -0.5, 25]}
        self._interpolation = self._generate_interpolation(ranges,
                                                           g_width=g_width,
                                                           l_width=l_width)

    def energy(self, kx, ky, kz):
        """
        Returns the binding energy in eV for the $k_{x}$, $k_{y}$, $k_{z}$
        values.

        Used to provide the energy of the band at the given momentum
        co-ordinates.

        Parameters
        ----------
        kx, ky, kz : float, float, float.
            The kx, ky, and kz values for which the energy is required.

        Returns
        -------
        Eb : float, in the unit of eV.
            The binding energy for the given kx, ky, kz value.
        """

        reciprocal_constant = 2 * math.pi / self.lattice_constants[1]

        kz_symm_points = [0, reciprocal_constant / 2]

        # shift the z 'origin' to the BZ boundary and into +ve half
        kz = (abs(kz) + reciprocal_constant / 2)
        # translate to the first BZ
        kz = kz % reciprocal_constant
        # shift the z 'origin' back to the BZ centre
        kz = abs(kz - reciprocal_constant / 2)
        # reduce the in-plane constants to the first BZ.
        reduced = reduce_to_firstBZ([kx, ky],
                                    lattice_constants=self.lattice_constants[0])

        points = []
        # for the 2 kz high symmetry points generate an energy
        for kz_symm, in_plane in zip(kz_symm_points, self.symmetry_lines):
            # Generate polynomials parallel to the kx axis for the given ky
            point_i = (in_plane[0]['x'](reduced[1]),
                       in_plane[0]['Eb'](reduced[1]).astype(float))
            point_f = (in_plane[1]['x'](reduced[1]),
                       in_plane[1]['Eb'](reduced[1]).astype(float))
            if point_i[0] >= point_f[0]:  # solves an edge case
                points.append([kz_symm, float(in_plane[0]['Eb'](reduced[1]))])
            else:
                in_plane_polynomial = generate1Dpoly([point_i, point_f])
                in_plane_Eb = float(in_plane_polynomial(reduced[0]))
                points.append([kz_symm, in_plane_Eb])

        polynomial = generate1Dpoly(points)
        Eb = float(polynomial(kz))

        return Eb

    def spectra(self, ranges, noise=0.04, temperature=300,
                work_function=5, default_Eph=45, max_angle=90,
                as_xarray=True):
        """ Returns an N-D spectra for the band.

        Generates a spectra for the band based on the input from 'ranges'
        and with random noise at the level given by 'noise' and the
        temperature given by 'temperature (which is used to apply the
        intensity drop across the Fermi level).

        Parameter
        ---------
        ranges : {'kx': float or (start, stop, num_steps),
                  'ky': float or (start, stop, num_steps),
                  'kz' or 'Eph': float or (start, stop, num_steps),
                  'Eb': float or (start, stop, num_steps)}
            A Dictionary providing the constant value, if float, or a
            (start, stop num_steps) tuple, if a range of values is required,
            for each of the potential axes of an ARPES spectra. momentum
            values are in inverse Angstroms and the energies are in eV. If
            using photon energy ('Eph') instead of kz then the function
            perpendicular_momentum is used to make the conversion.
        noise : float, optional.
            The noise level for the returned spectra.
        temperature : float, optional.
            The temperature (in K) of the sample (in K) used to generate the
            intensity drop across the Fermi level.
        work_function : float, optional.
            work function in eV used to determine the photo-emission horizon.
        default_Eph : float, optional.
            The photon energy (in eV) to use in the determination of the
            photoemission horizon if it isn't included in ranges.
        max_angle : float, optional.
            The maximum emission angle, in degrees, to use in the determination
            of the photoemission horizon.
        as_xarray : Bool, optional.
            Indicates if the returned spectra should be provided as an xarray or
            as a numpy array and coordinates dictionary.

        Returns
        -------
        xarray : xarray.DataArray, default.
            An xarray.DataArray containing the nd array as well as coordinate
            information, and long-names and unit information for the dataarray
            and coordinates. If the `as_xarray` kwarg is `False` then the
            optional intensity and axes_coords numpy arrays described below
            are returned.
        intensity : numpy.ndarray, optional.
            A numpy.ndarray holding the given N-D spectra.
        axes_coords : {'kx': numpy.ndarray, 'ky': numpy.ndarray,
                       'kz': numpy.ndarray, 'Eb': numpy.ndarray}, optional.
            The constant value, or range of values, for each of the potential
            spectral axes ($k_{x}$,$k_{y}$,$k_{z}$ and $E_{b}$)
        """

        axes = [axis for axis, value in ranges.items()
                if not isinstance(value, (int, float))]
        shape = [ranges[axis][2] for axis in axes]
        axes_coords = {axis: (np.array([float(value)])
                              if isinstance(value, (int, float))
                              else np.linspace(*value))
                       for axis, value in ranges.items()}

        # This next bit is required to deal with arbitrary spectra dimensions
        values = {axis: array.flatten()  # value lists for each spectra point
                  for axis, array in zip(axes_coords.keys(),
                                         np.meshgrid(*axes_coords.values()))}
        k_para = np.sqrt(np.square(values['kx']) + np.square(values['ky']))
        if 'Eph' in values.keys():
            E_kin = values['Eph'] - values['Eb'] - work_function
            values['kz'] = perpendicular_momentum(photon_energy=values['Eph'],
                                                  parallel_momentum=k_para,
                                                  binding_energy=values['Eb'])
            _ = values.pop('Eph')  # remove the converted Eph values
        else:
            E_kin = default_Eph - values['Eb'] - work_function

        k_para_max = np.sin(np.radians(max_angle)) * 0.5123 * np.sqrt(E_kin)
        k_para_max = np.where(3.6 > k_para_max, k_para_max, 3.6)

        coords = [[x, y, z, E] for x, y, z, E in zip(values['kx'], values['ky'],
                                                     values['kz'], values['Eb'])
                  ]
        coords = [reduce_to_firstBZ(coord,
                                    lattice_constants=self.lattice_constants)
                  for coord in coords]

        # Intensity interpolation and drop off with increased angle
        intensity = self._interpolation(coords) * gaussian(k_para, 0,
                                                           k_para_max / 2)
        # add noise with Fermi drop-off.
        intensity += (noise * np.random.rand(*intensity.shape) *
                      fermi(values['Eb'], zero_offset=0.2,
                            temperature=temperature))
        # add k parallel horizon.
        intensity = np.where(k_para <= k_para_max, intensity,
                             noise * 0.2 * np.random.rand(*intensity.shape))
        # reshape from 1D to spectra shape
        intensity = intensity.reshape(*shape)

        if as_xarray:
            return to_xarray(intensity, axes_coords)
        else:
            return intensity, axes_coords

    def _generate_interpolation(self, ranges, g_width=0.4, l_width=0.3):
        """Returns the interpolation function used for spectra calculations.

        Run at instantiation time only, this returns the interpolation
        function that is used to quickly generate spectra, via self.spectra(),
        during use.

        Parameter
        ---------
        ranges : {'kx': float or (start, stop, num_steps),
                  'ky': float or (start, stop, num_steps),
                  'kz': float or (start, stop, num_steps),
                  'Eb': float or (start, stop, num_steps)}
            A Dictionary providing the constant value, if float, or a
            (start, stop num_steps) tuple, if a range of values is required,
            for each of the potential axes of an ARPES spectra. momentum
            values are in inverse Angstroms and the energies are in eV.

        g_width, l_width : float, optional.
            The widths (in eV) of the gaussian (g_width) and lorentzian(l_width)
            broadening of the spectra returned by self.spectra(...).

        Returns
        -------
        interp : scipy.interpolate.RegularGridInterpolator
            Can be called to provide the intensity at a set of ($k_{x}$,
            $k_{y}$, $k_{z}$ and $E_{b}$) co-ordinates
        """

        axes = [axis for axis, value in ranges.items()
                if not isinstance(value, (int, float))]
        shape = [ranges[axis][2] for axis in axes]
        axes_coords = {axis: (np.array([float(value)])
                              if isinstance(value, (int, float))
                              else np.linspace(*value))
                       for axis, value in ranges.items()}
        # This next bit is required to deal with arbitrary spectra dimensions
        values = {axis: array.flatten()  # value lists for each spectra point
                  for axis, array in zip(axes_coords.keys(),
                                         np.meshgrid(*axes_coords.values()))}

        Eband = np.array([self.energy(kx, ky, kz)
                          for kx, ky, kz in zip(values['kx'], values['ky'],
                                                values['kz'])])
        intensity = self._intensity(values['kx'], values['Eb'], Eband,
                                    g_width=g_width, l_width=l_width)

        spectra = intensity.reshape(*shape)

        interp = RegularGridInterpolator((axes_coords['kx'],
                                          axes_coords['ky'], axes_coords['kz'],
                                          axes_coords['Eb']),
                                         spectra)

        return interp

    def _intensity(self, kx, Eb, Eband, g_width=0.4, l_width=0.3):
        """Return the Intensity at the kx, ky, kz, Eb point

        NOTE: due to the way that we step through ky and kx in
        self._generate_interpolation we are only worried about the kx value. The
        other values are taken care of in the Eband parameter

        Parameters
        ----------
        kx, Eb, Eband: float or numpy.ndarray.
            Value(s) of the momentum (in inverse Angstroms), binding energy
            (in eV) and band energy (in eV) for which to calculate the
            spectral intensity.
        g_width, l_width : float, optional.
            The widths (in eV) of the gaussian (g_width) and lorentzian(l_width)
            broadening of the spectra.

        Returns
        -------
        intensity : np.array.
            Returns the intensity for a range of ($k_{x}$, $k_{y}$, $k_{z}$
            and $E_{b}$) co-ordinates.
        """
        # width increase with increasing band energy
        added_widths = np.array([0.05 * abs(eband) for eband in Eband])
        intensity = np.zeros(*kx.shape)
        # The Gaussian, Lorentzian and Fermi broadening Intensity
        intensity += (gaussian(Eb, Eband, added_widths + g_width) *
                      lorentzian(Eb, Eband, added_widths + l_width) *
                      fermi(Eb))

        return intensity


class Bands:
    """A sub-class that holds information about a set of bands.

    Parameters
    ----------
    symmetry_energies : {'str':[[float,float,float],[float,float,float]],...}
        A Dictionary mapping 'band names' to a two element list of 3 element
        lists providing the binding energies for each of the $k_{z}$ direction
        high symmetry planes for each band. Each list has binding energies
        (in eV) for each of the high symmetry points in each $k_{z}$ high
        symmetry plane. These positions (in-plane coordinates) are:
            [0,0],
            [(in-plane lattice constant),
             (in_plane lattice constant)/sqrt(3)]
            [(in-plane lattice constant),0]
        NOTE: It is recommended, but not required, that the 'band names' follow
        the structure 'bandx' where x is an integer counting number.

    lattice_constants : float, optional.
            The lattice constants (in-plane, out-of-plane) for the hexagonal
            brillouin zone in Angstroms, default is approximately equal to that
            for graphene (2.5, 3.4).
    g_width, l_width : float or [float, ...], optional.
            The widths (in eV) of the gaussian (g_width) and lorentzian(l_width)
            broadening of the spectra. If a list is given then it must
            have the same length as there are elements in symmetry_energies.

    Attributes
    ----------
    many : Band class objects
        Band class objects for each of the bands provided in symmetry_energies
        with the attribute name being the 'key' from this dictionary.
    """

    def __init__(self, symmetry_energies, lattice_constants,
                 g_width=0.4, l_width=0.3):

        if isinstance(g_width, (int, float)):  # create a list
            g_width = [g_width] * len(symmetry_energies)

        if isinstance(l_width, (int, float)):  # create a list
            l_width = [l_width] * len(symmetry_energies)

        for width in [g_width, l_width]:
            if not isinstance(width, list):
                raise ValueError(f'During initialization of a `Bands` class '
                                 f'expected `g_width` and `l_width` to be int,'
                                 f' float, or list but instead got: '
                                 f'{type(g_width)=} and {type(l_width)=}')
            elif len(width) != len(symmetry_energies):
                raise ValueError(f'During initialization of a `Bands` class '
                                 f'expected `g_width` and `l_width` to have '
                                 f'the same length as `symmetry_point_energies'
                                 f' but instead got {len(g_width)=}, '
                                 f'{len(l_width)=}, and '
                                 f'{len(symmetry_energies)=}')

        num_bands = len(symmetry_energies)
        for i, (band, point_energies) in enumerate(symmetry_energies.items()):
            start_time = time.time()
            print(f'Working on {band} which is {i + 1} of {num_bands}')
            setattr(self, band,
                    Band(symmetry_point_energies=point_energies,
                         lattice_constants=lattice_constants,
                         g_width=g_width[i], l_width=l_width[i]))
            band_time = (time.time() - start_time)
            print(f'Time for {band} was {round(band_time)} s')
            print(f'Remaining time estimate: {round(band_time * 
                                                    (num_bands - i - 1))} s')


default_symmetry_energies = {'band1': [[-2, 3, 2], [-3, 7, 6]],
                             'band2': [[1, 9, 10], [2, 10, 6]],
                             'band3': [[10, -1, 0], [10, 2, 3]],
                             'band4': [[5.5, 2, 4], [6, 2, 0]]}


class Arpes:
    """Holds simulated ARPES data and provides spectra extraction methods.

    The aim of this class is to hold together various parameters and methods
    that allow for a generic simulated 'ARPES' spectra to be generated. It's
    main function is to use the `detector_image` method to simulate the
    output of a detector given the number of output parameters that can be
    changed. It does also provide, through the `self.spectra` method, the
    ability to generate N-D 'spectra' over the kx, ky, kz (Eph), Eb volume
    of electron momentum space. This can take longer than the ` detector_image`
    method to return but will provide a smoother spectra and extends it from
    2D to up to 4D.

    NOTE: These class objects take a while to initialize (~ 1 minute per band)
        as we generate a 4D intensity spectra and then fit a 4D function to
        this. This is done at initialization so that calls to self.spectra()
        and/or self.detector_image) can return large, high resolution N-D
        spectra quickly during use.

    Attributes
    ----------
    bands : Bands class object.
        A Bands class object that stores the information for each of the
        bands in the spectra.

    Methods
    -------
    spectra : self.spectra(ranges, noise=0.1, temperature=300)
        Returns a simulated ND spectra as defined by ranges.

    detector_image : self.detector_image(Eb=0., ky=None, Eph=40, T=300,
                                         noise=0.015)
        Returns a simulated detector image for the given input parameters.
    """

    def __init__(self, symmetry_energies=default_symmetry_energies,
                 lattice_constants=(2.5, 3.4), g_width=0.3, l_width=0.3):
        """The initialization method for the ArpesData class.

        Parameters
        ----------
        symmetry_energies : {'str':[[float,float,float],[float,float,float]],..}
            A Dictionary mapping 'band names' to a two element list of 3 element
            lists providing the binding energies for each of the $k_{z}$
            direction high symmetry planes for each band. Each list has binding
            energies (in eV) for each of the high symmetry points in each
            $k_{z}$ high symmetry plane. These positions (in-plane coordinates)
            are:
                [0,0],
                [(in-plane lattice constant),
                 (in_plane lattice constant)/sqrt(3)]
                [(in-plane lattice constant),0]
            NOTE: It is recommended, but not required, that the 'band names'
            follow the structure 'bandx' where x is an integer counting number.

        lattice_constants : float, optional.
            The lattice constants (in-plane, out-of-plane) for the hexagonal
            brillouin zone in Angstroms, default is approximately equal to that
            for graphene (2.5, 3.4).
        g_width, l_width : float or [float, ...], optional.
            The widths of the gaussian (g_width) and lorentzian(l_width)
            broadening of the spectra (in eV). If a list is given then it must
            have the same length as there are elements in symmetry_energies.
        """
        self.bands = Bands(symmetry_energies=symmetry_energies,
                           lattice_constants=lattice_constants,
                           g_width=g_width, l_width=l_width)

    def spectra(self, ranges, noise=0.04, temperature=300, default_Eph=45,
                as_xarray=True):
        """ Returns an N-D spectra for the dataset.

        Generates a spectra for each band based on the input from 'ranges'
        and with random noise at the level given by 'noise' and the
        temperature given by 'temperature (which is used to apply the
        intensity drop across the Fermi level). Sums all 'band' spectra
        together and returns the result.

        Parameters
        ----------
        ranges : {'kx': float or (start, stop, num_steps),
                  'ky': float or (start, stop, num_steps),
                  'kz' or 'Eph': float or (start, stop, num_steps),
                  'Eb': float or (start, stop, num_steps)}
            A Dictionary providing the constant value, if float, or a
            (start, stop num_steps) tuple, if a range of values is required,
            for each of the potential axes of an ARPES spectra. momentum
            values are in inverse Angstroms and the energies are in eV. If
            using photon energy ('Eph') instead of kz then the function
            perpendicular_momentum is used to make the conversion.
        noise : float, optional.
            The noise level for the returned spectra.
        temperature : float, optional.
            The temperature (in K) of the sample (in K) used to generate the
            intensity drop across the Fermi level.
        default_Eph : float, optional.
            The photon energy (in eV) to use in the determination of the
            photoemission horizon if it isn't included in ranges.
        as_xarray : Bool, optional.
            Indicates if the returned spectra should be provided as an xarray
            or as a numpy array and coordinates dictionary.

        Returns
        -------
        xarray : xarray.DataArray, default.
            An xarray.DataArray containing the nd array as well as coordinate
            information, and long-names and unit information for the dataarray
            and coordinates. If the `as_xarray` kwarg is `False` then the
            optional intensity and axes_coords numpy arrays described below
            are returned.
        intensity : numpy.ndarray, optional.
            A numpy.ndarray holding the given N-D spectra.
        axes_coords, optional : {'kx': numpy.ndarray, 'ky': numpy.ndarray,
                       'kz': numpy.ndarray, 'Eb': numpy.ndarray}.
            The constant value, or range of values, for each of the potential
            spectral axes ($k_{x}$,$k_{y}$,$k_{z}$ and $E_{b}$)
        """

        band_names = [x for x in self.bands.__dir__()
                      if not x.startswith(('_',))]

        band_attr = getattr(self.bands, band_names[0])
        intensity, axes_coords = band_attr.spectra(ranges=ranges, noise=noise,
                                                   temperature=temperature,
                                                   as_xarray=False)

        for i in range(1, len(band_names)):
            band_attr = getattr(self.bands, band_names[i])
            temp, _ = band_attr.spectra(ranges=ranges, noise=noise,
                                        temperature=temperature,
                                        default_Eph=default_Eph,
                                        as_xarray=False)
            intensity = intensity + temp

        intensity /= len(band_names)

        if as_xarray:
            return to_xarray(intensity, axes_coords)
        else:
            return intensity, axes_coords

    def detector_image(self, Eb=0., ky=None, Eph=40, T=300, noise=0.015,
                       aspect_ratio='1:1'):
        """Returns a 'detector image' for the given operating parameters.

        Uses the `self.spectra` method to simulate detector images, assuming
        a momentum microscope energy analyzer (i.e. can return either
        Eb vs kx and kx vs ky images). In order to reduce the time taken
        to generate the spectra it calls `self.spectra` with a small number
        of points along each required axis (currently 90) and then uses
        `scipy.ndimage.zoom`, with an order of 3, to increase the pixel count.
        This reduces the return time from minutes to ~ 250ms which allows this
        method to be used in a 'simulated beamline' with an artificially fast
        acquisition time. Spectra are however less 'smooth' as a result, for
        cases where the increased return time is not an issue use
        `self.spectra` instead.

        NOTES: Assumes a camera resolution of 1080x720

        ToDo: Consider adding a 'clip' kwarg that simulates over-exposing
        the detector.

        Parameters
        ----------
        Eb : float or [float, float], optional.
            Binding energy to be used, in eV. If 'ky' is provided then the
            image returned is kx vs Eb, otherwise it is kx vs ky. If 'ky'
            is given the 'Eb' can optionally be provided as a start/ stop
            energy tuple/list otherwise -12 to 0.5 eV energy range is used.
        Eph : float, optional.
            Photon energy used for the measurement, in eV.
        ky : float, optional.
            ky direction momentum value to be used, if not given, or `None`,
            then a kx vs Eb plot is returned. See Eb description for details
            of how to specify the energy range for the kx vs Eb plot.
        T : float, optional.
            Sample temperature, in K, for the measurement, which adjusts the
            Fermi-level width of the returned data.
        noise : float, optional.
            The signal to noise ratio for the plot, default is 0.015. This
            can be adjusted to simulate different 'exposure times' of the
            detector.
        aspect_ratio : str
            A string indicating the aspect ratio, can have values '1:1',
            '16:9' or '16:10'.
        """

        if aspect_ratio == '16:9':
            resolution = [1080, 1920]  # can be updated to any 16x9 resolution
            added_points = [20, 35]
            initial_resolution = [90, 120]
        elif aspect_ratio == '16:10':
            resolution = [1200, 1920]
            added_points = [14, 24]
            initial_resolution = [80, 100]
        elif aspect_ratio == '1:1':
            resolution = [450, 450]
            added_points = [7, 0]
            initial_resolution = [90, 76]
        else:
            raise ValueError(f'In a call to an ARPES.detector_image method the'
                             f' aspect_ratio ({aspect_ratio}) was not one of '
                             f' "16:9", "16:10" or "1:1"')

        if isinstance(ky, (float, int)):  # if a constant ky image is requested.
            # create non-used detector regions with k range>k horizon(2).
            ranges = {'kx': [-3.8, 3.8, initial_resolution[0]], 'ky': ky,
                      'Eph': Eph}
            if isinstance(Eb, (list, tuple)):
                ranges['Eb'] = [*Eb, initial_resolution[1]] # start finish tuple
            else:
                ranges['Eb'] = [12, -0.5, initial_resolution[1]]
            # generate the left/right detector region not without spectra.
            added_range = noise * 0.2 * np.random.rand(initial_resolution[0],
                                                       added_points[0]) / 2

        else:  # if a constant Eb image is requested
            # create non-used detector regions with k range>k horizon(2).
            ranges = {'kx': [-3.8, 3.8, initial_resolution[0]],
                      'ky': [-3.8, 3.8, initial_resolution[0]],
                      'Eb': Eb, 'Eph': Eph}
            # generate the left/right detector region not without spectra.
            if added_points[0]:
                added_range = noise * 0.2 * np.random.rand(initial_resolution[0]
                                                           ,added_points[1]) / 2

        # generate the spectra.
        image, axes_coords = self.spectra(ranges, temperature=T,
                                          noise=noise, as_xarray=False)
        # add the left/right regions
        if added_points[0]:
            image = np.hstack((added_range, image, added_range))

        # Update the axes_coords to the new number of points
        axes_coords['kx'] = np.linspace(ranges['kx'][0],
                                        ranges['kx'][1],
                                        resolution[0])

        # update the y axis to the new range/number of points
        if isinstance(ky, (float, int)):  # if a constant ky image is requested.
            dE = ((ranges['Eb'][1] - ranges['Eb'][0]) *
                  added_points[0] / initial_resolution[0])
            axes_coords['Eb'] = np.linspace(ranges['Eb'][0] - dE,
                                            ranges['Eb'][1] + dE,
                                            resolution[1])
        else:  # if a constant Eb is requested.
            dky = ((ranges['ky'][1] - ranges['ky'][0]) *
                   added_points[0] / initial_resolution[0])
            axes_coords['ky'] = np.linspace(ranges['ky'][0] - dky,
                                            ranges['ky'][1] + dky,
                                            resolution[1])

        # Resize to the required resolution.
        image = zoom(image, round(resolution[0] / initial_resolution[0]),
                     order=3)

        return image, axes_coords
