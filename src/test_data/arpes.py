import math
import numpy as np
import random
from scipy.interpolate import CubicHermiteSpline, RegularGridInterpolator
from scipy.ndimage import zoom


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

    BZ_size = (2 * math.pi / lattice_constants[0]) * (1 / 2)  # half the recip. constant.

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
    symmetry_point_energies : [float, float, float]
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

    BZ_size = (2 * math.pi / lattice_constant) * (1 / 2)  # half the recip. constant.
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

    # Generate the polynomials along the high symmetry lines of one symetry section.
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
        The 'zero' offset to shift the zero values by, default is 0.

    Returns
    -------
    amplitude : float or numpy.ndarray,
        The Fermi function amplitude at the given binding energy (or energies).

    """

    kb = 8.617333262e-05  # Boltzmans constant in eV/K
    amplitude = 1 / (np.exp(-(binding_energy - fermi_energy) / (kb *
                                                                temperature)) + 1)
    amplitude = (amplitude + zero_offset) / (1 + zero_offset)

    return amplitude


def perpendicular_momentum(photon_energy, parallel_momentum,
                           binding_energy=0, inner_potential=15,
                           work_function=5):
    r"""Converts photon energies to perpendicular momentum.

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

    """

    h_bar = 1.054571817E-34  # in J.s
    m_e = 9.1093837E-31  # in Kg or J.s^2/m^2 (E=mc^2)
    A_hbar = np.sqrt(2 * m_e) / h_bar  # in J^(-1/2).m^(-1)

    Eph = photon_energy * 1.6E-19  # convert from eV to J
    k_para = parallel_momentum*1E10  # convert from Ang to m
    Eb = binding_energy * 1.6E-19  # convert from eV to J
    V0 = inner_potential * 1.6E-19  # convert from eV to J
    WF = work_function * 1.6E-19  # convert from eV to J

    E_k = (Eph - Eb - WF)  # in J
    theta = np.arcsin(np.abs(k_para) / (A_hbar * np.sqrt(E_k)))  # in rad
    kz = A_hbar * np.sqrt(E_k * np.cos(theta) ** 2 + V0)  # in m^(-1)
    kz *= 1E-10  # convert from m^(-1) to Ang^(-1)

    return kz
