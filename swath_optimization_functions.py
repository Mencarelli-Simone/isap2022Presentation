# This script contains the functions and the procedure to find the optimal power to signal bandwidth
# given a radar geometry, the antenna pattern, the desired swath and the system
# parameters
# updated here with spherical earth geometry
# by Simone Mencarelli
# start date: 6-5-22
from matplotlib import pyplot as plt
from scipy import integrate
import numpy as np
from tqdm import tqdm
from scipy.optimize import root_scalar

from spherical_earth_geometry_radar import *
from farField import UniformAperture, Aperture
from utils import *

from design_functions import *


def core_SNR(radarGeo: RadarGeometry, aperture: UniformAperture, ground_range, wavelength, c_light=299792458.0,
             re=6371e3, ifsphere=False):
    """
    computes the following equation:

                        λ² c Bd Vs G²
    C = --------------------------------------------
        128 π^3 R^4 k sin(η) int_Bd(|H(fd)|^2  δfd)

    η incidence angle obtained from ground range
    R true range obtained from ground range
    G peak 2-way antenna gain
    H(fd) normalization filter given by antenna pattern over range-doppler
    Bd nominal doppler bandwidth

    :param radarGeo: Geometry object containing the problem geometry
    :param aperture: Aperture object containing the pattern description and antenna size
    :param ground_range: ground range axis
    :param wavelength: wave length of the signal
    :param c_light: optional propagation speed
    :return: C
    """
    if ifsphere:
        ## find incidence axis:
        # print('ground range =', ground_range)
        alpha = -ground_range / re
        # slant range
        r = np.sqrt((re * np.sin(alpha)) ** 2 + radarGeo.S_0[2] ** 2)
        # from sine theorem
        incidence = np.arcsin((re + radarGeo.S_0[2]) / r * np.sin(alpha))  ## found
        # print(incidence * 180 / np.pi)
        C, daz = core_snr_spherical(radarGeo, aperture, incidence, wavelength, radarGeo.abs_v, radarGeo.S_0[2])
        return C

    else:
        # %%    NOMINAL DOPPLER BANDWIDTH AND DOPPLER CENTROID
        integration_time = np.tan(np.arcsin(wavelength / aperture.L)) * radarGeo.S_0[2] / \
                           (np.cos(radarGeo.side_looking_angle) * radarGeo.abs_v)
        it = - integration_time / 2
        # 3-db beam-width Doppler bandwidth:
        doppler_bandwidth = float(-4 * radarGeo.abs_v ** 2 * it /
                                  (wavelength * np.sqrt(radarGeo.abs_v ** 2 * it ** 2 +
                                                        (radarGeo.S_0[2] / np.cos(
                                                            radarGeo.side_looking_angle)) ** 2)))  # this is wrong todo correct
        # doppler centroid calculated assuming the antenna pattern symmetric
        gamma = radarGeo.forward_squint_angle
        doppler_centroid = 2 * radarGeo.abs_v * np.sin(gamma) / wavelength

        # %%    RANGE-DOPPLER AXIS and COORDINATES TRANSFORMATIONS
        doppler_points_no = 281  # casual, but odd so we get a good sampling of the broadside value (good for simpson integration)
        doppler_axis = np.linspace(doppler_centroid - np.abs(doppler_bandwidth) / 2,
                                   doppler_centroid + np.abs(doppler_bandwidth) / 2,
                                   doppler_points_no)
        range_axis = np.sqrt(radarGeo.S_0[2] ** 2 + ground_range ** 2)
        # range doppler meshgrid
        R, D = np.meshgrid(range_axis, doppler_axis)
        # range Azimuth equivalent
        R, A = mesh_doppler_to_azimuth(R, D, float(wavelength), float(radarGeo.abs_v))
        # gcs points on ground
        X, Y = mesh_azimuth_range_to_ground_gcs(R, A, radarGeo.velocity, radarGeo.S_0)
        # lcs points as seen by radar
        X, Y, Z = mesh_gcs_to_lcs(X, Y, np.zeros_like(X), radarGeo.Bc2s, radarGeo.S_0)
        # spherical coordinates of the antenna pattern
        R1, T, P = meshCart2sph(X, Y, Z)

        # %%    ANTENNA PATTERN RETRIEVAL
        ground_illumination = aperture.mesh_gain_pattern_theor(T, P)
        # ground_illumination = aperture.mesh_gain_pattern(T, P)
        # print("done")
        # and normalize the gain pattern
        max_gain = aperture.max_gain()
        # max_gain = 4 * np.pi * aperture.W * aperture.L / wavelength**2
        ground_illumination /= max_gain

        # %%    INTEGRAL
        # the integrand then becomes
        I_norm = (4 * radarGeo.abs_v ** 2 - D ** 2 * wavelength ** 2) ** (3 / 2) / (np.abs(ground_illumination) ** 2)
        # and the azimuth integral for each ground range line is (integrated antenna pattern in range)
        w_range = integrate.simps(I_norm, D, axis=0)

        # %%    CORE SNR
        # the core snr over ground range is then given by:
        # Boltzman constant
        k_boltz = 1.380649E-23  # J/K
        # the sin of the incidence angle at each ground range point
        sin_eta = - ground_range / (np.sqrt(ground_range ** 2 + radarGeo.S_0[2] ** 2))
        # The range at each ground range point
        range_ = np.sqrt(ground_range ** 2 + radarGeo.S_0[2] ** 2) / np.cos(radarGeo.forward_squint_angle)
        # the equation is then: (equivalent to the above, just simplified)
        SNR_core = wavelength ** 3 * max_gain ** 2 * c_light * radarGeo.abs_v ** 2 * doppler_bandwidth / \
                   (32 * np.pi ** 3 * range_ ** 3 * k_boltz * sin_eta * w_range)

    return SNR_core.astype('float64')


def theor_core_SNR(radarGeo: RadarGeometry, aperture: Aperture, ground_range, wavelength, c_light=299792458.0):
    """
    computes the following equation:

                  λ^3 c G²
    C = ----------------------------
          256 π^3 R^3 k Vs sin(η)

    η incidence angle obtained from ground range
    R true range obtained from ground range
    G peak 2-way antenna gain

    :param radarGeo: Geometry object containing the problem geometry
    :param aperture: Aperture object containing the pattern description and antenna size
    :param ground_range: ground range axis
    :param wavelength: wave length of the signal
    :param c_light: optional propagation speed
    :return: C
    """
    k_boltz = 1.380649E-23  # J/K
    r = np.sqrt(ground_range ** 2 + radarGeo.S_0[2] ** 2)
    sin_eta = -ground_range / r
    return (wavelength ** 3 * c_light * (aperture.L * aperture.W * 4 * np.pi / wavelength ** 2) ** 2) / \
           (256 * np.pi ** 3 * r ** 3 * k_boltz * radarGeo.abs_v * sin_eta)


# %%
class RangeOptimizationProblem():
    def __init__(self, radarGeo: RadarGeometry, aperture: Aperture, wavelength, c_light=299792458.0,
                 desired_swath=20e3):
        """
        :param radarGeo: Geometry object containing the problem geometry
        :param aperture: Aperture object containing the pattern description and antenna size
        :param wavelength: wave length of the signal
        :param c_light: optional propagation speed
        :param desired_swath: optional, default 20 km swath for the optimization
        :return:
        """
        self.radarGeo = radarGeo
        self.aperture = aperture
        self.wavelength = wavelength
        self.c_light = c_light
        self.swath = desired_swath

    def error_function(self, swath_center):
        """
        error function for the optimization problem
        :param swath_center: input of the function
        :return: error: output of the function
        """
        near_range = swath_center - self.swath / 2 ** 6
        far_range = swath_center + self.swath / 2
        ground_range = np.array([near_range, far_range])
        snr_core = core_SNR(self.radarGeo, self.aperture, ground_range, self.wavelength, self.c_light, ifsphere=True)
        # print(near_range-far_range) this is ok
        error = snr_core[-1] - snr_core[0]
        return error

    def get_initial_swath_center(self):  # todo test
        # find broadside on ground
        incidence = looking_angle_to_incidence(self.radarGeo.side_looking_angle, self.radarGeo.S_0[2])
        r, rg = range_from_theta(180 / np.pi * incidence, self.radarGeo.S_0[2])
        return - float(rg)

    def optimize(self):
        opti_swath = root_scalar(self.error_function,
                                 method='secant',
                                 x0=self.get_initial_swath_center() - 200,
                                 x1=self.get_initial_swath_center() + 200)
        self.optiswath = opti_swath  # contains info about the optimization
        rmin = float(opti_swath.root) + self.swath / 2
        rmax = float(opti_swath.root) - self.swath / 2
        self.opti_rmin = rmin
        self.opti_rmax = rmax
        r_g = np.array([rmin, rmax])
        self.snr_core_edge = core_SNR(self.radarGeo, self.aperture, r_g, self.wavelength, ifsphere=True)
        # print(self.snr_core_edge)
        return rmin, rmax, opti_swath

    def power_over_bandwidth(self, loss_noise_figure, antenna_temperature, NESZ_min):
        if not hasattr(self, 'snr_core_edge'):
            print("performing optimization")
            self.optimize()
        if (self.snr_core_edge[0] - self.snr_core_edge[1]) ** 2 > 0.001:
            print("Error, swath optimization not converging")
        return loss_noise_figure * antenna_temperature / (NESZ_min * self.snr_core_edge[0])

    def power_over_bandwidth_theor(self, loss_noise_figure, antenna_temperature, NESZ_min):
        r = np.array((self.get_initial_swath_center()))
        return loss_noise_figure * antenna_temperature / (
                NESZ_min * theor_core_SNR(self.radarGeo, self.aperture, r, self.wavelength))


def better_sweep(NESZ_min=10 ** (-1 / 10), Ares=3):
    # sweeps over possible looking angles and antenna lengths, given a fixed antenna width
    # %% initial conditions
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
    c = 299792458.0
    wavel = c / f
    # create a radar geometry
    radGeo = RadarGeometry()
    #   looking angle deg
    side_looking_angle = 30  # degrees
    radGeo.set_rotation(side_looking_angle / 180 * np.pi, 0, 0)
    #   altitude
    altitude = 500e3  # m
    radGeo.set_initial_position(0, 0, altitude)
    #   speed
    radGeo.set_speed(radGeo.orbital_speed())

    # problem creation
    opti = RangeOptimizationProblem(radGeo, antenna, wavel)

    # %% physical parameters sweep
    looking_angle = np.linspace(30, 40, 2)
    antenna_length = np.linspace(1, 4, 11)
    Look_angle, Ant_l = np.meshgrid(looking_angle, antenna_length)
    C_min = np.zeros_like(Look_angle)
    opti.swath = 25  # km ## todo make this parametric
    # ACTUAL SWEEP
    for cc in tqdm(range(len(looking_angle))):
        # set looking angle
        opti.radarGeo.set_rotation(looking_angle[cc] * np.pi / 180, 0, 0)
        for rr in tqdm(range(len(antenna_length))):
            # set antenna length
            opti.aperture.set_length(antenna_length[rr])
            # get minimum power over bandwidth
            opti.optimize()
            # get core snr
            C_min[rr, cc] = np.average(opti.snr_core_edge)

    # %% system losses and powa normalization
    # sweep for diferent resolutions
    # Ares = np.array([1, 3, 5, 7])
    Ares = np.array([2])
    NESZ_min = 10 ** ((18 / 5 - 8 / 5 * Ares) / 10)
    # params
    Loss = 10 ** ((5 + 4) / 10)  # F + Lsys
    T_ant = 300
    for ii in tqdm(range(len(Ares))):
        print("Ares = ", Ares[ii])
        PoverB = Loss * T_ant / (NESZ_min[ii] * C_min)
        incid = looking_angle_to_incidence(Look_angle * np.pi / 180, opti.radarGeo.S_0[2])
        B = opti.c_light * Ant_l / (4 * Ares * np.sin(
            incid))
        P = PoverB * B
        # %% plotting
        fig1, ax = plt.subplots(1)
        plt.title(str("Ares = ") + str(Ares[ii]))
        ax2 = ax.twinx()
        for jj in range(len(looking_angle)):
            ax.plot(antenna_length, P[:, jj], label='theta = ' + str(looking_angle[jj]))
            ax2.plot(antenna_length, B[:, jj] / 1e6, '--', label='theta = ' + str(looking_angle[jj]))
        ax.legend()
        ax.set_xlabel('antenna length [m]')
        ax.set_ylabel('P_min [W]  _____')
        ax2.set_ylabel('B [MHz] - - - -')
        ax.grid()
        ax.grid()


def model_param(NESZ_min, Ares, Wg, La, looking_angle, L=10, T_ant=300):
    """
    paper's design methodology
    :param NESZ_min: minimum NESZ to obtain
    :param Ares: resolution area
    :param Wg: swath width [m]
    :param La: antenna length vector Has to be a Numpy array
    :param looking_angle: radar looking angle in degrees
    :param L: losses in decibel, default 10dB
    :param T_ant: antenna Temperature, default 300 k
    :return: power, bandwidth (vectors, same length of La)
    """

    # sweeps over possible looking angles and antenna lengths, given a fixed antenna width
    # %% initial conditions
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
    c = 299792458.0
    wavel = c / f
    # create a radar geometry
    radGeo = RadarGeometry()
    #   looking angle deg
    side_looking_angle = looking_angle  # degrees
    radGeo.set_rotation(side_looking_angle / 180 * np.pi, 0, 0)
    #   altitude
    altitude = 500e3  # m
    radGeo.set_initial_position(0, 0, altitude)
    #   speed
    radGeo.set_speed(radGeo.orbital_speed())

    # problem creation
    opti = RangeOptimizationProblem(radGeo, antenna, wavel)

    # looking_angle = np.linspace(30, 40, 2)
    antenna_length = La
    Look_angle, Ant_l = np.meshgrid(looking_angle, antenna_length)
    C_min = np.zeros_like(Ant_l)
    opti.swath = float(Wg)  # m
    # print(opti.swath)
    # ACTUAL SWEEP
    # set looking angle
    opti.radarGeo.set_rotation(looking_angle * np.pi / 180, 0, 0)
    for rr in tqdm(range(len(antenna_length))):
        # set antenna length
        opti.aperture.set_length(antenna_length[rr])
        # get minimum power over bandwidth
        rmin, rmax, opt = opti.optimize()
        # print(opt)
        # print(rmin, rmax, rmax - rmin)
        cc = core_SNR(opti.radarGeo, opti.aperture, np.array([rmin, rmax]), opti.wavelength, ifsphere=True)
        # print('c=', 10 * np.log10(cc))
        # get core snr
        C_min[rr] = np.average(opti.snr_core_edge.astype('float64')).astype('float64')

    # %% system losses and powa normalization
    # params
    Loss = 10 ** (L / 10)  # F + Lsys

    # print("Ares = ", Ares)
    PoverB = Loss * T_ant / (NESZ_min * C_min)
    incid = looking_angle_to_incidence(Look_angle * np.pi / 180, opti.radarGeo.S_0[2])
    B = opti.c_light * Ant_l / (4 * Ares * np.sin(
        incid))
    P = PoverB * B
    return P, B


if __name__ == '__main__':
    # sweep_res_nesz()
    better_sweep()
    a = input()
    # tested sphere in coresnr
