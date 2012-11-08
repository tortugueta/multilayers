# -*- coding: utf-8 -*-
"""
Name        : radcenter_distribution
Author      : Joan Juvert <trust.no.one.51@gmail.com>
Version     : 1.0
Description : This script calculates the influence of the distribution of
            : radiative centers in the active layer on the observed
            : spectrum.
"""

import multilayers as ml
import numpy as np
import bphysics as bp
import scipy.integrate as integ
import argparse as ap
import sys
import pdb

# Argument parsing
parser = ap.ArgumentParser(
        description = "This script calculates the effect of the " + \
        "distribution of radiative centers in the active layer on " + \
        "the modificator to the spectrum. The observation angle is " + \
        "a fixed parameter. Optionally, the output can be plotted " + \
        "and output to the standard output or to a file. The matrix " + \
        "containing the values of F(z, lambda) can be saved to a file " + \
        "and recovered in a following run of the program to avoid " + \
        "recalculating it in case we want to calculate the effect of " + \
        "different distributions on the same system.")
parser.add_argument(
        "--graph",
        help = "Plot the results",
        action = "store_true")
parser.add_argument(
        "-o",
        "--output",
        help = "Dump the results to a file")
parser.add_argument(
        "-s",
        "--savematrix",
        help = "Save the matrix with the F(z, lambda) values to a file")
parser.add_argument(
        "-l",
        "--loadmatrix",
        help = "Load the matrix with the F(z, lambda) values from a file")
args = parser.parse_args()

# Load the depth distribution of radiative centers. Note that the origin
# and units of z must be the same as in the multilayer.The distribution
# should be normalized to 1.
print("Loading the distribution...")
path = "/home/joan/Dropbox/CNM/projectes/simulations_report/figures/" + \
        "rcdistributions/"
distribution = bp.rdfile(path + "gaussian_m25_s07.dat", usecols = [0, 1])[1]
print("Done")

print("Checking the distribution...")
integral = integ.simps(distribution[:, 1], distribution[:, 0], 0)
np.testing.assert_almost_equal(integral, 1, 2)
print("Done")

# If we load the values of F(z, lambda) calculated in a previous
# execution we do not need to build the multilayer and repeat the
# calculation of the F function. Notice that the values of z at which
# the new distribution is sampled should be the same as the previous
# one.
if args.loadmatrix:
    print("Loading matrix...")
    fmatrix = np.load(args.loadmatrix)
    zlist = fmatrix['zlist']
    np.testing.assert_array_equal(zlist, distribution[:, 0])
    wlist = fmatrix['wlist']
    angle = fmatrix['angle']
    fte = fmatrix['fte']
    ftm = fmatrix['ftm']
    print("Done")
else:
    # Create the materials
    print("Loading materials... ")
    silicon = ml.Medium("silicon.dat")
    air = ml.Medium("air.dat")
    sio2 = ml.Medium("sio2.dat")
    poly = ml.Medium("polysilicon.dat")
    print("Done")

    # Set the fixed parameters.
    angle = np.deg2rad(0)

    # Create the multilayer
    print("Building multilayer and allocating memory... ")
    thicknesses = [300, 50]
    multilayer = ml.Multilayer([
            air,
            [poly, thicknesses[0]],
            [sio2, thicknesses[1]],
            silicon])

    # Define the wavelengths and z coordinates at which F will be calculated
    # and allocate memory for the results. We will use a structured array to
    # store the values of F(z, lambda).
    wstep = 1
    wmin = multilayer.getMinMaxWlength()[0]
    wmax = multilayer.getMinMaxWlength()[1]
    wlist = np.arange(wmin, wmax, wstep)
    zlist = distribution[:, 0]

    ftype = np.dtype([
            ('fx', np.complex128),
            ('fy', np.complex128),
            ('fz', np.complex128)])
    resmatrix = np.empty((zlist.size, wlist.size), dtype = ftype)
    print("Done")

    # I(wavelength, theta) = s(wavelength) * F'(wavelength, theta), where
    # F'(wav, theta) = integral[z](|F|^2 * rcdist(z). Therefore, we
    # calculate the new spectrum as a modification to the original spectrum.
    # The modification factor F'(wav, theta) is an integral over z.

    # First calculate |Fy|^2 for te and |Fx*cos^2 + Fz*sin^2|^2 for tm. We
    # do fx and fz in one loop and fy in another independent loop to avoid
    # recalculating the characteristic matrix at every iteration due to the
    # change of polarization.
    print("Calculating F...")
    for (widx, wlength) in enumerate(wlist):
        percent = (float(widx) / wlist.size) * 100
        print("%.2f%%" % percent)
        for (zidx, z) in enumerate(zlist):
            resmatrix[zidx][widx]['fx'] = multilayer.calculateFx(z, wlength, angle)
            resmatrix[zidx][widx]['fz'] = multilayer.calculateFz(z, wlength, angle)
        for (zidx, z) in enumerate(zlist):
            resmatrix[zidx][widx]['fy'] = multilayer.calculateFy(z, wlength, angle)

    # We are probably more interesed on the effect of the multilayer on the
    # energy rather than the electric field. What we want is |Fy(z)|^2 for
    # TE waves and |Fx(z) cosA^2 + Fz(z) sinA^2|^2 for TM waves.
    ftm = np.absolute(
            resmatrix['fx'] * np.cos(angle) ** 2 + \
            resmatrix['fz'] * np.sin(angle) ** 2) ** 2
    fte = np.absolute(resmatrix['fy']) ** 2
    print("Done")

    # Notice that until now we have not used the distribution of the
    # radiative ceneters, but the calculation of ftm and fte is costly.
    # If requested, we can save fte and ftm to a file. In a following
    # execution of the script, the matrix can be loaded from the file
    # instead of recalculated.
    if args.savematrix:
        print("Saving matrix...")
        np.savez(args.savematrix, fte = fte, ftm = ftm, zlist = zlist,
                wlist = wlist, angle = angle)
        print("Done")

# Build or load the original spectrum. It should be sampled at the same
# wavelengths defined in wlist. If we are interested only in the
# modificator to the spectrum, not in the modified spectrum, we can
# leave it at 1.
original_spec = 1

# Multiply each F(z, lambda) by the distribution.
print("Integrating...")
distval = distribution[:, 1].reshape(distribution[:, 1].size, 1)
fte_mplied = fte * distval
ftm_mplied = ftm * distval
fte_int = integ.simps(fte_mplied, zlist, axis = 0)
ftm_int = integ.simps(ftm_mplied, zlist, axis = 0)
spectrum_modte = original_spec * fte_int
spectrum_modtm = original_spec * ftm_int
print("Done")

# Dump data to file or stdout
comments = "# F_TE = |Fy^2|^2\n" + \
        "# F_TM = |Fx * cosA^2 + Fz * sinA^2|^2\n" + \
        "# Modified spectrum for TE and TM waves for a\n" + \
        "# distributions of the radiative centers.\n" + \
        "# wlength\tF_TE\tF_TM"

if args.output:
    bp.wdfile(args.output, comments,
            np.array([wlist, spectrum_modte, spectrum_modtm]).T, '%.6e')
else:
    print(comments)
    for i in xrange(wlist.size):
        print("%.6e\t%.6e\t%.6e" % (wlist[i], spectrum_modte[i],
                spectrum_modtm[i]))

# Plot data if requested
if args.graph:
    import matplotlib.pyplot as plt

    plt.plot(wlist, spectrum_modte, label='TE', color = 'r')
    plt.plot(wlist, spectrum_modtm, label='TM', color = 'b')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Energy ratio')
    plt.grid()
    plt.legend(loc=2)
    plt.title('%.1f rad' % angle)
    plt.show()
    plt.close()
