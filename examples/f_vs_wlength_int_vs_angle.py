# -*- coding: utf-8 -*-
"""
Name        : f_vs_wlength_int_vs_angle
Author      : Joan Juvert <trust.no.one.51@gmail.com>
Version     : 1.0
Description : This script is used to calculate the integral of
              F(lambda) as a function of the output angle for a fixed
              position of the radiative center.
"""

import multilayers as ml
import numpy as np
import bphysics as bp
import scipy.integrate as int
import argparse as ap

# Argument parsing
parser = ap.ArgumentParser(
        description = "This script calculates the effect of the angle " + \
        "on the integral of F(lambda) for a fixed position of the " + \
        "radiative center. The output can be dumped into a file or to " + \
        "standard output. Additionally, it can be plotted to screen using " + \
        "matplotlib. The multilayer must be defined in the source code.")
parser.add_argument(
        "--graph",
        help = "Plot the results",
        action = "store_true")
parser.add_argument(
        "-o",
        "--output",
        help = "Dump the results to a file")
args = parser.parse_args()

# Create the materials
print("Loading materials... ")

silicon = ml.Medium("silicon.dat")
air = ml.Medium("air.dat")
sio2 = ml.Medium("sio2.dat")
poly = ml.Medium("polysilicon.dat")

print("Done")

# Set the fixed parameters (position in the same units as the
# wavelengths in the files).
z = 25

# Create the multilayer
print("Building multilayer and allocating memory... ")

thicknesses = [300, 50]
multilayer = ml.Multilayer([
        air,
        [poly, thicknesses[0]],
        [sio2, thicknesses[1]],
        silicon])

# Define the wavelengths and angles at which F will be calculated
wstep = 1
wmin = multilayer.getMinMaxWlength()[0]
wmax = multilayer.getMinMaxWlength()[1]
wlist = np.arange(wmin, wmax, wstep)
astep = np.deg2rad(2.5)
amin = 0
amax = np.pi / 2
alist = np.arange(amin, amax, astep)

# We define here a suitable structured array to hold the results and
# initialize the array to store the values.
ftype = np.dtype([
        ('fx', np.complex128),
        ('fy', np.complex128),
        ('fz', np.complex128)])
resmatrix = np.empty((alist.size, wlist.size), dtype = ftype)

print("Done")

# Calculate Fx, Fy and Fz. The inner loop will iterate over angles
# while the outer one will iterate over the wavelengths. The reason is
# that each time we change the angle we only have to recalculate the
# angle, but each time we change the wavelength we recalculate both the
# refractive indices and the angles. Within the loop, we calculate fx,
# fz and fy (in that specific order) in order to change polarization two
# times instead of three.
print("Calculating F... ")

for (widx, wlength) in enumerate(wlist):
    percent = (float(widx) / wlist.size) * 100
    print("%.2f%%" % percent)
    for (aidx, angle) in enumerate(alist):
        resmatrix[aidx][widx]['fx'] = multilayer.calculateFx(z, wlength, angle)
        resmatrix[aidx][widx]['fz'] = multilayer.calculateFz(z, wlength, angle)
        resmatrix[aidx][widx]['fy'] = multilayer.calculateFy(z, wlength, angle)

# We are probably more interesed on the effect of the multilayer on the
# energy rather than the electric field. What we want is |Fy(z)|^2 for
# TE waves and |Fx(z) cosA^2 + Fz(z) sinA^2|^2 for TM waves.
ftm = np.absolute(
        resmatrix['fx'] * np.cos(alist.reshape(alist.size, 1)) ** 2 + \
        resmatrix['fz'] * np.sin(alist.reshape(alist.size, 1)) ** 2) ** 2
fte = np.absolute(resmatrix['fy']) ** 2

# For each angle, integrate over the wavelengths. We want the integral
# in relation to what we would have without the multilayer, which is the
# length of the wavelength interval multiplied by 1 (the value of Fte
# and Ftm in case there is no multilayer).
nomulti = wmax - wmin
intte = int.simps(fte, wlist, axis = 1) / nomulti
inttm = int.simps(ftm, wlist, axis = 1) / nomulti

print("Done")

# Dump data to file or stdout
comments = "# F_TE = |Fy^2|^2\n" + \
        "# F_TM = |Fx * cosA^2 + Fz * sinA^2|^2\n" + \
        "# Integral over wlength of F_TE and F_TM for different\n" + \
        "# angles at a fixed position.\n" + \
        "#angle\tInt (TE)\tInt (TM)\n"
if args.output:
    bp.wdfile(args.output, comments, np.array([alist, intte, inttm]).T, '%.6e')
else:
    print(comments)
    for i in xrange(alist.size):
        print("%.6e\t%.6e\t%.6e" % (alist[i], intte[i], inttm[i]))

# Plot data if requested
if args.graph:
    import matplotlib.pyplot as plt

    plt.plot(alist, intte, label='TE', color = 'r')
    plt.plot(alist, inttm, label='TM', color = 'b')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Integral of the energy ratio')
    plt.grid()
    plt.legend(loc=2)
    plt.title('%.1f nm' % z)
    plt.show()
    plt.close()
