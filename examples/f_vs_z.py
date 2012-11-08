# -*- coding: utf-8 -*-
"""
Name          : f_vs_z
Author        : Joan Juvert <trust.no.one.51@gmail.com>
Version       : 1.0
Description   : This script calculates F(z; theta, lambda). theta and
                lambda are fixed parameters.
"""

import multilayers as ml
import numpy as np
import bphysics as bp
import argparse as ap

# Argument parsing
parser = ap.ArgumentParser(
        description = "This script calculates the effect of a multilayer " + \
        "system on TE and TM waves generated at different points of the " + \
        "system. The output can be dumped into a file or to standard " + \
        "output. Additionally, it can be plotted to screen using " + \
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

# Set the fixed parameters (wavelength in the same units as the
# wavelengths in the files).
angle = np.deg2rad(0)
wlength = 500

# Create the multilayer
print("Building multilayer and allocating memory... ")

thicknesses = [300, 50]
multilayer = ml.Multilayer([
        air,
        [poly, thicknesses[0]],
        [sio2, thicknesses[1]],
        silicon])

# Define the positions at which F(z; theta, lambda) will be calculated
# and reserve memory for the results. Careful because Fx, Fy and Fz are
# complex numbers.
zstep = 0.5
zmin = -100
zmax = 600
zlist = np.arange(zmin, zmax + zstep, zstep)
fx = np.empty(zlist.size, dtype=np.complex128)
fy = np.empty(zlist.size, dtype=np.complex128)
fz = np.empty(zlist.size, dtype=np.complex128)

print("Done")

# Calculate Fx, Fy and Fz. We calculate Fx and Fy together in the same
# loop because they both work in TM mode. This way, the characterstic
# matrices are updated when calculating Fx and are not unnecessarily
# updated again when calculating Fz (because the characteristic matrix
# does not change.)
print("Calculating F... ")

for (index, z) in enumerate(zlist):
    fx[index] = multilayer.calculateFx(z, wlength, angle)
    fz[index] = multilayer.calculateFz(z, wlength, angle)
for (index, z) in enumerate(zlist):
    fy[index] = multilayer.calculateFy(z, wlength, angle)

# We are probably more interesed on the effect of the multilayer on the
# energy rather than the electric field. What we want is |Fy(z)|^2 for
# TE waves and |Fx(z) cosA^2 + Fz(z) sinA^2|^2 for TM waves.
ftm = np.absolute(fx * np.cos(angle) ** 2 + fz * np.sin(angle) ** 2) ** 2
fte = np.absolute(fy) ** 2

print("Done")

# Dump data to file or stdout
comments = "#z\tF_TE\tF_TM\n" + \
        "# F_TE = |Fy^2|^2\n" + \
        "# T_TM = |Fx * cosA^2 + Fz * sinA^2|^2\n"
if args.output:
    bp.wdfile(args.output, comments, np.array([zlist, fte, ftm]).T, '%.6e')
else:
    print(comments)
    for i in xrange(zlist.size):
        print("%.6e\t%.6e\t%.6e" % (zlist[i], fte[i], ftm[i]))

# Plot data if requested
if args.graph:
    import matplotlib.pyplot as plt

    plt.plot(zlist, fte, label='TE', color = 'r')
    plt.plot(zlist, ftm, label='TM', color = 'b')
    plt.axvline(x = 0, color = 'k')
    cumulative = 0
    thicknesses.reverse()
    for thick in thicknesses:
        cumulative += thick
        plt.axvline(x = cumulative, color = 'k')
    plt.xlabel('z (nm)')
    plt.ylabel('F')
    plt.grid()
    plt.legend(loc=2)
    plt.title('%.1f deg, %.1f nm' % (np.rad2deg(angle), wlength))
    plt.show()
    plt.close()
