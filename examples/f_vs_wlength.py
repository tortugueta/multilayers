# -*- coding: utf-8 -*-
"""
Name        : f_vs_wlength
Author      : Joan Juvert <trust.no.one.51@gmail.com>
Version     : 1.0
Description : This scripts calculates F()lambda; theta, z). theta
              and z are fixed parameters.

Copyright 2012 Joan Juvert

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import multilayers as ml
import numpy as np
import bphysics as bp
import argparse as ap

# Argument parsing
parser = ap.ArgumentParser(
        description = "This script calculates the effect of a multilayer " + \
        "system on TE and TM waves of different wavelengths. The output " + \
        "can be dumped into a file or to standard output. Additionally, " + \
        "it can be plotted to screen using matplotlib. The multilayer " + \
        "must be defined in the source code.")
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
angle = np.deg2rad(65)
z = 25

# Create the multilayer
print("Building multilayer and allocating memory... ")

thicknesses = [300, 50]
multilayer = ml.Multilayer([
        air,
        [poly, thicknesses[0]],
        [sio2, thicknesses[1]],
        silicon])

# Define the wavelengths for which F(z; theta, lambda) will be
# calculated and reserve memory for the results. Careful because Fx, Fy
# and Fz are complex numbers.
wstep = 0.5
wmin = multilayer.getMinMaxWlength()[0]
wmax = multilayer.getMinMaxWlength()[1]
wlist = np.arange(wmin, wmax, wstep)
fx = np.empty(wlist.size, dtype=np.complex128)
fy = np.empty(wlist.size, dtype=np.complex128)
fz = np.empty(wlist.size, dtype=np.complex128)

print("Done")

# Calculate Fx, Fy and Fz. We calculate first fx and fz because there is
# no state change between them so there will only fix the state twice in
# each iteration. If we did fx, fy and fz, then we would set the state
# three times because there is a change in polarization from fx to fy
# and from fy to fz. We don't do it in two separate loops as in
# f_vs_z.py because in that case we would calculate the refractive
# indices and angles twice instead of once.
print("Calculating F... ")

for (index, wlength) in enumerate(wlist):
    fx[index] = multilayer.calculateFx(z, wlength, angle)
    fz[index] = multilayer.calculateFz(z, wlength, angle)
    fy[index] = multilayer.calculateFy(z, wlength, angle)

# We are probably more interesed on the effect of the multilayer on the
# energy rather than the electric field. What we want is |Fy(z)|^2 for
# TE waves and |Fx(z) cosA^2 + Fz(z) sinA^2|^2 for TM waves.
ftm = np.absolute(fx * np.cos(angle) ** 2 + fz * np.sin(angle) ** 2) ** 2
fte = np.absolute(fy) ** 2

print("Done")

# Dump data to file or stdout
comments = "#wlength\tF_TE\tF_TM\n" + \
        "# F_TE = |Fy^2|^2\n" + \
        "# T_TM = |Fx * cosA^2 + Fz * sinA^2|^2\n"
if args.output:
    bp.wdfile(args.output, comments, np.array([wlist, fte, ftm]).T, '%.6e')
else:
    print(comments)
    for i in xrange(wlist.size):
        print("%.6e\t%.6e\t%.6e" % (wlist[i], fte[i], ftm[i]))

# Plot data if requested
if args.graph:
    import matplotlib.pyplot as plt

    plt.plot(wlist, fte, label='TE', color = 'r')
    plt.plot(wlist, ftm, label='TM', color = 'b')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Energy ratio')
    plt.grid()
    plt.legend(loc=2)
    plt.title('%.1f rad, %.1f nm' % (angle, z))
    plt.show()
    plt.close()
