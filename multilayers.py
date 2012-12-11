# -*- coding: utf-8 -*-
"""
Name          : multilayers
Author        : Joan Juvert <trust.no.one.51@gmail.com>
Version       : 1.0
Description   : A class library to simulate light propagation in
              : multilayer systems.

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

################################# ToDo ################################
#
# Some attributes that have a 'get' method could be decorated as
# properties in order to supress the parantheses in the method call.
#
# I should take advantage of the symmetries in the characterisic
# matrix of individual layers (m11 and m22 are equal)
#
# Add tests for the symmetry of R and T when all the layers are
# insulators.
#
#######################################################################


import bphysics as bp
import numpy as np
import scipy.interpolate as interpolation


############################ Class definitions ########################


class Medium(object):
    """
    The Medium class implements an object representing an optical
    medium (basically its refractive index).

    It contains the minimum and maximum wavelengths for which the
    refractive index is known and a couple of interpolators to calculate
    the refractive index and extintion coefficient at any wavelength in
    the available range.

    All the attributes are private and accessed through the provided
    methods.
    """

    def __init__(self, filename, comments='#', delimiter=None,
                 converters=None, skiprows=0, usecols=None):
        """
        Initialize a Medium instance.

        The refractive indices that characterize the medium are read
        from a text file. After loading the table of refractive indices
        an interpolator is built that allows to calculate the refractive
        index at any wavelength within the available range.

        Note that the table is actually read through the numpy.loadtxt
        function. The loaded text file must have a column with the
        wavelength values, another with the real part of the refractive
        index, and another with its imaginary part. If there are other
        columns in your file, or there are not in that order, the
        'usecols' optional argument can be used to select which columns
        to read.

        Parameters
        ----------
        filename : str
            Path to the file containing the table of triplets
            (wavelenght, n, k) that characterize the index of refraction
            of the medium.
        comments : str, optional
            The character used to indicate the start of a comment;
            default: '#'.
        delimiter : str, optional
            The string used to separate values.  By default, this is any
            whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will
            convert that column to a float.  E.g., if column 0 is a date
            string:``converters = {0: datestr2num}``.  Converters can
            also be used to provide a default value for missing data
            (but see also `genfromtxt`):
            ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines; default: 0.
        usecols : sequence, optional
            Which columns to read, with 0 being the first.  For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th
            columns. The default, None, results in all columns being
            read.

        Returns
        -------
        out : Medium
            A Medium instance.

        See also
        --------
        numpy.loadtxt
        """

        # Initialize variables
        self.__maxWlength = None
        self.__minWlength = None
        self.__nInterpolator = None
        self.__kInterpolator = None

        # Load the table of refractive indices and generate the
        # interpolators.
        table = np.loadtxt(filename, 'float', comments, delimiter,
                           converters, skiprows, usecols)
        wavelengths = table[:, 0]
        refrIndex = table[:, 1]
        extCoef = table[:, 2]
        self.__maxWlength = wavelengths.max()
        self.__minWlength = wavelengths.min()
        self.__nInterpolator = interpolation.interp1d(
                wavelengths, refrIndex, kind='cubic')
        self.__kInterpolator = interpolation.interp1d(
                wavelengths, extCoef, kind='cubic')

    def getRefrIndex(self, wavelength):
        """
        Returns the complex refractive index at the given wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength at which we want to calculate the complex
            refractive index. In the same units as in the file from
            which the refractive indices were loaded.

        Returns
        -------
        out : numpy.complex128
            The complex refractive index.
        """

        try:
            return self.__nInterpolator(wavelength) + \
                   self.__kInterpolator(wavelength) * 1j
        except ValueError:
            print("Error: you are trying to work at a wavelength outside " + \
                  "the range where the refractive indices are known")
            raise

    def getMinMaxWlength(self):
        """
        Returns a tuple (min, max) with the shortest and longest
        wavelengths for which the refractive index is known.

        Returns
        -------
        out : tuple
            A tuple with the minimum and maximum wavelengths for which
            the refractive index can be calculated. In the same units as
            in the file from which the refractive indices were loaded.
        """

        return (self.__minWlength, self.__maxWlength)


class Multilayer(object):
    """
    The Multilayer class implements a layered optical medium in a
    logical way. That allows to perform some complex calculations in an
    understandable and flexible way.

    All the attributes are private and accessed through the provided
    methods. The structure is the following:

    workingWavelength
    minMaxWlength
    polarization
    charMatrixUpDown
    charMatrixDownUp
    coefficientsUpDown --> {'r', 't', 'R', 'T'}
    coefficientsDownUp --> {'r', 't', 'R', 'T'}
    stack --> [
               top medium,
               layer 1
               .
               .
               .
               layer N,
               bottom medium  ----> {
              ]                      'medium',  ------> Medium instance
                                     'position',
                                     'thickness',
                                     'angle',
                                     'matrix'
                                     'refindex'
                                    }

    #There are properties that are common to the whole system:
        - Wavelength of the light.
        - Minimum and maximum wavelengths at which the refractive
          indices can be calculated in all the layers.
        - Polarization of the light.
        - The characteristic matrix in the up-down direction of
          propagation.
        - The characteristic matrix in the down-up direction of
          propagation.
        - The optical coefficients (reflection coefficient, refraction
          coefficient, reflectance and transmittance).

    The stack is implemented as a list and contains parameters that
    change in each layer. Each layer is a dictionary with the following
    data:
        - The medium (determines the refractive index). This is a
          reference to a Medium instance.
        - The position (z coordinate) of the layer.
        - The thickness of the layer.
        - The propagation angle of the light
        - The characteristic matrix of the layer.
        - The complex refractive index of the layer at the current
          wavelength.
    """

    def __init__(self, mediums):
        """
        Generates a multilayer structure.

        Note that a system with a layer of zero thickness is physically
        the same as the system without that layer, and the results of
        the simulations will be the same. However, bear in mind that you
        cannot "grow" a nonexistent layer but you can "grow" an existing
        zero thickness layer (i.e, change its thickness).

        Initializing the multilayer does not involve providing any
        information regarding the properties of the light propagating
        across it. Before any calculation can be made you must enter the
        wavelength, polarization and propagation angle of the light
        using the appropiate methods.

        Parameters
        ----------
        mediums : list
            A list containing the Medium instances that form the
            multilayer system. The first element of the list corresponds
            to the upper medium and the last one corresponds to the
            bottom medium. At least two mediums must be given.

            Each element of the list (except the first and the last) is
            another list with two elements: the first is a reference to
            a Medium instance, and the second is a scalar representing
            the thickness of that layer. If only a Medium instance is
            given instead of a list with two elements, then the
            thickness will be considered zero. If the thickness is an
            'int' it will be promoted to 'float'. The thickness must be
            in the same units as the wavelength.

            The first and last elements of "mediums" are just a
            reference to the corresponding Medium instances. The
            thickness is not necessary because they represent the top
            and bottom mediums and the thickness will be considered
            infinite.

        Returns
        -------
        out : Multilayer
            A multilayer instance.

        Example
        -------
        If topmedium, layer1, layer2 and bottommedium are Medium
        instances, then the following statement builds a system with
        topmedium and bottommedium as top and bottom mediums
        respectively, and two layers, one of layer1 10 units thick and
        another of layer2 15 units thick. The thickness in the same
        units as the wavelengths.
        system = Multilayer([
                topmedium,
                [layer1, 10],
                [layer2, 15],
                bottommedium])
        """

        # Properties of the light common to all the layers of the system
        self.__workingWavelength = None
        self.__polarization = None
        self.__minMaxWlength = None

        # List of the mediums conforming the multilayer system. Each
        # element is a dictionary with the following elements keys:
        #    - medium: a reference to the corresponding Medium instance.
        #    - position: position of the lower interface of the layer.
        #      This is calculated automatically. The origin is at the
        #      boundary between the lower medium and the next layer.
        #    - thickness: thickness of the layer.
        #    - propangle: propagation angle of the light.
        #    - matrix: characteristic matrix of the layer.
        self.__stack = []

        # The following instance variables contain the characteristic
        # matrices of the system (one for the up->down direction and
        # another for the opposite) and a the coefficients of the system
        # (also for both directions). The coefficients are stored in a
        # dictionary with the following keys:
        #   - r: reflection coefficient
        #   - t: transmission coefficient
        #   - R: reflectivity
        #   - T: transmittivity
        self.__charMatrixUpDown = None
        self.__charMatrixDownUp = None
        self.__coefficientsUpDown = {
                'r': None, 't': None, 'R': None, 'T': None}
        self.__coefficientsDownUp = {
                'r': None, 't': None, 'R': None, 'T': None}

        # Check that we get at least two mediums
        try:
            len(mediums)
        except TypeError:
            error = "Multilayer creation error: a list of mediums is expected"
            print(error)
            raise
        if len(mediums) < 2:
            error = "Multilayer creation error: at least two mediums must " + \
                    "be given"
            print(error)
            raise ValueError

        # Start the creation of the multilayer
        for (index, medium) in enumerate(mediums):
            if (index == 0) or (index == len(mediums) - 1):
                # First and last mediums.
                # Check that we are given a Medium instance.
                if not isinstance(medium, Medium):
                    error = "Multilayer creation error: element " + \
                            "%i is not a Medium instance" % index
                    print(error)
                    raise TypeError

                self.__stack.append({
                        'medium': medium, 'position': None,
                        'thickness': np.infty, 'propangle': None,
                        'matrix': None, 'refindex': None})
            else:
                # Intermediate layers.
                # If we have a Medium instance we consider the
                # thickness to be zero. Otherwise we expect a list
                # [medium, thickness]
                if isinstance(medium, Medium):
                    self.__stack.append({
                            'medium': medium, 'position': None,
                            'thickness': 0.0, 'propangle': None,
                            'matrix': None, 'refindex': None})
                elif isinstance(medium, list):
                    if len(medium) != 2:
                        error = "Multilayer creation error: " + \
                                "element %i must be either a " % index + \
                                "Medium instance or a list [Medium, thickness]"
                        print(error)
                        raise TypeError
                    if not isinstance(medium[0], Medium):
                        error = "Multilayer creation error: first " + \
                                "component of element %i must be " % index + \
                                "a Medium instance"
                        print(error)
                        raise TypeError
                    try:
                        thick = np.float(medium[1])
                    except TypeError:
                        error = "Multilayer creation error: element " + \
                                "%i, thickness must be a 'float' " % index + \
                                "or 'float'"
                        print(error)
                        raise
                    except ValueError:
                        error = "Multilayer creation error: element " + \
                                "%i thickness must be an 'float' " % index + \
                                "or 'float'"
                        print(error)
                        raise
                    if medium[1] < 0:
                        error = "Multilayer creation error: element " + \
                                "%i, thickness must be >= 0" % index
                        print(error)
                        raise ValueError

                    self.__stack.append({
                            'medium': medium[0], 'position': None,
                            'thickness': thick, 'propangle': None,
                            'matrix': None, 'refindex': None})
                else:
                    error = "Multilayer creation error: element " + \
                            "%i must be either a Medium instance " % index + \
                            "or a list [Medium, thickness]"
                    print(error)
                    raise TypeError

        # Calculate the positions of each layer
        self.calcPositions()

        # Make sure that there is a common range of wavelengths where
        # the refractive index can be calculated.
        # What we have to do is find de shortest and longest wavelength
        # for each medium of the multilayer and then find de longest of
        # the shortest and the shortest of the longest.
        minimums = np.empty(self.numLayers())
        maximums = np.empty(self.numLayers())
        for index, layer in enumerate(self.__stack):
            minimums[index] = layer['medium'].getMinMaxWlength()[0]
            maximums[index] = layer['medium'].getMinMaxWlength()[1]

        minimum = np.max(minimums)
        maximum = np.min(maximums)

        # Check that minimum is lower than maximum. Otherwise the
        # intersection of the ranges of all the mediums is zero and
        # therefore we do not have any refractive index common to all
        # the layers
        if minimum >= maximum:
            error = "Fatal error: it is not possible to calculate any " + \
                    "refractive index common to all mediums"
            print(error)
            raise ValueError

        self.__minMaxWlength = (minimum, maximum)

    def calcPositions(self):
        """
        This method calculates the positions of each layer along the
        z axis (the direction perpendicular to the interfaces between
        layers.

        The position of a layer corresponds to the z coordinate of its
        lower surface. The origin is located at the interface between
        the last layer and the bottom medium. Therefore, the position of
        the bottom medium is -infinity, the position of the last layer
        is 0, the one above is at 0 + the thickness of the one below,
        and so on.

        This method does not return anything, it just sets the position
        of each layer. It is automatically executed during instantiation
        of a multilayer. The user typically does not need to call it.
        """

        # We start setting the positions from below
        reverseIndex = range(len(self.__stack))
        reverseIndex.reverse()
        self.__stack[reverseIndex[0]]['position'] = -np.infty
        self.__stack[reverseIndex[1]]['position'] = 0.0
        for layerIndex in reverseIndex[2:]:
            self.__stack[layerIndex]['position'] = self.__stack[
                    layerIndex + 1]['position'] + \
                    self.getThickness(layerIndex + 1)

    def getPosition(self, layerIndex):
        """
        This method returns the position of the layer with index
        'layerIndex'.

        Parameters
        ----------
        layerIndex : int
            The index of the layer. Index 0 corresponds to the top
            medium.

        Returns
        -------
        out : float
            The position of the layer, which corresponds to the z
            coordinate of its lower surface.
        """

        if layerIndex < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        return self.__stack[layerIndex]['position']

    def setThickness(self, thickness, layerIndex):
        """
        This method changes the thickness of the layer with index
        'layerIndex' to a new value.

        Index 0 corresponds to the top medium. The positions of the
        layers above the one being changed will be recalculated
        accordingly.

        Note that the thickness of the top and bottom mediums cannot be
        changed because they are infinite.

        The characteristic matrices of the system and the coefficients
        will be reset to zero because they must be recalculated after
        the thickness change. However, only the individual matrix of
        the layer being modified will be reset to zero. The individual
        matrices of all other layers remain the same.

        Parameters
        ----------
        thickness : float
            The thickness of the layer. In the same units as the
            wavelengths.
        layerIndex : int
            The index of the layer. Index 0 corresponds to the top
            medium.
        """

        # Change the thickness of the layer
        if thickness < 0:
            error = "Negative thickness not accepted"
            print(error)
            raise ValueError
        if layerIndex < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        if (layerIndex == 0) or (layerIndex == len(self.__stack) - 1):
            error = "Error setting thickness: the thickness of the top " + \
                    "and bottom mediums cannot be changed"
            print(error)
            raise IndexError

        self.__stack[layerIndex]['thickness'] = np.float(thickness)

        # Recalculate the z coordinates of the layers and reset matrices
        # and coefficients.
        self.calcPositions()
        self.__stack[layerIndex]['matrix'] = None
        self.__charMatrixUpDown = None
        self.__charMatrixDownUp = None
        self.__coefficientsUpDown['r'] = None
        self.__coefficientsUpDown['t'] = None
        self.__coefficientsUpDown['R'] = None
        self.__coefficientsUpDown['T'] = None
        self.__coefficientsDownUp['r'] = None
        self.__coefficientsDownUp['t'] = None
        self.__coefficientsDownUp['R'] = None
        self.__coefficientsDownUp['T'] = None

    def getThickness(self, layerIndex):
        """
        This method returns the thickness of the layer with index
        'layerindex'. Index 0 corresponds to the top medium.

        Parameters
        ----------
        layerIndex : int
            The index of the layer. Index 0 corresponds to the top
            medium.

        Returns
        -------
        out : float
            The thickness of the layer. In the same units as the
            wavelengths.
        """

        if layerIndex < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        return self.__stack[layerIndex]['thickness']

    def getMinMaxWlength(self):
        """
        This method returns a tuple (min, max) with the shortest and
        longest wavelengths for which the refractive index can be
        calculated in all the layers forming the multilayer system.

        Returns
        -------
        out : tuple
            A tuple containing the minimum and maximum wavelengths
            within which the refractive index can be interpolated in all
            the mediums of the multilayer. In the same units as in the
            file from which the refractive indices were loaded.
        """

        return self.__minMaxWlength

    def numLayers(self):
        """
        This method returns the number of layers of the multilayer
        system including the top and bottom mediums.

        Returns
        -------
        out : int
            The number of layers of the multilayer system, including the
            top and bottom mediums.
        """

        return len(self.__stack)

    def getIndexAtPos(self, z):
        """
        Returns the index of the layer within which z lies.

        Parameters
        ----------
        z : float
            A z coordinate. z = 0 is at the surface between the bottom
            medium and the next layer. The position of a layer is the z
            coordinate of its lower interface. The units are the same as
            the thickness and wavelengths.

        Returns
        -------
        out : int
            The index of the layer within which z lies.
        """

        # For each layer starting at the upper medium, check if the
        # given z is larger or equal than the position of the layer. If
        # it is, then z lies in the current layer, otherwise move to the
        # next one.
        for index in range(self.numLayers()):
            if z >= self.getPosition(index):
                return index

    def setWlength(self, wavelength, rilist=None):
        """
        This method sets the wavelength of the light going through the
        multilayer system and sets the actual refractive index in each
        layer.

        Since changing the working wavelength also changes the
        refractive index in effect, the propagation angles must be
        recalculated. For that reason, this method will reset the
        propagation angle in all the layers to 'None'. That will force
        the user to execute again the setPropAngle() method.
        Otherwise the calculation of the characteristic matrices will
        rise an error.

        Also, the characteristic matrices must be recalculated.
        Therefore, they will also be reset to None along with the
        coefficients.

        Optionally, the refractive indices of the layers can be
        passed explicitly with a list. This avoids calculating each
        refractive index using the interpolator. Using this is
        dangerous and discouraged. Use it only if you know what you are
        doing.

        Parameters
        ----------
        wavelength : float
            The wavelength of the light going across the system. In the
            same units as in the file from which the refractive indices
            were loaded.
        rilist : list, optional
            A list containing the refractive index of each layer at the
            wavelength being set. The items must be ordered, the first
            one corresponding to the top layer and the last one to the
            bottom layer. Remember that the refractive indices are
            complex numbers. If you pass a real number, it will be
            converted to numpy.complex128. The use of this option is
            discouraged. Use it only if you know what you are doing.
        """

        # Only accept wavelengths within the available range
        minimum, maximum = self.getMinMaxWlength()
        if wavelength < minimum or wavelength > maximum:
            error = "Error: Wavelength out of bounds"
            print(error)
            raise ValueError
        self.__workingWavelength = np.float64(wavelength)

        # Calculate the refractive indices of each layer and reset the
        # variables that must be recalculated due to the change in the
        # wavelength.
        if rilist == None:
            for index in range(self.numLayers()):
                self.__stack[index]['propangle'] = None
                self.__stack[index]['matrix'] = None
                self.__stack[index]['refindex'] = \
                        self.__stack[index]['medium'].getRefrIndex(wavelength)
        else:
            for index in range(self.numLayers()):
                self.__stack[index]['propangle'] = None
                self.__stack[index]['matrix'] = None
                try:
                    ri = rilist[index]
                except:
                    error = "rilist must be an ordered sequence and have " + \
                            "as many items as layers in the system"
                    print(error)
                    raise TypeError
                try:
                    ri = np.complex128(ri)
                except:
                    error = "The refractive index must be a number"
                    print(error)
                    raise TypeError
                self.__stack[index]['refindex'] = ri

        self.__charMatrixUpDown = None
        self.__charMatrixDownUp = None
        self.__coefficientsUpDown['r'] = None
        self.__coefficientsUpDown['t'] = None
        self.__coefficientsUpDown['R'] = None
        self.__coefficientsUpDown['T'] = None
        self.__coefficientsDownUp['r'] = None
        self.__coefficientsDownUp['t'] = None
        self.__coefficientsDownUp['R'] = None
        self.__coefficientsDownUp['T'] = None

    def getWlength(self):
        """
        This method returns the current wavelength of the light going
        through the multilayer.

        Returns
        -------
        out : float
            The wavelength of the light going across the system. In the
            same units as in the file from which the refractive indices
            were loaded.
        """

        return self.__workingWavelength

    def setPolarization(self, polarization):
        """
        Sets the polarization of the light going through the multilayer
        system.

        Since the characteristic matrices will change, they will be
        reset to None in order to force the user to calculate them again
        with the corresponding methods. The same goes for the
        coefficients.

        Parameters
        ----------
        polarization : str
            The polarization of the ligth going across the system. It
            may be "te" or "tm", case insensitive.
        """

        try:
            polarization = polarization.upper()
            if (polarization != 'TE') and (polarization != 'TM'):
                raise ValueError
        except ValueError:
            error = "Error setting polarization: polarization must be " + \
                    "'te' or 'tm'"
            print(error)
            raise ValueError
        except AttributeError:
            error = "Error setting polarization: polarization must be " + \
                    "'te' or 'tm'"
            print(error)
            raise AttributeError

        self.__polarization = polarization

        # Reset the characteristic matrices and coefficients
        for index in range(self.numLayers()):
            self.__stack[index]['matrix'] = None

        self.__charMatrixUpDown = None
        self.__charMatrixDownUp = None
        self.__coefficientsUpDown['r'] = None
        self.__coefficientsUpDown['t'] = None
        self.__coefficientsUpDown['R'] = None
        self.__coefficientsUpDown['T'] = None
        self.__coefficientsDownUp['r'] = None
        self.__coefficientsDownUp['t'] = None
        self.__coefficientsDownUp['R'] = None
        self.__coefficientsDownUp['T'] = None

    def getPolarization(self):
        """
        Returns the polarization of the light going through the
        multilayer system.

        Returns
        -------
        out : str
            The polarization of the ligth going across the system. It
            may be "TE" or "TM".
        """

        return self.__polarization

    def setPropAngle(self, angle, index=0):
        """
        Sets the propagation angle of light in the layer of given
        index.

        The propagation angle in all other layers is automatically
        calculated using Snell's Law. The angle must be given in
        radians.

        If a list is given instead of a single angle, the angles will
        be set to those found in the list following the natural order
        (first angle to top medium, last angle to bottom medium). Use
        of this feature is strongly discouraged. Use it only if you
        know what you are doing.

        Since the characteristic matrices and coefficients must be
        recalculated, they will be reset to None to force the user to
        execute again the relevant methods for its calculation.

        Parameters
        ----------
        angle : float or complex or list of floats or complexes
            Propagation angle in radians. Use of a list instead of a
            single angle is strongly discouraged. Use it only if you
            know what you are doing.
        index : int, optional
            The index of the layer at which light propagates with the
            given angle. If not specified, it will be assumed that the
            angle corresponds to the propagation in the upper medium
            (index = 0).
        """

        # Do not accept a negative index.
        if index < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        if index >= self.numLayers():
            error = "Layer %i does not exist" % index
            print(error)
            raise IndexError

        # We want to work always with complex angles for when we have
        # propagation beyond the critical angle.
        angle = np.complex128(angle)

        if self.getWlength() == None:
            error = "Error setting propagation angle: a working " + \
                    "wavelength has not been set"
            print(error)
            raise ValueError
        wavelength = self.getWlength()

        if type(angle) == np.complex128:
            # We set the angle in the layer specified in the argument. All
            # other layers get the appropiate angle calculated using Snell's
            # law
            sine_i = np.sin(angle)
            n_i = self.getRefrIndex(index)
            for layerIndex in range(self.numLayers()):
                if layerIndex == index:
                    self.__stack[layerIndex]['propangle'] = angle
                else:
                    n_f = self.getRefrIndex(layerIndex)
                    if n_f == n_i:
                        self.__stack[layerIndex]['propangle'] = angle
                    else:
                        self.__stack[layerIndex]['propangle'] = np.arcsin(
                                n_i * sine_i / n_f)
        else:
            # In this case we have a list of angles. We copy them
            # directly to the layer.
            for layerIndex in range(self.numLayers()):
                try:
                    self.__stack[layerIndex]['propangle'] = angle[layerIndex]
                except:
                    error = "angle must be a number or a list of numbers " + \
                            "with as many items as layers in the system"
                    print(error)
                    raise TypeError

        # Reset the characteristic matrices and the coefficients
        for index in range(self.numLayers()):
            self.__stack[index]['matrix'] = None

        self.__charMatrixUpDown = None
        self.__charMatrixDownUp = None
        self.__coefficientsUpDown['r'] = None
        self.__coefficientsUpDown['t'] = None
        self.__coefficientsUpDown['R'] = None
        self.__coefficientsUpDown['T'] = None
        self.__coefficientsDownUp['r'] = None
        self.__coefficientsDownUp['t'] = None
        self.__coefficientsDownUp['R'] = None
        self.__coefficientsDownUp['T'] = None

    def getPropAngle(self, index):
        """
        Returns the propagation angle of the light in the layer with the
        given index.

        Parameters
        ----------
        index : int
            The index of the layer. Index 0 corresponds to the upper
            medium.

        Returns
        -------
        out : complex
            The propagation angle. It may be complex if there has been
            total internal reflection in a lower interface.
        """

        if index < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        if index >= self.numLayers():
            error = "Layer %i does not exist" % index
            print(error)
            raise IndexError
        return self.__stack[index]['propangle']

    def getRefrIndex(self, index):
        """
        Returns the complex refractive index at the current wavelength
        within the layer with the given index.

        For example, multilayer.getRefrIndex(0) would return the complex
        refractive index at 400 units of length (nm typically) in the
        top medium.

        Parameters
        ----------
        index : int
            The index of the layer. Index 0 corresponds to the upper
            medium.

        Returns
        -------
        out : complex128
            The complex refractive index for the given wavelength at the
            medium of given index.
        """

        if index < 0:
            error = "Negative index not accepted"
            print(error)
            raise IndexError
        if index >= self.numLayers():
            error = "Layer %i does not exist" % index
            print(error)
            raise IndexError
        return self.__stack[index]['refindex']

    def calcMatrices(self, layerIndexes=[]):
        """
        This method calculates the characteristic matrix of the
        specified layers.

        Note that the top and bottom medium do not have characteristic
        matrices. An error will be raised if you try to calculate their
        characteristic matrices.

        The matrix is stored in a numpy.ndarray variable. Note that this
        method does not return anything, it just stores the calculated
        matrices in the corresponding field of the multilayer.

        Parameters
        ----------
        layerIndexes : list, optional
            A list of the indices of the layers whose characteristic
            matrix should be calculated. If the list is empty, the
            matrices of all the layers (except top and bottom mediums)
            will be calculated. Same if the parameter is skipped.

        See Also
        --------
        getMatrix
        """

        if not isinstance(layerIndexes, list):
            error = "Error: the argument of calcMatrices must be a list"
            print(error)
            raise ValueError
        if len(layerIndexes) == 0:
            # Calculate all the characteristic matrices
            layerList = range(1, self.numLayers() - 1)
        else:
            # Calculate only the characteristic matrices of the given
            # layers.
            layerList = layerIndexes

        # Perform here the actual calculation
        for layerIndex in layerList:
            if not isinstance(layerIndex, int):
                error = "Error: the layer index must be an integer"
                print(error)
                raise ValueError
            if (layerIndex == 0) or (layerIndex == self.numLayers() - 1):
                error = "Error: the characteristic matrix of the top and " + \
                        "bottom mediums cannot be calculated"
                print(error)
                raise ValueError
            if (layerIndex >= self.numLayers()) or (layerIndex < 0):
                error = "Error: valid layer indices from %i to %i" % \
                        (1, self.numLayers() - 2)
                print(error)
                raise IndexError
            lambda0 = self.getWlength()
            if lambda0 == None:
                error = "Error: the wavelength is not set"
                print(error)
                raise ValueError
            angle = self.getPropAngle(layerIndex)
            if angle == None:
                error = "Error: the propagation angle is not set"
                print(error)
                raise ValueError
            pol = self.getPolarization()
            if pol == None:
                error = "Error: the polarization is not set"
                print(error)
                raise ValueError

            n = self.getRefrIndex(layerIndex)
            cosineAngle = np.cos(angle)
            d = self.getThickness(layerIndex)
            if pol == 'TE':
                p = n * cosineAngle
            else:
                p = cosineAngle / n
            b = 2 * np.pi * n * d * cosineAngle / lambda0
            m11 = np.cos(b)
            m12 = -1j * np.sin(b) / p
            m21 = -1j * p * np.sin(b)
            m22 = m11

            self.__stack[layerIndex]['matrix'] = \
                    np.matrix([[m11, m12], [m21, m22]])

    def getMatrix(self, layerIndex):
        """
        This method returns the characteristic matrix of a given layer
        in the stack.

        Parameters
        ----------
        layerIndex : int
            The index of the layer. Index 0 corresponds to the upper
            medium.

        Returns
        -------
        out : numpy.ndarray
            The characteristic matrix of the layer with the given index.
        """

        if (layerIndex >= self.numLayers()) or (layerIndex < 0):
            error = "Error: valid layer indices from %i to %i" % \
                    (1, self.numLayers() - 2)
            print(error)
            raise IndexError
        return self.__stack[layerIndex]['matrix']

    def updateCharMatrix(self):
        """
        This method calculates the characteristic matrix of the
        multilayer in the up-down direction and the down-up direction.
        Then the coefficients r, t, R and T (reflection coefficient,
        transmission coefficient, reflectance and transmittance,
        respectively) are calculated from the characteristic matrix.

        Note that both the reflection and transmission coefficients
        refer to the ratios between reflected (transmitted) ELECTRIC
        fields to the incident ELECTRIC field regardless of the
        polarization of the wave (TE or TM).

        The coefficients are calculated for both possible directions of
        propagation (top-down and down-top). In the former case, the top
        medium is considered to be the input medium and the bottom
        medium is considered to be the exit medium. The reverse holds
        for the down-top direction.

        Before executing this method, the calcMatrices method must
        be invoked in order to calculate the characteristic matrices of
        each individual layer.

        Note that this method does not return anything, it just stores
        the global characteristic matrix in the corresponding attribute
        of the multilayer.
        """

        # Calculation of the characteristic matrices
        # Up-down direction
        charMatrixUD = np.eye(2, 2)
        for index in range(1, self.numLayers() - 1):
            matrix = self.getMatrix(index)
            if matrix == None:
                error = "Error: the characteristic matrix cannot be " + \
                        "calculated because some of the individual " + \
                        "matrices has not been calculated"
                print(error)
                raise ValueError
            charMatrixUD = charMatrixUD * matrix

        self.__charMatrixUpDown = charMatrixUD

        # Down-up direction
        charMatrixDU = np.eye(2, 2)
        for index in range(self.numLayers() - 2, 0, -1):
            matrix = self.getMatrix(index)
            if matrix == None:
                error = "Error: the characteristic matrix cannot be " + \
                        "calculated because one of the individual " + \
                        "matrices has not been calculated"
                print(error)
                raise ValueError
            charMatrixDU = charMatrixDU * matrix

        self.__charMatrixDownUp = charMatrixDU

        # Calculation of the coefficients
        # Auxiliary variables
        bottom_index = self.numLayers() - 1
        n_top = self.getRefrIndex(0)
        n_bottom = self.getRefrIndex(bottom_index)
        cos_top = np.cos(self.getPropAngle(0))
        cos_bottom = np.cos(self.getPropAngle(bottom_index))

        # Up-down direction
        # Determine the value of p according to the polarization
        if self.getPolarization() == 'TE':
            p_i = n_top * cos_top
            p_l = n_bottom * cos_bottom
        else:
            p_i = cos_top / n_top
            p_l = cos_bottom / n_bottom

        # Calculate the coefficients
        m11 = charMatrixUD[0, 0]
        m12 = charMatrixUD[0, 1]
        m21 = charMatrixUD[1, 0]
        m22 = charMatrixUD[1, 1]
        a = (m11 + m12 * p_l) * p_i
        b = (m21 + m22 * p_l)

        r = (a - b) / (a + b)
        reflectivity = np.absolute(r) ** 2

        # Attention: in the case of TM waves, the coefficients r and t
        # refer to the ratio of the reflected (transmitted) MAGNETIC
        # field to the incident MAGNETIC field. r is in fact equal to
        # the ratio of the electric field amplitudes, but t must be
        # modified to put it in terms of the electric field.
        if self.getPolarization() == 'TE':
            t = 2 * p_i / (a + b)
        else:
            t = (n_top / n_bottom) * 2 * p_i / (a + b)
            p_i = n_top * cos_top
            p_l = n_bottom * cos_bottom

        # Note that, when p_l or p_i are complex (for instance because
        # we are beyond the critical angle or because the medium has
        # nonzero extintion coefficient) the transmittivity will be a
        # complex number.
        transmittivity = np.absolute(t) ** 2 * p_l / p_i

        self.__coefficientsUpDown['r'] = r
        self.__coefficientsUpDown['t'] = t
        self.__coefficientsUpDown['R'] = reflectivity
        self.__coefficientsUpDown['T'] = transmittivity

        # Down-up direction
        # Determine the value of p according to the polarization
        if self.getPolarization() == 'TE':
            p_i = n_bottom * cos_bottom
            p_l = n_top * cos_top
        else:
            p_i = cos_bottom / n_bottom
            p_l = cos_top / n_top

        # Calculate the coefficients
        m11 = charMatrixDU[0, 0]
        m12 = charMatrixDU[0, 1]
        m21 = charMatrixDU[1, 0]
        m22 = charMatrixDU[1, 1]
        a = (m11 + m12 * p_l) * p_i
        b = (m21 + m22 * p_l)

        r = (a - b) / (a + b)
        reflectivity = np.absolute(r) ** 2

        # Attention: in the case of TM waves, the coefficients r and t
        # refer to the ratio of the reflected (transmitted) MAGNETIC
        # field to the incident MAGNETIC field. r is in fact equal to
        # the ratio of the electric field amplitudes, but t must be
        # modified to put it in terms of the electric field.
        if self.getPolarization() == 'TE':
            t = 2 * p_i / (a + b)
        else:
            t = (n_bottom / n_top) * 2 * p_i / (a + b)
            p_i = n_bottom * cos_bottom
            p_l = n_top * cos_top

        # Note that, when p_l or p_i are complex (for instance because
        # we are beyond the critical angle or because the medium has
        # nonzero extintion coefficient) the transmittivity will be a
        # complex number.
        transmittivity = np.absolute(t) ** 2 * p_l / p_i

        self.__coefficientsDownUp['r'] = r
        self.__coefficientsDownUp['t'] = t
        self.__coefficientsDownUp['R'] = reflectivity
        self.__coefficientsDownUp['T'] = transmittivity

    def getCharMatrixUpDown(self):
        """
        This method returns the characteristic matrix in the up-down
        direction.

        Returns
        -------
        out : numpy.ndarray
            The characteristic matrix of the system in the top-down
            direction of propagation.
        """

        return self.__charMatrixUpDown

    def getCharMatrixDownUp(self):
        """
        This method returns the characteristic matrix in the down-up
        direction

        Returns
        -------
        out : numpy.ndarray
            The characteristic matrix of the system in the down-top
            direction of propagation.
        """

        return self.__charMatrixDownUp

    def getCoefficientsUpDown(self):
        """
        This method returns a dictionary with the reflection and
        transmission coefficients, the reflectance and the transmittance
        of the multilayer system.in the up-down direction of
        propagation.

        Returns
        -------
        out : dictionary
            A dictionary with the reflection and transmission
            coefficients, the reflectance and the transmittance of the
            multilayer system. The keys are {'r', 't', 'R', 'T'}.
        """

        return self.__coefficientsUpDown

    def getCoefficientsDownUp(self):
        """
        This method returns a dictionary with the reflection and
        transmission coefficients, the reflectance and the transmittance
        of the multilayer system.in the down-up direction of
        propagation.

        Returns
        -------
        out : dictionary
            A dictionary with the reflection and transmission
            coefficients, the reflectance and the transmittance of the
            multilayer system. The keys are {'r', 't', 'R', 'T'}.
        """

        return self.__coefficientsDownUp

    def calculateFx(self, z, wlength, angle, index=0):
        """
        Calculates Fx(z; lambda, theta) of the multilayer.

        The direction x is parallel to both the layer interfaces and the
        plane of incidence of the light (or the direction of the
        intersection between plane of incidence and interfaces). The
        plane of incidence is always perpendicular to the interfaces.
        The direction z is perpendicular to the interfaces.

        The state of the multilayer will be changed according to the
        parameters passed to the method and then F(z) will be
        calculated.

        Fx is defined only for TM waves. If the multilayer is currently
        in TE, it will be changed to TM to perform the calculations.

        Parameters
        ----------
        z : float
            The z coordinate of the emitting dipole.
        wlength : float
            The wavelength of the light across the multilayer In the
            same units as in the file from which the refractive
            indices were loaded.
        angle : float
            The propagation angle in radians
        index : int
            The index of the layer where we are fixing the propagation
            angle.

        Returns
        -------
        out : complex128
            The value of Fx(z, lambda, angle)
        """

        # Determine what has to be changed and wether or not to update
        # the matrices
        if self.getPolarization() != 'TM':
            self.setPolarization('TM')
        if wlength != self.getWlength():
            self.setWlength(wlength)
            self.setPropAngle(angle, index)
        if self.getPropAngle(index) != angle:
            self.setPropAngle(angle, index)
        if self.getCharMatrixUpDown() == None:
            self.calcMatrices()
            self.updateCharMatrix()

        # Calculate Fx(z)
        # Find out in which layer the dipole is located
        dipole_layer_index = self.getIndexAtPos(z)

        # Calculate Fx according to the position of the dipole
        if dipole_layer_index == 0:
            # Fx(z) in case the dipole is in the top medium
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            n0 = self.getRefrIndex(0)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            r01 = self.getCoefficientsUpDown()['r']

            # Calculate function
            fx = 1 - r01 * np.exp(2 * eta0 * (z - z0) * 1j)

        elif dipole_layer_index == self.numLayers() - 1:
            # Fx(z) in case the dipole is in the bottom medium
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            thetaN = self.getPropAngle(dipole_layer_index)
            n0 = self.getRefrIndex(0)
            nN = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaN = 2 * np.pi * np.sqrt(nN ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            t1N = self.getCoefficientsUpDown()['t']

            # Calculate function. We handle separately the case
            # where theta0 is 0 to avoid a NaN result. Bear in
            # mind that if we have a dipole oscilating along x
            # there is no light propagating along x.
            if theta0 == np.pi / 2:
                fx = 1 + 0j
            else:
                fx = t1N * np.exp(eta0 * (z - z0) * 1j - etaN * z * 1j) * \
                        np.cos(thetaN) / np.cos(theta0)

        else:
            # Fx(z) in case the dipole is within any of the layers
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            thetaj = self.getPropAngle(dipole_layer_index)
            n0 = self.getRefrIndex(0)
            nj = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            zj = self.getPosition(dipole_layer_index)
            zj1 = self.getPosition(dipole_layer_index - 1)
            dj = self.getThickness(dipole_layer_index)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaj = 2 * np.pi * np.sqrt(nj ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients. We have to build some
            # submultilayers first.

            # Submultilayer from the top medium to dipole_layer_index.
            rilist = [self.getRefrIndex(0)]
            alist = [self.getPropAngle(0)]
            layers = [self.__stack[0]['medium']]
            for index in range(1, dipole_layer_index):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[dipole_layer_index]['medium'])
            rilist.append(self.getRefrIndex(dipole_layer_index))
            alist.append(self.getPropAngle(dipole_layer_index))
            sub_above = Multilayer(layers)
            sub_above.setWlength(wavelength, rilist)
            sub_above.setPropAngle(alist)
            sub_above.setPolarization('TM')
            sub_above.calcMatrices()
            sub_above.updateCharMatrix()

            # Submultilayer from dipole_layer_index to the bottom
            # medium.
            rilist = [self.getRefrIndex(dipole_layer_index)]
            alist = [self.getPropAngle(dipole_layer_index)]
            layers = [self.__stack[dipole_layer_index]['medium']]
            for index in range(dipole_layer_index + 1,
                    self.numLayers() - 1):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[self.numLayers() - 1]['medium'])
            rilist.append(self.getRefrIndex(self.numLayers() - 1))
            alist.append(self.getPropAngle(self.numLayers() - 1))
            sub_below = Multilayer(layers)
            sub_below.setWlength(wavelength, rilist)
            sub_below.setPropAngle(alist)
            sub_below.setPolarization(self.getPolarization())
            sub_below.calcMatrices()
            sub_below.updateCharMatrix()

            # Now we can retreive the relevant coefficients
            t1j = sub_above.getCoefficientsUpDown()['t']
            rjjp1 = sub_below.getCoefficientsUpDown()['r']
            rjjm1 = sub_above.getCoefficientsDownUp()['r']

            # Calculate function. We handle separately the case
            # where theta0 is 0 to avoid a NaN result. Bear in
            # mind that if we have a dipole oscilating along x
            # there is no light propagating along x.
            if theta0 == np.pi / 2:
                fx = 1 + 0j
            else:
                numerator = t1j * \
                        (1 - rjjp1 * np.exp(2 * etaj * (z - zj) * 1j))
                denominator = \
                        1 - rjjp1 * rjjm1 * np.exp(2 * etaj * dj * 1j)
                factor = np.exp(eta0 * (z - z0) * 1j - etaj * \
                        (z - zj1) * 1j) * np.cos(thetaj) / np.cos(theta0)
                fx = numerator * factor / denominator

        return np.complex128(fx)

    def calculateFy(self, z, wlength, angle, index=0):
        """
        Calculates Fy(z) of the multilayer.

        The direction y is parallel to the layer interfaces and
        perpendicular to the plane of incidence. The plane of incidence
        is always perpendicular to the interfaces. The direction z is
        perpendicular to the interfaces.

        The state of the multilayer will be changed according to the
        parameters passed to the method and then F(z) will be
        calculated.

        Fy is defined only for TE waves. If the multilayer is currently
        in TM, it will be changed to TE to perform the calculations.

        Parameters
        ----------
        z : float
            The z coordinate of the emitting dipole.
        wlength : float
            The wavelength of the light across the multilayer In the
            same units as in the file from which the refractive indices
            were loaded.
        angle : float
            The propagation angle in radians
        index : int
            The index of the layer where we are fixing the propagation
            angle.

        Returns
        -------
        out : complex128
            The value of Fy(z, lambda, angle)
        """

        # Determine what has to be changed and update matrices
        if self.getPolarization() != 'TE':
            self.setPolarization('TE')
        if wlength != self.getWlength():
            self.setWlength(wlength)
            self.setPropAngle(angle, index)
        if self.getPropAngle(index) != angle:
            self.setPropAngle(angle, index)
        if self.getCharMatrixUpDown() == None:
            self.calcMatrices()
            self.updateCharMatrix()

        # Calculate Fy(z)
        # Find out in which layer the dipole is located
        dipole_layer_index = self.getIndexAtPos(z)

        # Calculate Fy according to the position of the dipole
        if dipole_layer_index == 0:
            # Fy(z) in case the dipole is in the top medium
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            n0 = self.getRefrIndex(0)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            r01 = self.getCoefficientsUpDown()['r']

            # Calculate function
            fy = 1 + r01 * np.exp(2 * eta0 * (z - z0) * 1j)

        elif dipole_layer_index == self.numLayers() - 1:
            # Fy(z) in case the dipole is in the bottom
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            n0 = self.getRefrIndex(0)
            nN = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaN = 2 * np.pi * np.sqrt(nN ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            t1N = self.getCoefficientsUpDown()['t']

            # Calculate function
            fy = t1N * np.exp(eta0 * (z - z0) * 1j - etaN * z * 1j)

        else:
            # Fy(z) in case the dipole is within any of the layers
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            n0 = self.getRefrIndex(0)
            nj = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            zj = self.getPosition(dipole_layer_index)
            zj1 = self.getPosition(dipole_layer_index - 1)
            dj = self.getThickness(dipole_layer_index)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaj = 2 * np.pi * np.sqrt(nj ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients. We have to build some
            # submultilayers first

            # Submultilayer from the top medium to dipole_layer_index
            rilist = [self.getRefrIndex(0)]
            alist = [self.getPropAngle(0)]
            layers = [self.__stack[0]['medium']]
            for index in range(1, dipole_layer_index):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[dipole_layer_index]['medium'])
            rilist.append(self.getRefrIndex(dipole_layer_index))
            alist.append(self.getPropAngle(dipole_layer_index))
            sub_above = Multilayer(layers)
            sub_above.setWlength(wavelength, rilist)
            sub_above.setPropAngle(alist)
            sub_above.setPolarization(self.getPolarization())
            sub_above.calcMatrices()
            sub_above.updateCharMatrix()

            # Submultilayer from dipole_layer_index to the bottom
            # medium.
            rilist = [self.getRefrIndex(dipole_layer_index)]
            alist = [self.getPropAngle(dipole_layer_index)]
            layers = [self.__stack[dipole_layer_index]['medium']]
            for index in range(dipole_layer_index + 1,
                    self.numLayers() - 1):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[self.numLayers() - 1]['medium'])
            rilist.append(self.getRefrIndex(self.numLayers() - 1))
            alist.append(self.getPropAngle(self.numLayers() - 1))
            sub_below = Multilayer(layers)
            sub_below.setWlength(wavelength, rilist)
            sub_below.setPropAngle(alist)
            sub_below.setPolarization(self.getPolarization())
            sub_below.calcMatrices()
            sub_below.updateCharMatrix()

            # Now we can retreive the relevant coefficients
            t1j = sub_above.getCoefficientsUpDown()['t']
            rjjp1 = sub_below.getCoefficientsUpDown()['r']
            rjjm1 = sub_above.getCoefficientsDownUp()['r']

            # Calculate function
            numerator = t1j * \
                    (1 + rjjp1 * np.exp(2 * etaj * (z - zj) * 1j))
            denominator = 1 - \
                    rjjp1 * rjjm1 * np.exp(2 * etaj * dj * 1j)
            factor = np.exp(eta0 * (z - z0) * 1j - etaj * (z - zj1) * 1j)
            fy = numerator * factor / denominator

        return np.complex128(fy)

    def calculateFz(self, z, wlength, angle, index=0):
        """
        Calculates Fz(z) of the multilayer.

        The direction z is perpendicular to the interfaces.

        The state of the multilayer will be changed according to the
        parameters passed to the method and then F(z) will be
        calculated.

        Fz is defined only for TM waves. If the multilayer is currently
        in TE, it will be changed to TM to perform the calculations.

        Parameters
        ----------
        z : float
            The z coordinate of the emitting dipole.
        wlength : float
            The wavelength of the light across the multilayer. In the
            same units as in the file from which the refractive
            indices were loaded.
        angle : float
            The propagation angle in radians.
        index : int
            The index of the layer where we are fixing the propagation
            angle.

        Returns
        -------
        out : complex128
            The value of Fz(z, lambda, angle)
        """

        # Calculate Fz(z)
        # Determine what has to be changed and update matrices
        if self.getPolarization() != 'TM':
            self.setPolarization('TM')
        if wlength != self.getWlength():
            self.setWlength(wlength)
            self.setPropAngle(angle, index)
        if self.getPropAngle(index) != angle:
            self.setPropAngle(angle, index)
        if self.getCharMatrixUpDown() == None:
            self.calcMatrices()
            self.updateCharMatrix()

        # Find out in which layer the dipole is located
        dipole_layer_index = self.getIndexAtPos(z)

        # Calculate Fz according to the position of the dipole
        if dipole_layer_index == 0:
            # Fz(z) in case the dipole is in the top medium
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            n0 = self.getRefrIndex(0)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            r01 = self.getCoefficientsUpDown()['r']

            # Calculate function
            fz = 1 + r01 * np.exp(2 * eta0 * (z - z0) * 1j)

        elif dipole_layer_index == self.numLayers() - 1:
            # Fz(z) in case the dipole is in the bottom medium
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            thetaN = self.getPropAngle(dipole_layer_index)
            n0 = self.getRefrIndex(0)
            nN = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaN = 2 * np.pi * np.sqrt(nN ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients
            t1N = self.getCoefficientsUpDown()['t']

            # Calculate function. We handle separately the case
            # where theta0 is one to avoid a NaN result. Bear in
            # mind that if we have a dipole oscilating along z
            # there is no light propagating along z.
            if theta0 == 0:
                fz = 1 + 0j
            else:
                fz = t1N * np.exp(eta0 * (z - z0) * 1j - etaN * z * 1j) * \
                    np.sin(thetaN) / np.sin(theta0)

        else:
            # Fz(z) in case the dipole is within any of the layers
            wavelength = self.getWlength()
            theta0 = self.getPropAngle(0)
            thetaj = self.getPropAngle(dipole_layer_index)
            n0 = self.getRefrIndex(0)
            nj = self.getRefrIndex(dipole_layer_index)

            # Calculate parameters
            z0 = self.getPosition(0)
            zj = self.getPosition(dipole_layer_index)
            zj1 = self.getPosition(dipole_layer_index - 1)
            dj = self.getThickness(dipole_layer_index)
            eta0 = 2 * np.pi * np.sqrt(n0 ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength
            etaj = 2 * np.pi * np.sqrt(nj ** 2 - (n0 * np.sin(theta0)) ** 2) \
                    / wavelength

            # Retreive coefficients. We have to build some
            # submultilayers first.
            
            # Submultilayer from the top medium to dipole_layer_index.
            rilist = [self.getRefrIndex(0)]
            alist = [self.getPropAngle(0)]
            layers = [self.__stack[0]['medium']]
            for index in range(1, dipole_layer_index):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[dipole_layer_index]['medium'])
            rilist.append(self.getRefrIndex(dipole_layer_index))
            alist.append(self.getPropAngle(dipole_layer_index))
            sub_above = Multilayer(layers)
            sub_above.setWlength(wavelength, rilist)
            sub_above.setPropAngle(alist)
            sub_above.setPolarization(self.getPolarization())
            sub_above.calcMatrices()
            sub_above.updateCharMatrix()

            # Submultilayer from the dipole_layer_index to the bottom
            # medium.
            rilist = [self.getRefrIndex(dipole_layer_index)]
            alist = [self.getPropAngle(dipole_layer_index)]
            layers = [self.__stack[dipole_layer_index]['medium']]
            for index in range(dipole_layer_index + 1,
                    self.numLayers() - 1):
                layers.append([self.__stack[index]['medium'],
                        self.getThickness(index)])
                rilist.append(self.getRefrIndex(index))
                alist.append(self.getPropAngle(index))
            layers.append(self.__stack[self.numLayers() - 1]['medium'])
            rilist.append(self.getRefrIndex(self.numLayers() - 1))
            alist.append(self.getPropAngle(self.numLayers() - 1))
            sub_below = Multilayer(layers)
            sub_below.setWlength(wavelength, rilist)
            sub_below.setPropAngle(alist)
            sub_below.setPolarization(self.getPolarization())
            sub_below.calcMatrices()
            sub_below.updateCharMatrix()

            # Now we can retreive the relevant coefficients
            t1j = sub_above.getCoefficientsUpDown()['t']
            rjjp1 = sub_below.getCoefficientsUpDown()['r']
            rjjm1 = sub_above.getCoefficientsDownUp()['r']

            # Calculate function. We handle separately the case
            # where theta0 is one to avoid a NaN result. Bear in
            # mind that if we have a dipole oscilating along z
            # there is no light propagating along z.
            if theta0 == 0:
                fz = 1 + 0j
            else:
                numerator = t1j * \
                        (1 + rjjp1 * np.exp(2 * etaj * (z - zj) * 1j))
                denominator = 1 - rjjp1 * rjjm1 * \
                        np.exp(2 * etaj * dj * 1j)
                factor = np.exp(
                        eta0 * (z - z0) * 1j - etaj * (z - zj1) * 1j) * \
                        np.sin(thetaj) / np.sin(theta0)
                fz = numerator * factor / denominator

        return np.complex128(fz)
