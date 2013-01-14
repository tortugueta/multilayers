# -*- coding: utf-8 -*-

import unittest
import multilayers as ml
import numpy as np
from os import remove


class TestMedium(unittest.TestCase):
    """
    Test the Medium class.
    """

    def setUp(self):
        """
        Prepare the ground with common data for the tests that follow.
        """

        # Create two simple temporary files that we will use as our
        # source of data for the tested mediums.
        comstr = "%BogusCol0\tn\tk\twlength\n%Comment2\n"
        datastr = \
                " 1.0\t2.0\t0.030\t200\n" + \
                " NaN\t3.1\t0.025\t250\n" + \
                " 1.2\t4.2\t0.020\t300\n" + \
                " 9.9\t4.0\t0.010\t400\n" + \
                "-3.0\t3.8\t0.050\t500\n" + \
                "-1.0\t3.7\t0.000\t600\n" + \
                "-5  \t3.7\t0.000\t700\n"
        fhandle = open("testmedium1.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#BogusCol0%n%k%wlength%#Comment2\n"
        datastr = \
                "300%1.0%1.030\n" + \
                "350%2.1%1.025\n" + \
                "400%3.2%1.020\n" + \
                "500%3.0%1.010\n" + \
                "600%2.8%0.000\n" + \
                "700%2.7%1.000\n" + \
                "800%2.7%1.000\n"
        fhandle = open("testmedium2.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        self.medium1 = ml.Medium("testmedium1.txt", '%', usecols=[3, 1, 2])
        self.medium2 = ml.Medium("testmedium2.txt", delimiter='%')

    def tearDown(self):
        """
        Clean up.
        """

        remove("testmedium1.txt")
        remove("testmedium2.txt")

    def test_init(self):
        """
        Test the init method
        """

        # Test for instantiation with wrong arguments
        self.assertRaises(IOError, ml.Medium, "")
        self.assertRaises(IOError, ml.Medium, "nonexistent.dat")
        self.assertRaises(TypeError, ml.Medium, "testmedium1.txt", 4)
        self.assertRaises(TypeError, ml.Medium, "testmedium1.txt", '#', 0)
        self.assertRaises(AttributeError, ml.Medium, "testmedium1.txt", '#',
                '\t', 9)
        self.assertRaises(IOError, ml.Medium, "testmedium1.txt", '#', '\t',
                None, 10)
        self.assertRaises(TypeError, ml.Medium, "testmedium1.txt", '#', '\t',
                None, 'wrong type')
        self.assertRaises(TypeError, ml.Medium, "testmedium1.txt", '#', '\t',
                None, 0, 0)
        self.assertRaises(TypeError, ml.Medium, "testmedium1.txt", '#', '\t',
                None, 0, ['w', 't'])

    def test_getMinMaxWlengths(self):
        """
        Test the getMinMaxWlengths method.
        """

        min_m1, max_m1 = self.medium1.getMinMaxWlength()
        min_m2, max_m2 = self.medium2.getMinMaxWlength()

        # Check that we are getting numpy floats
        self.assertTrue(isinstance(min_m1, np.float64))
        self.assertTrue(isinstance(max_m1, np.float64))
        self.assertTrue(isinstance(min_m2, np.float64))
        self.assertTrue(isinstance(max_m2, np.float64))

        # Check that we get the wavelength range correctly
        self.assertEqual(min_m1, 200)
        self.assertEqual(max_m1, 700)
        self.assertEqual(min_m2, 300)
        self.assertEqual(max_m2, 800)

    def test_getRefrIndex(self):
        """
        Test the getRefrIndex method.
        """

        # Check that trying to calculate the refractive index outside
        # the valid interpolating range raises the proper exception.
        self.assertRaises(ValueError, self.medium1.getRefrIndex, 199.9)
        self.assertRaises(ValueError, self.medium1.getRefrIndex, 700.01)
        self.assertRaises(ValueError, self.medium2.getRefrIndex, 250)
        self.assertRaises(ValueError, self.medium2.getRefrIndex, 850)

        # When interpolating at the wavelenghts where we actually had
        # the refractive index sampled we expect to recover the exact
        # value found in the file. Allow for roundoff errors.
        self.assertAlmostEqual(self.medium1.getRefrIndex(400), 4 + 0.01j, 14)
        self.assertAlmostEqual(self.medium2.getRefrIndex(500), 3 + 1.01j, 14)


class TestMultilayer(unittest.TestCase):
    """
    Test the multilayer class.
    """

    def setUp(self):
        """
        Prepare the ground with common data for the tests that follow.
        """

        # Create two simple temporary files that we will use as our
        # source of data for the tested mediums.
        comstr = "%BogusCol0\tn\tk\twlength\n%Comment2\n"
        datastr = \
                " 1.0\t2.0\t0.030\t200\n" + \
                " NaN\t3.1\t0.025\t250\n" + \
                " 1.2\t4.2\t0.020\t300\n" + \
                " 9.9\t4.0\t0.010\t400\n" + \
                "-3.0\t3.8\t0.050\t500\n" + \
                "-1.0\t3.7\t0.000\t600\n" + \
                "-5  \t3.7\t0.000\t700\n"
        fhandle = open("testmedium1.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#BogusCol0%n%k%wlength%#Comment2\n"
        datastr = \
                "300%1.0%1.030\n" + \
                "350%2.1%1.025\n" + \
                "400%3.2%1.020\n" + \
                "500%3.0%1.010\n" + \
                "600%2.8%0.000\n" + \
                "700%2.7%1.000\n" + \
                "800%2.7%1.000\n"
        fhandle = open("testmedium2.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#Outcast file\n"
        datastr = \
                " 1.0\t2.0\t0.030\t900\n" + \
                " NaN\t3.1\t0.025\t910\n" + \
                " 1.2\t4.2\t0.020\t920\n" + \
                " 9.9\t4.0\t0.010\t930\n" + \
                "-3.0\t3.8\t0.050\t940\n" + \
                "-1.0\t3.7\t0.000\t950\n" + \
                "-5  \t3.7\t0.000\t960\n"
        fhandle = open("testmedium3.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#BogusCol0%n%k%wlength%#Comment2\n"
        datastr = \
                "300%1.0%0.000\n" + \
                "350%1.1%0.000\n" + \
                "400%1.2%0.000\n" + \
                "500%1.0%0.000\n" + \
                "600%1.8%0.000\n" + \
                "700%1.7%0.000\n" + \
                "800%1.7%0.000\n"
        fhandle = open("testmedium4.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#Crawford system: silver\n#wl\tn\tk"
        datastr = \
                "300;0.0427;3.3988\n" + \
                "350;0.0427;3.3988\n" + \
                "400;0.0427;3.3988\n" + \
                "500;0.0427;3.3988\n" + \
                "600;0.0427;3.3988\n" + \
                "700;0.0427;3.3988\n" + \
                "800;0.0427;3.3988\n"
        fhandle = open("cs_silver.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#Crawford system: dielectric\n#wl\tn\tk"
        datastr = \
                "300;1.45;0.0\n" + \
                "350;1.45;0.0\n" + \
                "400;1.45;0.0\n" + \
                "500;1.45;0.0\n" + \
                "600;1.45;0.0\n" + \
                "700;1.45;0.0\n" + \
                "800;1.45;0.0\n"
        fhandle = open("cs_dielectric.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        comstr = "#Crawford system: ambient\n#wl\tn\tk"
        datastr = \
                "300;1.0;0.0\n" + \
                "350;1.0;0.0\n" + \
                "400;1.0;0.0\n" + \
                "500;1.0;0.0\n" + \
                "600;1.0;0.0\n" + \
                "700;1.0;0.0\n" + \
                "800;1.0;0.0\n"
        fhandle = open("cs_ambient.txt", "w")
        fhandle.write(comstr + datastr)
        fhandle.close()

        self.medium1 = ml.Medium("testmedium1.txt", '%', usecols=[3, 1, 2])
        self.medium2 = ml.Medium("testmedium2.txt", delimiter='%')
        self.medium3 = ml.Medium("testmedium3.txt", usecols=[3, 1, 2])
        self.medium4 = ml.Medium("testmedium4.txt", delimiter='%')
        self.cs_silver = ml.Medium("cs_silver.txt", delimiter=';')
        self.cs_dielectric = ml.Medium("cs_dielectric.txt", delimiter=';')
        self.cs_ambient = ml.Medium("cs_ambient.txt", delimiter=';')

        # Create a few multilayers for the test
        self.zerothick = ml.Multilayer([
                self.medium1,
                self.medium2,
                self.medium1])
        self.mlminimum = ml.Multilayer([
                self.medium1,
                self.medium2])
        self.ml2layers = ml.Multilayer([
                self.medium1,
                [self.medium2, 10],
                [self.medium1, 20],
                self.medium2])
        self.mlsame = ml.Multilayer([
                self.medium1,
                [self.medium1, 5],
                [self.medium1, 5],
                self.medium1])
        self.mlminsame = ml.Multilayer([
                self.medium1,
                self.medium1])
        self.symmetry = ml.Multilayer([
                self.medium1,
                [self.medium2, 300],
                self.medium4])
        self.cssystem_f2_film = ml.Multilayer([
                self.cs_ambient,
                [self.cs_dielectric, 300],
                self.cs_silver])
        self.cssystem_f2_nofilm = ml.Multilayer([
                self.cs_ambient,
                self.cs_silver])
        self.cssystem_f3_film = ml.Multilayer([
                self.cs_ambient,
                [self.cs_dielectric, 200],
                self.cs_silver])
        self.cssystem_f3_nofilm = ml.Multilayer([
                self.cs_ambient,
                self.cs_silver])
        self.cssystem_f4_film = ml.Multilayer([
                self.cs_dielectric,
                [self.cs_ambient, 200],
                self.cs_silver])
        self.cssystem_f4_nofilm = ml.Multilayer([
                self.cs_dielectric,
                self.cs_silver])
        self.cssystem_f5_film = ml.Multilayer([
                self.cs_ambient,
                [self.cs_dielectric, 50],
                self.cs_silver])
        self.cssystem_f5_nofilm = ml.Multilayer([
                self.cs_ambient,
                self.cs_silver])
        self.cssystem_f5_film_alt = ml.Multilayer([
                self.cs_ambient,
                [self.cs_dielectric, 25],
                [self.cs_dielectric, 25],
                self.cs_silver])

    def tearDown(self):
        """
        Clean up.
        """

        remove("testmedium1.txt")
        remove("testmedium2.txt")
        remove("testmedium3.txt")
        remove("testmedium4.txt")
        remove("cs_silver.txt")
        remove("cs_dielectric.txt")
        remove("cs_ambient.txt")

    def test_init(self):
        """
        Test the init method of the multilayer.
        """

        # Check that the correct exceptions are raised when the
        # multilayer is instantiated with the wrong arguments.
        self.assertRaises(TypeError, ml.Multilayer)
        self.assertRaises(TypeError, ml.Multilayer, 1)
        self.assertRaises(ValueError, ml.Multilayer, [])
        self.assertRaises(ValueError, ml.Multilayer, ())
        self.assertRaises(ValueError, ml.Multilayer, [self.medium1])
        self.assertRaises(TypeError, ml.Multilayer, (self.medium1))
        self.assertRaises(TypeError, ml.Multilayer, [self.medium1, 0])
        self.assertRaises(TypeError, ml.Multilayer, ('string', self.medium1))
        self.assertRaises(TypeError, ml.Multilayer,
                [[self.medium1], self.medium2])
        self.assertRaises(TypeError, ml.Multilayer,
                (self.medium1, [self.medium1, 10]))
        self.assertRaises(TypeError, ml.Multilayer,
                [self.medium1, [self.medium2], self.medium1])
        self.assertRaises(TypeError, ml.Multilayer,
                (self.medium1, [self.medium2], self.medium1))
        self.assertRaises(TypeError, ml.Multilayer,
                [self.medium1, [0, self.medium2], self.medium1])
        self.assertRaises(TypeError, ml.Multilayer,
                (self.medium1, [0, self.medium2], self.medium1))
        self.assertRaises(TypeError, ml.Multilayer,
                [self.medium1, [self.medium2, 0, 0], self.medium1])
        self.assertRaises(TypeError, ml.Multilayer,
                (self.medium1, [self.medium2, 0, 0], self.medium1))
        self.assertRaises(ValueError, ml.Multilayer,
                [self.medium1, [self.medium2, -1], self.medium1])
        self.assertRaises(ValueError, ml.Multilayer,
                (self.medium1, [self.medium2, -2], self.medium1))
        self.assertRaises(ValueError, ml.Multilayer,
                [self.medium1, [self.medium2, 'str'], self.medium1])
        self.assertRaises(ValueError, ml.Multilayer,
                (self.medium1, [self.medium2, 'str'], self.medium1))
        self.assertRaises(TypeError, ml.Multilayer,
                [self.medium1, {2: 'a', 3: 'b'}, self.medium1])

    def test_getPosition(self):
        """
        Test the getPosition method.
        """

        # Test that we can't access a layer beyond the last or use a
        # negative index.
        self.assertRaises(IndexError, self.mlminimum.getPosition, 2)
        self.assertRaises(IndexError, self.mlminimum.getPosition, -1)

        # Test that we get an output of the right type
        self.assertEqual(type(self.mlminimum.getPosition(0)), float)
        self.assertEqual(type(self.mlminimum.getPosition(1)), float)
        self.assertEqual(type(self.ml2layers.getPosition(0)), float)
        self.assertEqual(type(self.ml2layers.getPosition(1)), float)
        self.assertEqual(type(self.ml2layers.getPosition(2)), float)
        self.assertEqual(type(self.ml2layers.getPosition(3)), float)

    def test_calcPositions(self):
        """
        Once the getPosition mehtod has been tested, check that the
        positions are calculated correctly by the calcPositions method.
        """

        self.assertEqual(self.mlminimum.getPosition(0), 0.0)
        self.assertEqual(self.mlminimum.getPosition(1), -np.infty)
        self.assertEqual(self.ml2layers.getPosition(0), 30.0)
        self.assertEqual(self.ml2layers.getPosition(1), 20.0)
        self.assertEqual(self.ml2layers.getPosition(2), 0.0)
        self.assertEqual(self.ml2layers.getPosition(3), -np.infty)

    def test_thickness(self):
        """
        Test that the thicknesses are stored correctly during
        instantiation and have the correct type.
        """

        self.assertEqual(type(self.mlminimum.getThickness(0)), float)
        self.assertEqual(type(self.mlminimum.getThickness(1)), float)
        self.assertEqual(type(self.ml2layers.getThickness(0)), float)
        self.assertEqual(type(self.ml2layers.getThickness(1)), float)
        self.assertEqual(type(self.ml2layers.getThickness(2)), float)
        self.assertEqual(type(self.ml2layers.getThickness(3)), float)

        self.assertEqual(self.mlminimum.getThickness(0), np.infty)
        self.assertEqual(self.mlminimum.getThickness(1), np.infty)
        self.assertEqual(self.ml2layers.getThickness(0), np.infty)
        self.assertEqual(self.ml2layers.getThickness(1), 10.0)
        self.assertEqual(self.ml2layers.getThickness(2), 20.0)
        self.assertEqual(self.ml2layers.getThickness(3), np.infty)
        self.assertEqual(self.zerothick.getThickness(1), 0.0)

        # Check that a negative index is rejected
        self.assertRaises(IndexError, self.ml2layers.getThickness, -1)

    def test_setThickness(self):
        """
        We want to make sure that after calling the setThickness method
        everything is updated as expected.
        """

        # First check that we cannot change the thickness of the
        # top and bottom mediums or supply a negative index.
        self.assertRaises(IndexError, self.mlminimum.setThickness, 0, 0)
        self.assertRaises(IndexError, self.mlminimum.setThickness, 0, 1)
        self.assertRaises(IndexError, self.ml2layers.setThickness, 0, 0)
        self.assertRaises(IndexError, self.ml2layers.setThickness, 0, 3)
        self.assertRaises(IndexError, self.ml2layers.setThickness, 0, -1)

        # Negative thicknesses do not make sense
        self.assertRaises(ValueError, self.ml2layers.setThickness, -10.0, 1)

        # Check that the change is stored properly and the new
        # positions calculated accordingly.
        self.ml2layers.setThickness(5, 2)
        self.assertEqual(self.ml2layers.getThickness(3), np.infty)
        self.assertEqual(self.ml2layers.getThickness(2), 5.0)
        self.assertEqual(self.ml2layers.getThickness(1), 10.0)
        self.assertEqual(self.ml2layers.getThickness(0), np.infty)
        self.assertEqual(self.ml2layers.getPosition(3), -np.infty)
        self.assertEqual(self.ml2layers.getPosition(2), 0.0)
        self.assertEqual(self.ml2layers.getPosition(1), 5.0)
        self.assertEqual(self.ml2layers.getPosition(0), 15.0)

        self.ml2layers.setThickness(50, 2)
        self.ml2layers.setThickness(100, 1)
        self.assertEqual(self.ml2layers.getThickness(3), np.infty)
        self.assertEqual(self.ml2layers.getThickness(2), 50.0)
        self.assertEqual(self.ml2layers.getThickness(1), 100)
        self.assertEqual(self.ml2layers.getThickness(0), np.infty)
        self.assertEqual(self.ml2layers.getPosition(3), -np.infty)
        self.assertEqual(self.ml2layers.getPosition(2), 0.0)
        self.assertEqual(self.ml2layers.getPosition(1), 50)
        self.assertEqual(self.ml2layers.getPosition(0), 150)

    def test_getMinMaxWlengths(self):
        """
        Test the getMinMaxWlength method.
        """

        range1 = self.mlminimum.getMinMaxWlength()
        range2 = self.ml2layers.getMinMaxWlength()
        min1, max1 = range1
        min2, max2 = range2

        # Check that it returns a tuple with two floats
        self.assertEqual(type(range1), tuple)
        self.assertEqual(type(range2), tuple)
        self.assertEqual(type(min1), np.float64)
        self.assertEqual(type(max1), np.float64)
        self.assertEqual(type(min2), np.float64)
        self.assertEqual(type(max2), np.float64)

        # Check that the minimum is smaller than the maximum
        self.assertTrue(min1 < max1)
        self.assertTrue(min2 < max2)

        # Check that we get the expected result
        self.assertEqual(min1, 300.0)
        self.assertEqual(max1, 700.0)
        self.assertEqual(min2, 300.0)
        self.assertEqual(max2, 700.0)

        # If there is no intersection we have a problem
        self.assertRaises(ValueError, ml.Multilayer,
                [self.medium1, self.medium3])

    def test_numLayers(self):
        """
        Test the numLayers method
        """

        # Check that the return value is an int and that we get the
        # expected result.
        numlayers1 = self.mlminimum.numLayers()
        numlayers2 = self.ml2layers.numLayers()
        self.assertEqual(type(numlayers1), int)
        self.assertEqual(type(numlayers2), int)
        self.assertEqual(numlayers1, 2)
        self.assertEqual(numlayers2, 4)

    def test_getIndexAtPos(self):
        """
        Test the getIndexAtPos method.
        """

        # Check that the output is an int and the results are correct
        self.assertEqual(type(self.ml2layers.getIndexAtPos(-1)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(-1), 3)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(0)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(0), 2)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(10)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(10), 2)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(20)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(20), 1)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(25)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(25), 1)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(30)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(30), 0)
        self.assertEqual(type(self.ml2layers.getIndexAtPos(31)), int)
        self.assertEqual(self.ml2layers.getIndexAtPos(31), 0)
        self.assertEqual(self.mlminimum.getIndexAtPos(-1), 1)
        self.assertEqual(self.mlminimum.getIndexAtPos(1), 0)

    def test_setgetWlength(self):
        """
        Test the setWlength method and getWlength
        """

        # Check that only wavelengths within the available range are
        # accepted.
        self.assertRaises(ValueError, self.ml2layers.setWlength, 299.9)
        self.assertRaises(ValueError, self.ml2layers.setWlength, 700.1)

        # Check that the value is stored correctly
        self.ml2layers.setWlength(500)
        wl = self.ml2layers.getWlength()
        self.assertEqual(type(wl), np.float64)
        self.assertEqual(wl, 500)

        # Check that the refractive indices have been calculated
        # correctly.
        self.assertAlmostEqual(3.8 + 0.05j, self.ml2layers.getRefrIndex(0), 12)
        self.assertAlmostEqual(3.0 + 1.01j, self.ml2layers.getRefrIndex(1), 12)
        self.assertAlmostEqual(3.8 + 0.05j, self.ml2layers.getRefrIndex(2), 12)
        self.assertAlmostEqual(3.0 + 1.01j, self.ml2layers.getRefrIndex(3), 12)

        # Check the behavior of the optional refractive index list
        # argument.
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600, 1.0)
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600, 1)
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600, 'hola')
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600,
                ['a', 'b'])
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600,
                ('a', 'b'))
        self.assertRaises(TypeError, self.ml2layers.setWlength, 600,
                [1.2, 3.1 + 1.0j])

        self.ml2layers.setWlength(600, [1.2, 3.1 + 1.0j, 10, 2 + 2j])
        self.assertEqual(type(self.ml2layers.getRefrIndex(0)), np.complex128)
        self.assertEqual(type(self.ml2layers.getRefrIndex(1)), np.complex128)
        self.assertEqual(type(self.ml2layers.getRefrIndex(2)), np.complex128)
        self.assertEqual(type(self.ml2layers.getRefrIndex(3)), np.complex128)
        self.assertAlmostEqual(1.2 + 0j, self.ml2layers.getRefrIndex(0), 12)
        self.assertAlmostEqual(3.1 + 1j, self.ml2layers.getRefrIndex(1), 12)
        self.assertAlmostEqual(10 + 0j, self.ml2layers.getRefrIndex(2), 12)
        self.assertAlmostEqual(2 + 2j, self.ml2layers.getRefrIndex(3), 12)

    def test_setgetPolarization(self):
        """
        Test the setPolarization and getPolarization methods.
        """

        # Check that only correct values are accepted
        self.assertRaises(ValueError, self.ml2layers.setPolarization, 'hola')
        self.assertRaises(AttributeError, self.ml2layers.setPolarization, 2)

        # Check that the value is stored correctly
        self.ml2layers.setPolarization('te')
        self.assertEqual(self.ml2layers.getPolarization(), 'TE')
        self.ml2layers.setPolarization('tE')
        self.assertEqual(self.ml2layers.getPolarization(), 'TE')
        self.ml2layers.setPolarization('Te')
        self.assertEqual(self.ml2layers.getPolarization(), 'TE')
        self.ml2layers.setPolarization('TE')
        self.assertEqual(self.ml2layers.getPolarization(), 'TE')
        self.ml2layers.setPolarization('tm')
        self.assertEqual(self.ml2layers.getPolarization(), 'TM')
        self.ml2layers.setPolarization('tM')
        self.assertEqual(self.ml2layers.getPolarization(), 'TM')
        self.ml2layers.setPolarization('Tm')
        self.assertEqual(self.ml2layers.getPolarization(), 'TM')
        self.ml2layers.setPolarization('TM')
        self.assertEqual(self.ml2layers.getPolarization(), 'TM')

    def test_getsetPropAngle(self):
        """
        Test the setPropAngle and getPropAngle method.
        """

        # Check that appropiate exceptions are raised when called with
        # wrong arguments.
        self.assertRaises(TypeError, self.mlminimum.setPropAngle, 'hola')
        self.assertRaises(IndexError, self.mlminimum.setPropAngle, 0, -1)
        self.assertRaises(IndexError, self.mlminimum.setPropAngle, 1 + 1j, 2)

        # Check that the appropiate exception is raised when we try to
        # set the propagation angle before having set the wavelength.
        self.assertRaises(ValueError, self.mlminimum.setPropAngle, 0)
        self.assertRaises(ValueError, self.mlminimum.setPropAngle, 0 + 1j)

        # Check that we get expected results with a multilayer where
        # all three layers are the same medium.
        self.mlsame.setWlength(400)
        self.mlsame.setPropAngle(0)
        self.assertRaises(IndexError, self.mlsame.getPropAngle, -1)
        self.assertRaises(IndexError, self.mlsame.getPropAngle, 4)
        self.assertEqual(type(self.mlsame.getPropAngle(0)), np.complex128)
        self.assertEqual(type(self.mlsame.getPropAngle(1)), np.complex128)
        self.assertEqual(type(self.mlsame.getPropAngle(2)), np.complex128)
        self.assertEqual(type(self.mlsame.getPropAngle(3)), np.complex128)
        self.assertAlmostEqual(self.mlsame.getPropAngle(0), 0 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(1), 0 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(2), 0 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(3), 0 + 0j, 14)

        self.mlsame.setPropAngle(0.5)
        self.assertAlmostEqual(self.mlsame.getPropAngle(0), 0.5 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(1), 0.5 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(2), 0.5 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(3), 0.5 + 0j, 14)
        
        self.mlsame.setPropAngle(0.6, 1)
        self.assertAlmostEqual(self.mlsame.getPropAngle(0), 0.6 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(1), 0.6 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(2), 0.6 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(3), 0.6 + 0j, 14)

        self.mlsame.setPropAngle(0.7, 2)
        self.assertAlmostEqual(self.mlsame.getPropAngle(0), 0.7 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(1), 0.7 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(2), 0.7 + 0j, 14)
        self.assertAlmostEqual(self.mlsame.getPropAngle(3), 0.7 + 0j, 14)

        # Check that we get expected results with a multilayer with
        # different mediums
        self.ml2layers.setWlength(600)
        self.ml2layers.setPropAngle(0)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(0), 0 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(1), 0 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(2), 0 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(3), 0 + 0j, 14)

        self.ml2layers.setPropAngle(0.5)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(0), 0.5 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(1),
                0.686102733 + 0j, 8)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(2), 0.5 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(3),
                0.686102733 + 0j, 8)

        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(1.2)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(0), 1.2 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(1),
                1.061219996 - 0.640869953j, 8)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(2), 1.2 + 0j, 14)
        self.assertAlmostEqual(self.ml2layers.getPropAngle(3),
                1.061219996 - 0.640869953j, 8)

        # Check that everything works when using a list instead of a
        # single angle.
        self.assertRaises(TypeError, self.ml2layers.setPropAngle, [0, 0, 0])
        self.ml2layers.setPropAngle([0, 1 + 1j, 2, 3 + 3j])
        self.assertEqual(type(self.ml2layers.getPropAngle(0)), np.complex128)
        self.assertEqual(type(self.ml2layers.getPropAngle(1)), np.complex128)
        self.assertEqual(type(self.ml2layers.getPropAngle(2)), np.complex128)
        self.assertEqual(type(self.ml2layers.getPropAngle(3)), np.complex128)
        self.assertAlmostEqual(0 + 0j, self.ml2layers.getPropAngle(0), 12)
        self.assertAlmostEqual(1 + 1j, self.ml2layers.getPropAngle(1), 12)
        self.assertAlmostEqual(2 + 0j, self.ml2layers.getPropAngle(2), 12)
        self.assertAlmostEqual(3 + 3j, self.ml2layers.getPropAngle(3), 12)

    def test_getRefrIndex(self):
        """
        Test the getRefrIndex method.
        """

        # Check that appropiate exceptions are raised when supplied
        # with bad arguments.
        self.assertRaises(IndexError, self.ml2layers.getRefrIndex, -1)
        self.assertRaises(IndexError, self.ml2layers.getRefrIndex, 4)

    def test_calcMatrices(self):
        """
        Test the calcMatrices and getMatrix methods.
        """

        # Check that we raise the proper exception when called with
        # bad arguments.
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, (1, 2))
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, 2)
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, ['hola'])
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [2.3])
        self.assertRaises(IndexError, self.ml2layers.calcMatrices, [4])
        self.assertRaises(IndexError, self.ml2layers.calcMatrices, [-2])
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [0])
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [3])

        self.assertRaises(IndexError, self.ml2layers.getMatrix, -1)
        self.assertRaises(IndexError, self.ml2layers.getMatrix, 4)

        # Check that initially all matrices are set to None
        self.assertEqual(self.ml2layers.getMatrix(0), None)
        self.assertEqual(self.ml2layers.getMatrix(1), None)
        self.assertEqual(self.ml2layers.getMatrix(2), None)
        self.assertEqual(self.ml2layers.getMatrix(3), None)

        # Check that we raise the proper exception when there is
        # missing data.
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [1, 2])
        self.ml2layers.setWlength(400)
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [1, 2])
        self.ml2layers.setPropAngle(0)
        self.assertRaises(ValueError, self.ml2layers.calcMatrices, [1, 2])
        self.ml2layers.setPolarization('te')

        # Check that the end mediums still have the matrix to None
        # after calculating all the matrices
        self.ml2layers.calcMatrices()
        self.assertEqual(self.ml2layers.getMatrix(0), None)
        self.assertEqual(self.ml2layers.getMatrix(3), None)

        # Check that we get expected results for simple cases
        self.ml2layers.setThickness(0, 1)
        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        mat1 = self.ml2layers.getMatrix(1)
        mat2 = self.ml2layers.getMatrix(2)
        realmat2 = np.matrix([
                [0.309018519313945 - 0.002987837079530j,
                -0.000351709278732 - 0.237764423120503j],
                [0.013393840560846 - 3.804235130228049j,
                0.309018519313945 - 0.002987837079530j]])
        np.testing.assert_array_equal(mat1, np.eye(2, 2))
        np.testing.assert_array_almost_equal(mat2, realmat2, 14)
        self.ml2layers.setPolarization('tm')
        self.ml2layers.calcMatrices()
        mat1 = self.ml2layers.getMatrix(1)
        mat2 = self.ml2layers.getMatrix(2)
        realmat2 = np.matrix([
                [0.309018519313945 - 0.002987837079530j,
                0.013393840560846 - 3.80423513022804j],
                [-0.000351709278732 - 0.237764423120503j,
                0.309018519313945 - 0.002987837079530j]])
        np.testing.assert_array_equal(mat1, np.eye(2, 2))
        np.testing.assert_array_almost_equal(mat2, realmat2, 14)

    def test_charMatrix(self):
        """
        Check the updateCharMatrix, getCharMatrixUpDown,
        getCharMatrixDownUp, getCoefficientsUpDown and
        getCoefficientsDownUp methods.
        """

        # Check that we get the proper exception when one of the
        # individual matrices is not calculated.
        self.assertRaises(ValueError, self.ml2layers.updateCharMatrix)
        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices([1])
        self.assertRaises(ValueError, self.ml2layers.updateCharMatrix)
        self.assertEqual(self.ml2layers.getCharMatrixDownUp(), None)
        self.assertEqual(self.ml2layers.getCharMatrixUpDown(), None)

        # Check that we get the expected result in some simple cases
        self.ml2layers.setThickness(0, 1)
        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        mat = self.ml2layers.getCharMatrixUpDown()
        realmat = np.matrix([
                [0.309018519313945 - 0.002987837079530j,
                -0.000351709278732 - 0.237764423120503j],
                [0.013393840560846 - 3.804235130228049j,
                0.309018519313945 - 0.002987837079530j]])
        np.testing.assert_array_almost_equal(mat, realmat)
        mat = self.ml2layers.getCharMatrixDownUp()
        np.testing.assert_array_almost_equal(mat, realmat)

        self.ml2layers.setThickness(0, 1)
        self.ml2layers.setThickness(0, 2)
        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        mat = self.ml2layers.getCharMatrixUpDown()
        np.testing.assert_array_equal(mat, np.eye(2, 2))
        mat = self.ml2layers.getCharMatrixDownUp()
        np.testing.assert_array_equal(mat, np.eye(2, 2))

        # Check the results for some simple cases.
        # Pure dielectric, same material everywhere
        self.mlsame.setWlength(600)
        self.mlsame.setPropAngle(0)
        self.mlsame.setPolarization('te')
        self.mlsame.calcMatrices()
        self.mlsame.updateCharMatrix()
        cud = self.mlsame.getCoefficientsUpDown()
        cdu = self.mlsame.getCoefficientsDownUp()
        self.assertAlmostEqual(cud['r'], 0, 14)
        self.assertAlmostEqual(cud['R'], 0, 14)
        self.assertAlmostEqual(cud['t'], 0.92587058481 + 0.37784078681j, 10)
        self.assertAlmostEqual(cud['T'], 1, 14)
        self.assertAlmostEqual(cdu['r'], 0, 14)
        self.assertAlmostEqual(cdu['R'], 0, 14)
        self.assertAlmostEqual(cdu['t'], 0.92587058481 + 0.37784078681j, 10)
        self.assertAlmostEqual(cdu['T'], 1, 14)

        # Non-zero ext. coefficients. Same material everywhere
        self.mlsame.setWlength(400)
        self.mlsame.setPropAngle(0)
        self.mlsame.setPolarization('te')
        self.mlsame.calcMatrices()
        self.mlsame.updateCharMatrix()
        cud = self.mlsame.getCoefficientsUpDown()
        cdu = self.mlsame.getCoefficientsDownUp()
        self.assertEqual(cud['r'], 0)
        self.assertEqual(cud['R'], 0)
        self.assertTrue(cud['R'] + cud['T'] < 1)
        self.assertEqual(cdu['r'], 0)
        self.assertEqual(cdu['R'], 0)
        self.assertTrue(cdu['R'] + cdu['T'] < 1)

        # Same as before, but the layer has zero thickness
        self.mlsame.setThickness(0, 1)
        self.mlsame.setThickness(0, 2)
        self.mlsame.calcMatrices()
        self.mlsame.updateCharMatrix()
        cud = self.mlsame.getCoefficientsUpDown()
        cdu = self.mlsame.getCoefficientsDownUp()
        self.assertEqual(cud['r'], 0)
        self.assertEqual(cud['R'], 0)
        self.assertEqual(cud['t'], 1)
        self.assertEqual(cud['T'], 1)
        self.assertEqual(cdu['r'], 0)
        self.assertEqual(cdu['R'], 0)
        self.assertEqual(cdu['t'], 1)
        self.assertEqual(cdu['T'], 1)

        # Pure dielectric again, now with an angle
        self.mlminsame.setWlength(600)
        self.mlminsame.setPropAngle(1)
        self.mlminsame.setPolarization('te')
        self.mlminsame.calcMatrices()
        self.mlminsame.updateCharMatrix()
        cud = self.mlminsame.getCoefficientsUpDown()
        cdu = self.mlminsame.getCoefficientsDownUp()
        self.assertEqual(cud['r'], 0)
        self.assertEqual(cud['R'], 0)
        self.assertEqual(cud['t'], 1)
        self.assertEqual(cud['T'], 1)
        self.assertEqual(cdu['r'], 0)
        self.assertEqual(cdu['R'], 0)
        self.assertEqual(cdu['t'], 1)
        self.assertEqual(cdu['T'], 1)

        # Interface between two dielectrics. We hope to get the results
        # from the Fresnel formulae.
        self.mlminimum.setWlength(600)
        self.mlminimum.setPropAngle(0)
        self.mlminimum.setPolarization('te')
        self.mlminimum.calcMatrices()
        self.mlminimum.updateCharMatrix()
        cud = self.mlminimum.getCoefficientsUpDown()
        cdu = self.mlminimum.getCoefficientsDownUp()
        n = 2.8 / 3.7
        self.assertAlmostEqual(cud['R'], ((n - 1) / (n + 1)) ** 2, 14)
        self.assertAlmostEqual(cud['T'] + cud['R'], 1, 14)
        n = 3.7 / 2.8
        self.assertAlmostEqual(cdu['R'], (-(n - 1) / (n + 1)) ** 2, 14)
        self.assertAlmostEqual(cdu['T'] + cdu['R'], 1, 14)

        # Since we are at normal incidence, TM and TE should be the
        # same. Again we compare to the result given by the Fresnel
        # formulae.
        self.mlminimum.setWlength(600)
        self.mlminimum.setPropAngle(0)
        self.mlminimum.setPolarization('tm')
        self.mlminimum.calcMatrices()
        self.mlminimum.updateCharMatrix()
        cud = self.mlminimum.getCoefficientsUpDown()
        cdu = self.mlminimum.getCoefficientsDownUp()
        n = 2.8 / 3.7
        self.assertAlmostEqual(cud['R'], ((n - 1) / (n + 1)) ** 2, 14)
        self.assertAlmostEqual(cud['T'] + cud['R'], 1, 14)
        n = 3.7 / 2.8
        self.assertAlmostEqual(cdu['R'], (-(n - 1) / (n + 1)) ** 2, 14)
        self.assertAlmostEqual(cdu['T'] + cdu['R'], 1, 14)

        # Same as before, but no normal incidence now
        self.mlminimum.setWlength(600)
        self.mlminimum.setPropAngle(0.5)
        self.mlminimum.setPolarization('te')
        self.mlminimum.calcMatrices()
        self.mlminimum.updateCharMatrix()
        cud = self.mlminimum.getCoefficientsUpDown()
        cdu = self.mlminimum.getCoefficientsDownUp()
        Ai = self.mlminimum.getPropAngle(0)
        Af = self.mlminimum.getPropAngle(1)
        expectedr = -(np.sin(Ai - Af)) / (np.sin(Ai + Af))
        expectedR = np.abs(expectedr) ** 2
        self.assertAlmostEqual(cud['R'], expectedR, 14)
        self.assertAlmostEqual(cud['T'] + cud['R'], 1, 14)
        Ai = self.mlminimum.getPropAngle(1)
        Af = self.mlminimum.getPropAngle(0)
        expectedr = -(np.sin(Ai - Af)) / (np.sin(Ai + Af))
        expectedR = np.abs(expectedr) ** 2
        self.assertAlmostEqual(cdu['R'], expectedR, 14)
        self.assertAlmostEqual(cdu['T'] + cdu['R'], 1, 14)

        # Same as before but TM.
        self.mlminimum.setWlength(600)
        self.mlminimum.setPropAngle(0.5)
        self.mlminimum.setPolarization('tm')
        self.mlminimum.calcMatrices()
        self.mlminimum.updateCharMatrix()
        cud = self.mlminimum.getCoefficientsUpDown()
        cdu = self.mlminimum.getCoefficientsDownUp()
        Ai = self.mlminimum.getPropAngle(0)
        Af = self.mlminimum.getPropAngle(1)
        expectedr = np.tan(Ai - Af) / np.tan(Ai + Af)
        expectedR = np.abs(expectedr) ** 2
        self.assertAlmostEqual(cud['R'], expectedR, 14)
        self.assertAlmostEqual(cud['T'] + cud['R'], 1, 14)
        Ai = self.mlminimum.getPropAngle(1)
        Af = self.mlminimum.getPropAngle(0)
        expectedr = np.tan(Ai - Af) / np.tan(Ai + Af)
        expectedR = np.abs(expectedr) ** 2
        self.assertAlmostEqual(cdu['R'], expectedR, 14)
        self.assertAlmostEqual(cdu['T'] + cdu['R'], 1, 14)

    def test_calculateFx(self):
        """
        Test the calculateFx method. Further tests of this methods are
        performed in the test_crawford method.
        """

        # In a system with all the layers made of the same material, we
        # expect Fx(z) = 1 for any configuration. Bear in mind that a
        # dipole oscilating along the x axis does not emit light along
        # the x axis.
        self.assertAlmostEqual(self.mlsame.calculateFx(-5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(-5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(15, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(15, 400, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFx(-5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(-5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(15, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(15, 600, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFx(-5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(-5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(15, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFx(15, 400, 0.5)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFx(-5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFx(-5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFx(5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFx(15, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFx(15, 400, np.pi / 2)),
                np.complex128)

    def test_calculateFy(self):
        """
        Test the calculateFy method. Further tests of this methods are
        performed in the test_crawford method.
        """

        # In a system with all the layers made of the same material, we
        # expect Fy(z) = 1 for any configuration.
        self.assertAlmostEqual(self.mlsame.calculateFy(-5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(-5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(15, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(15, 400, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFy(-5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(-5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(15, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(15, 600, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFy(-5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(-5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(15, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFy(15, 400, 0.5)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFy(-5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFy(-5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFy(5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFy(15, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFy(15, 400, np.pi / 2)),
                np.complex128)

    def test_calculateFz(self):
        """
        Test the calculateFz method. Further tests of this methods are
        performed in the test_crawford method.
        """

        # In a system with all the layers made of the same material, we
        # expect Fz(z) = 1 for any configuration. However, bear in mind
        # that a dipole oscilating along z does not emit light along z.
        self.assertAlmostEqual(self.mlsame.calculateFz(-5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(-5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(5, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(5, 400, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(15, 400, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(15, 400, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFz(-5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(-5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(5, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(5, 600, 0)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(15, 600, 0), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(15, 600, 0)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFz(-5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(-5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(5, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(5, 400, 0.5)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(15, 400, 0.5), 1, 14)
        self.assertEqual(type(self.mlsame.calculateFz(15, 400, 0.5)),
                np.complex128)

        self.assertAlmostEqual(self.mlsame.calculateFz(-5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFz(-5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(5, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFz(5, 400, np.pi / 2)),
                np.complex128)
        self.assertAlmostEqual(self.mlsame.calculateFz(15, 400, np.pi / 2), 1,
                14)
        self.assertEqual(type(self.mlsame.calculateFz(15, 400, np.pi / 2)),
                np.complex128)

    def test_thickness_reset(self):
        """
        After changing the thickness of a layer, its matrix, the
        matrices of the whole system (in both directions) and the
        coefficients should be reset to None.
        """

        self.assertTrue(self.ml2layers.getPropAngle(0) == None)
        self.assertTrue(self.ml2layers.getPropAngle(1) == None)
        self.assertTrue(self.ml2layers.getPropAngle(2) == None)
        self.assertTrue(self.ml2layers.getPropAngle(3) == None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0.5)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) != None)
        self.assertTrue(self.ml2layers.getMatrix(2) != None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() != None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() != None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() != {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() != {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setThickness(5.0, 1)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) != None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

    def test_wlength_reset(self):
        """
        After changing the wavelength of the light going across the
        system, the matrices of the whole system (in both directions)
        and each individual layer as well as the coefficients should be
        reset to None. The propagation angles are also reset to None
        since the refractive index changes with the wavelength and
        therefore the propagation angles change too.
        """

        self.assertTrue(self.ml2layers.getPropAngle(0) == None)
        self.assertTrue(self.ml2layers.getPropAngle(1) == None)
        self.assertTrue(self.ml2layers.getPropAngle(2) == None)
        self.assertTrue(self.ml2layers.getPropAngle(3) == None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0.5)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) != None)
        self.assertTrue(self.ml2layers.getMatrix(2) != None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() != None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() != None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() != {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() != {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setWlength(600)
        self.assertTrue(self.ml2layers.getPropAngle(0) == None)
        self.assertTrue(self.ml2layers.getPropAngle(1) == None)
        self.assertTrue(self.ml2layers.getPropAngle(2) == None)
        self.assertTrue(self.ml2layers.getPropAngle(3) == None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

    def test_polarization_reset(self):
        """
        After changing the polarization of the light going across the
        system, the matrices of the whole system (in both directions)
        and each individual layer as well as the coefficients should be
        reset to None.
        """

        self.assertTrue(self.ml2layers.getPropAngle(0) == None)
        self.assertTrue(self.ml2layers.getPropAngle(1) == None)
        self.assertTrue(self.ml2layers.getPropAngle(2) == None)
        self.assertTrue(self.ml2layers.getPropAngle(3) == None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0.5)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) != None)
        self.assertTrue(self.ml2layers.getMatrix(2) != None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() != None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() != None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() != {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() != {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setPolarization('tm')
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

    def test_propangle_reset(self):
        """
        After changing the propagation angle of the light, the matrices
        of the whole system (in both directions) and each individual
        layer as well as the coefficients should be reset to None.
        """

        self.assertTrue(self.ml2layers.getPropAngle(0) == None)
        self.assertTrue(self.ml2layers.getPropAngle(1) == None)
        self.assertTrue(self.ml2layers.getPropAngle(2) == None)
        self.assertTrue(self.ml2layers.getPropAngle(3) == None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setWlength(400)
        self.ml2layers.setPropAngle(0.5)
        self.ml2layers.setPolarization('te')
        self.ml2layers.calcMatrices()
        self.ml2layers.updateCharMatrix()
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) != None)
        self.assertTrue(self.ml2layers.getMatrix(2) != None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() != None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() != None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() != {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() != {'r': None,
                't': None, 'R': None, 'T': None})

        self.ml2layers.setPropAngle(0.2, 1)
        self.assertTrue(self.ml2layers.getPropAngle(0) != None)
        self.assertTrue(self.ml2layers.getPropAngle(1) != None)
        self.assertTrue(self.ml2layers.getPropAngle(2) != None)
        self.assertTrue(self.ml2layers.getPropAngle(3) != None)
        self.assertTrue(self.ml2layers.getMatrix(0) == None)
        self.assertTrue(self.ml2layers.getMatrix(1) == None)
        self.assertTrue(self.ml2layers.getMatrix(2) == None)
        self.assertTrue(self.ml2layers.getMatrix(3) == None)
        self.assertTrue(self.ml2layers.getCharMatrixUpDown() == None)
        self.assertTrue(self.ml2layers.getCharMatrixDownUp() == None)
        self.assertTrue(self.ml2layers.getCoefficientsUpDown() == {'r': None,
                't': None, 'R': None, 'T': None})
        self.assertTrue(self.ml2layers.getCoefficientsDownUp() == {'r': None,
                't': None, 'R': None, 'T': None})

    def test_reference(self):
        """
        This test guarantees that the internal structure of the
        multilayer is as it should be. It is also ilustrative of how
        the mediums of each layer are linked to the multilayer.
        """

        # Create a multilayer with two equal mediums and check that
        # the refractive indices correspond to the medium that we
        # chose.
        self.testlayer = ml.Multilayer([
                self.medium1,
                self.medium1])
        self.testlayer.setWlength(400)
        self.assertAlmostEqual(self.testlayer.getRefrIndex(0), 4 + 0.01j,
                14)
        self.assertAlmostEqual(self.testlayer.getRefrIndex(1), 4 + 0.01j,
                14)

        # self.medium1 is nothing but a pointer to the actual medium
        # object. Here we update the pointer self.medium1 to point to
        # the same object as self.medium2.
        self.assertEqual(self.medium1.getMinMaxWlength(), (200, 700))
        self.medium1 = self.medium2
        self.assertEqual(self.medium1.getMinMaxWlength(), (300, 800))

        # However, the layers of the multilayer still point to the
        # medium originally pointed to by self.medium1. Therefore
        # nothing has changed in the multilayer.
        self.assertAlmostEqual(self.testlayer.getRefrIndex(0), 4 + 0.01j,
                14)
        self.assertAlmostEqual(self.testlayer.getRefrIndex(1), 4 + 0.01j,
                14)

    def test_crawford(self):
        """
        Test that we reproduce the results published by Crawford in his
        paper J. Chem. Phys. 89 (10), 1998.
        """

        angle = np.deg2rad(65)
        wlength = 520

        # Figure 2 with film
        test_cases = [
                (-100, 4.91930941e-05),
                (0, 0.24190098),
                (100, 2.54848769),
                (200, 0.02325695),
                (300, 2.35591136),
                (400, 3.82854307),
                (500, 3.53145497),
                (600, 1.74851184),
                (700, 0.18329850),
                (800, 0.33136105),
                (900, 2.05122719)]
        for (z, output) in test_cases:
            f = self.cssystem_f2_film.calculateFy(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f) ** 2, output, 7)

        # Figure 2 without film
        test_cases = [
                (-100, 1.15413355e-05),
                (0, 0.05675309),
                (100, 1.38531364),
                (200, 3.29590705),
                (300, 3.96297991),
                (400, 2.74915060),
                (500, 0.81422121),
                (600, 0.00699794),
                (700, 1.09877478),
                (800, 3.04636957),
                (900, 3.98887445)]
        for (z, output) in test_cases:
            f = self.cssystem_f2_nofilm.calculateFy(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f) ** 2, output, 7)

        # Figure 3 with film
        test_cases = [
                (-100, 3.71093788e-06),
                (0, 0.55104199),
                (100, 0.06835578),
                (200, 3.17735585),
                (300, 2.05746777),
                (400, 0.51889431),
                (500, 0.03172744),
                (600, 1.06145038)]
        for (z, output) in test_cases:
            f = self.cssystem_f3_film.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

        # Figure 3 without film
        test_cases = [
                (-100, 3.25375474e-06),
                (0, 2.13578650),
                (100, 0.57832585),
                (200, 0.01825289),
                (300, 0.99071188),
                (400, 2.56652773),
                (500, 3.24002371),
                (600, 2.36768098)]
        for (z, output) in test_cases:
            f = self.cssystem_f3_nofilm.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

        # Figure 4 with film
        test_cases = [
                (-100, 2.94760736e-07),
                (0, 0.26262692),
                (100, 0.33368022),
                (200, 0.38948285),
                (300, 2.58686843),
                (400, 3.06396006),
                (500, 0.95224614),
                (600, 0.09595269)]
        for (z, output) in test_cases:
            f = self.cssystem_f4_film.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

        # Figure 4 without film
        test_cases = [
                (-100, 7.40331028e-06),
                (0, 1.49218807),
                (100, 2.60156534e-04),
                (200, 1.45670696),
                (300, 3.21016995),
                (400, 2.06859361),
                (500, 0.11013703),
                (600, 0.90003208)]
        for (z, output) in test_cases:
            f = self.cssystem_f4_nofilm.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

        # Figure 5 with film, dipole x
        z = 0
        anglerange = np.arange(0, np.pi / 2 + 0.2, 0.2)

        test_cases = [
                (0.0, 0.5576867),
                (0.2, 0.54647638),
                (0.4, 0.5124186),
                (0.6, 0.45436745),
                (0.8, 0.37100651),
                (1.0, 0.26300115),
                (1.2, 0.1402264),
                (1.4, 0.03523924),
                (1.6, 0.00108746)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_film.calculateFx(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.cos(angle)) ** 2, output,
                    7)
            f = self.cssystem_f5_film_alt.calculateFx(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.cos(angle)) ** 2, output,
                    7)
        test_cases = [
                (0.0, 0.5576867),
                (0.2, 0.5415086),
                (0.4, 0.49379517),
                (0.6, 0.41727818),
                (0.8, 0.31761507),
                (1.0, 0.20546283),
                (1.2, 0.09890071),
                (1.4, 0.0228346),
                (1.6, 0.00068492)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_film.calculateFy(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f) ** 2, output, 7)
            f = self.cssystem_f5_film_alt.calculateFy(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f) ** 2, output, 7)
        test_cases = [
                (0.0, 0.0),
                (0.2, 0.05618238),
                (0.4, 0.2004675),
                (0.6, 0.36845438),
                (0.8, 0.47772657),
                (1.0, 0.45862621),
                (1.2, 0.29612175),
                (1.4, 0.08250928),
                (1.6, 0.00261383)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_film.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)
            f = self.cssystem_f5_film_alt.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

        # Figure 5 without film
        test_cases = [
                (0.0, 0.31647911),
                (0.2, 0.31638766),
                (0.4, 0.31555465),
                (0.6, 0.31210118),
                (0.8, 0.30196986),
                (1.0, 0.27634856),
                (1.2, 0.214995),
                (1.4, 0.08734856),
                (1.6, 0.00338848)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_nofilm.calculateFx(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.cos(angle)) ** 2, output,
                    7)
        test_cases = [
                (0.0, 3.16479113e-01),
                (0.2, 3.04032518e-01),
                (0.4, 2.68641274e-01),
                (0.6, 2.15851575e-01),
                (0.8, 1.53952045e-01),
                (1.0, 9.26880237e-02),
                (1.2, 4.17390122e-02),
                (1.4, 9.19490314e-03),
                (1.6, 2.71727613e-04)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_nofilm.calculateFy(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f) ** 2, output, 7)
        test_cases = [
                (0.0, 0.0),
                (0.2, 0.14378723),
                (0.4, 0.54571475),
                (0.6, 1.11877807),
                (0.8, 1.7188322),
                (1.0, 2.1302493),
                (1.2, 2.00697),
                (1.4, 0.90407421),
                (1.6, 0.03600333)]
        for (angle, output) in test_cases:
            f = self.cssystem_f5_nofilm.calculateFz(z, wlength, angle)
            self.assertAlmostEqual(np.absolute(f * np.sin(angle)) ** 2, output,
                    7)

    def test_symmetry(self):
        """
        Test that the reflectance and the transmittance are equal
        in the up-down and down-up direction, provided all the
        layers are insulators.
        """

        self.symmetry.setPolarization('TE')
        self.symmetry.setWlength(600)
        self.symmetry.setPropAngle(np.deg2rad(10))
        self.symmetry.calcMatrices()
        self.symmetry.updateCharMatrix()
        Tud = self.symmetry.getCoefficientsUpDown()['T']
        Rud = self.symmetry.getCoefficientsUpDown()['R']
        Tdu = self.symmetry.getCoefficientsDownUp()['T']
        Rdu = self.symmetry.getCoefficientsDownUp()['R']
        self.assertAlmostEqual(Tud, Tdu, 12)
        self.assertAlmostEqual(Rud, Rdu, 12)

if __name__ == '__main__':
    unittest.main()
