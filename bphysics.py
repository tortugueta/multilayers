# -*- coding: utf-8 -*-

import numpy as np
#import scipy.interpolate as sp

###############################################################################
#                               Constants                                     #
###############################################################################

# Useful physical constants
# Source: http://en.wikipedia.org

q       = 1.602176487e-19       # Elementary charge     [C]
k_jk    = 1.3806488e-23         # Boltzmann constant    [J/K]
k_evk   = 8.6173324e-5          # Boltzmann constant    [eV/K]
h_eV    = 4.135667516e-15       # Planck's constant     [eV·s]
h_j     = 6.62606957e-34        # Planck's constant     [J·s]
hbar_ev = 6.58211928e-16        # Planck's constant     [eV·s]
hbar_j  = 1.054571726e-34       # Planck's constant     [J·s]
e0      = 8.854187817620e-12    # Vacuum permittivity   [F/m]
ksi     = 11.9                  # Rel. perm. of silicon

###############################################################################
#                                Classes                                      #
###############################################################################


###############################################################################
#                               Functions                                     #
###############################################################################

def rdfile(fname, dtype='float', commentchar="#", delimiter=None,
           converters=None, skiprows=0, usecols=None, unpack=False):
        """
        Load comments and data from a text file.

        Each row in the text file must have the same number of values. A first
        section of the function reads the comments. Then it calls
        numpy.loadtxt to read the data. The parameters are passed directly to
        numpy.loadtxt.

        Parameters
        ----------
        fname : file or str
            File or filename to read.  If the filename extension is ``.gz`` or
            ``.bz2``, the file is first decompressed.
        dtype : dtype, optional
            Data type of the resulting array.  If this is a record data-type,
            the resulting array will be 1-dimensional, and each row will be
            interpreted as an element of the array.   In this case, the number
            of columns used must match the number of fields in the data-type.
        commentchar : str, optional
            The character used to indicate the start of a comment.
        delimiter : str, optional
            The string used to separate values.  By default, this is any
            whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will convert
            that column to a float.  E.g., if column 0 is a date string:
            ``converters = {0: datestr2num}``. Converters can also be used to
            provide a default value for missing data:
            ``converters = {3: lambda s: float(s or 0)}``.
        skiprows : int, optional
            Skip the first `skiprows` lines.
        usecols : sequence, optional
            Which columns to read, with 0 being the first.  For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = rdfile(...)``. Default is False.

        Returns
        -------
        out : (string, ndarray)
            A tuple with the comments and data read from the text file.

        See Also
        -------
        numpy.loadtxt
        """

        # Read the comments
        fhandle = open(fname, "r")
        comments = ""
        for line in fhandle:
            lstripline = line.lstrip()
            if lstripline == "" or lstripline[0] == commentchar:
                comments = "".join([comments, line])
            else:
                break
        fhandle.close()

        # Read the data
        data = np.loadtxt(fname, dtype, commentchar, delimiter, converters,
                          skiprows, usecols, unpack)

        return (comments, data)


def wdfile(fname, comments, data, fmt='%.18e', delimiter='\t'):
        """
        Save comments and data to a text file.

        A first section of the funcion saves the comments. Then it open the
        file in append mode and calls numpy.savetxt in order to save the data.

        Parameters
        ----------
        fname : filename or file handle
            If the filename ends in ``.gz``, the file is automatically saved in
            compressed gzip format.  `loadtxt` understands gzipped files
            transparently.
        comments : string
            The comments section of the file.
        data : array_like
            Data to be saved to a text file.
        fmt : str or sequence of strs, optional
            A single format (%10.5f), a sequence of formats, or a
            multi-format string, e.g. 'Iteration %d -- %10.5f', in which
            case `delimiter` is ignored.
        delimiter : str, optional
            Character separating columns.

        See Also
        --------
        numpy.savetxt
        """

        # Write the comments
        fhandle = open(fname, "w")
        fhandle.write("".join([comments, '\n']))
        fhandle.close()

        # Write the data
        fhandle = open(fname, "a")
        np.savetxt(fhandle, data, fmt, delimiter)
        fhandle.close()


###############################################################################
#                               Self test                                     #
###############################################################################

if __name__ == "__main__":
    
    pass
