#!/usr/bin/python
#
#    libstat.py
#    Copyright 2013-2015 Jacopo Nespolo <jacopo.nespolo@pi.infn.it>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import numpy as np
from sys import stderr, stdout

__version__ = "0.1.3"

class StatError(Exception):
    '''
    Base Stat module exception
    '''
    pass

class BlockingError(StatError):
    '''
    To be raised when blocksize does not divide number of data,
    hence discarding of statistics would happen.
    Value can optionally be the number of data that would be discarded.
    '''
    def __init__(self, discard="", msg=None):
        self.msg = "Would discard data"
        if discard != "":
            self.value = discard
            self.msg += ": " + str(self.value)
        if msg != None:
            self.msg = msg
        return msg
    def __str__(self):
        return self.msg


def block(data, bsize=2, ignore_discarded='Error', reverse=True):
    '''
    Groups data in blocks of size bsize.

    Parameter
    ---------
    data : array like
        The data to be bloked.
    bsize : int (defaults to 2)
        Size of blocks.
    ignore_discarded : string (defaults to 'Error')
        Determines what to do if len(data) is not an integer multiple of
        bsize. Possible values are
        'Error' - raise BlockingError;
        'Warn'  - print warning message and continue exectution, discarding
                  excess data;
        'Silent'- (or anything else) Keep executing silently.
    reverse : bool (default = True)
        Block backwards, so that if any data is lost in the blocking, it is
        the beginning of the series rather than the end.
    Returns
    -------
    blocked : Array of blocekd data. 
    '''
    if bsize == 1:
        return data
    n = len(data)
    if (n % bsize != 0):
        if ignore_discarded == 'Error':
            raise BlockingError(n % bsize)
        elif ignore_discarded == 'Warn':
            stderr.write("Warning: BlockingError: (%d)\n" %(n%bsize))
        else:
            pass

    nblocks = n // bsize
    try:
        blocked = np.empty((nblocks, np.shape(data)[1]))
    except IndexError:
        blocked = np.empty(nblocks)
    
    if reverse == 'True':
        rdata = data[::-1]
    else:
        rdata = data

    for i in range(nblocks):
        blocked[i] = np.mean(rdata[i*bsize: (i+1)*bsize], axis=0)
    if reverse == "True":
        return blocked[::-1]
    else:
        return blocked
#


def scf_k(data, k=1, mean=None, n=None, normalise=False, var=None, 
          zeromean=False):
    '''
    Computes the auto-correlation at lag k
        C(k) = < (x_n - x_mean)(x_{n+k} - x_mean) >

    Parameters
    ----------
    data : array like
        Data to compute the ACF on.
    k : int
        Lag to compute the ACF at, i.e. C(k).

    Optional Parameters
    -------------------
    mean : float or array like
        Mean of data, down the rows. If missing, it will be computed from 
        data.
    var : float or array like
        Variance of data, down the rows. It is only needed if k = 0 
        (C(k) = var), or if the normalised ACF is desired.
        If missing, it will be computed from data.
    normalise : boolean (defaults to False)
        Wether to normalise the ACF.
    zeromean : boolean
        Wether the mean was alredy subtracted from data.

    Returns
    -------
    sc : float or array like
        The ACF at lag k.
    '''
    if k == 0:
        return np.var(data, ddof=1, axis=0)

    if n == None:
        n = len(data)

    if zeromean != True:
        if mean == None:
            mean = np.mean(data)
        diff = data - mean
    else:
        diff = data
    sc = np.mean(diff[:-k] * np.roll(diff, -k, axis=0)[:-k], 
                 axis=0)
    sc *= n / (n - 1.)

    if normalise == True:
        if var != None:
            sc /= var
        else:
            sc /= np.var(data)
    return sc
#


def scf(*args, **kwargs):
    '''
    Wrapper for __scf_fft.
    Please refer to relative documentation. 
    '''
    return __scf_fft(*args, **kwargs)
#


def __scf_fft(data, mean=None, var=None, kmax=1000, 
              zeromean=False, normalise=False):
    '''
    Compute the whole autocorrelation function by Fast Fourier Transform
    method. Let F be the Fourier transform, f its inverse.
    Calling g the variate of interest (data), and gg = g - <g>,
    C(k) = f[ F[gg] F[gg]* ](k)

    Parameters
    ----------
    data : array like
        Data to analyse.
    mean, var: floats or array like
        Mean and variance of data. The mean is only needed if zeromean=False,
        the variance if normalise=False.
        If not provided and needed, they will be estimated from data.
    kmax : int (default = min(1000, len(data)))
        Maximum lag to compute the ACF for. Note that further truncation
        might be needed.
    zeromean : boolean (default = False)
        Wether the mean was already subtracted from data.
    normalise : boolean (default = False)
        Wether to normalise the ACF, i.e. C(0) = 1.

    Returns
    -------
    scf : array like
    '''
    if zeromean == True:
        zeromean = data
    else:
        if mean == None:
            mean = np.mean(data, axis=0)
        zeromean = data - mean

    scf = np.fft.fft(zeromean, axis=0)
    scf *= np.conj(scf)
    scf = np.fft.ifft(scf, axis=0)

    if not normalise:
        if var == None:
            var = np.var(data, axis=0, ddof=1)
        scf *= var

    return np.real(scf[:min(kmax, len(scf))])


def __scf(data, kmax=1000, mean=None, n=None, 
        **kwargs):
    '''
    Computes the auto-correlation function by calling scf_k.

    Parameters
    ----------
    data : array like
        Data to be analysed.
    kmax : int (default = 1000)
        Maximum lag to calculate the ACF for. Note that right now there is 
        no check on wether the ACF for such a lag can be computed.
        TODO: implement a check and test it.
    mean: float or array like
        Mean of data. If not provided, it will be estimated from data.
    n : int
        Length of dataset. 
    **kwargs :
        Additional parameters to be passed on to scf_k. 
        Please refer to the relative documentation.

    Returns
    -------
    scff : array like
        Autocorrelation function.
    '''
    try:
        scff = np.empty((kmax, np.shape(data)[1]))
    except IndexError:
        scff = np.empty(kmax)

    if mean == None:
        mean = np.mean(data, axis=0)
    if n == None:
        n = len(data)

    for k in range(0, kmax):

        scff[k] = scf_k(data - mean, k=k, n=n, zeromean=True, **kwargs)
    return scff
#


def scf_err_k(scf, k, N, cutoff=100):
    '''
    Error of the self-correlation at lag k, dC(k).
    Implementation of Equation (E.11) from
    Luscher, hep-lat/0409106

    Parameters
    ----------
    scf : array like
        Auto-correlation function to assess errors for.
    k : int
        Lag at which to compute the error.
    N : int
        Size of original dataset from which scf was calculated. This is 
        in general different from len(scf)!
    cutoff : int (default = 100)
        Lambda in the article by Luscher.

    Returns
    -------
    float or array like.
    '''
    try:
        sqerr = np.empty(np.shape(scf)[1])
    except IndexError:
        sqerr = 0.
    
    for t in range(1, k + cutoff):
        sqerr += (scf[k+t] + scf[abs(k-t)] - 2 * scf[k] * scf[t])**2
    return np.sqrt(sqerr / N)
#


def scf_err(scf, N, tmax=None, cutoff=100, plot=False):
    '''
    Error of the whole self-correlation function.
    This simply calls scf_err_k many times.

    Parameters
    ----------
    scf : array like
        The auto-correlation function to compute the error for.
    N : int
        Size of full dataset from which the ACF was calculated.
        This is in general different from len(scf).
    tmax : int or None (default None)
        Maximum lag to compute the error for. If not provided, it will 
        be the largest possible lag computable given cutoff and len(scf).
    cutoff : int (default = 100)
        Quite technical... Please refer to scf_err_k documentation.
    plot : boolean (default = False)
        Wether to plot the ACF and its error.

    Returns
    -------
    scf_err : array like
    '''
    kmax = len(scf)
    if tmax == None or tmax > ((kmax-cutoff) // 2):
        tmax = (kmax - cutoff) // 2
        if tmax < 0:
            raise ValueError("kmax > cutoff")
    
    try:
        scf_errf = np.empty((tmax, np.shape(scf)[1]))
    except IndexError:
        scf_errf = np.empty(tmax)

    scfnorm = scf / scf[0] #just to make sure it is normalised
    for k in range(tmax):
        scf_errf[k] = scf_err_k(scfnorm, k, N, cutoff=cutoff)

    if plot == True:
        import matplotlib.pyplot as plt
        try:
            cols = np.shape(scf_errf)[1]
        except IndexError:
            cols = 1
        xxx = np.arange(tmax)
        for i in range(cols):
            try:
                plt.errorbar(xxx, scfnorm[:tmax, i], yerr=scf_errf[:, i])
            except IndexError:
                plt.errorbar(xxx, scfnorm[:tmax], yerr=scf_errf[:])

        plt.plot(xxx, np.zeros(tmax))
        plt.show()
    
    return scf_errf
#


def scf_cutoff(scf, scf_err):
    '''
    Cutoff for the computation of the integrated auto-correlation time.
    Please refer to appendix E in Luscher, hep-lat/0409106 for discussion.
    
    Parameters
    ----------
    scf : one dimensional array
        Autocorrelation function.
    scf_err : one dimensional array
        Error on scf.

    Returns
    -------
    cutoff : int
    '''
    for cutoff in range(len(scf_err)):
        if scf[cutoff] - scf_err[cutoff] < 0:
            return cutoff
#


def tauint(scf, scf_err, col='all', cutoff=None, N=None, full_output=False):
    '''
    Computers the integrated auto-correlation time.
               1       W     C(k)
        tau = --- + Sum     ------
               2       k=1   C(0)
    Parameters
    ----------
    scf : array like
        Auto-correlation function.
    scf_err: array like
        Error on the ACF.
    col : tuple of int or 'all' (default)
        Column for which to compute tau.
    cutoff : int or None
        The maximum lag W at wich to stop the integration of the 
        normalised ACF. If None, it will be computed by means of scf_cutoff.
    N : int or None
        If not None, it is the size of the dataset from which the ACF is
        computed and triggers the computation of the error on the integrated
        auto-correlation time:
                      2(2W + 1)
            dtau^2 = ----------- tau^2
                          N
    full_output : boolean (default = False)
        Wether to return extensive output. See below.

    Returns
    -------
    (tau, dtau, cutoff, col) if full_output is True
    (tau, dtau) if not.

    tau : int or array of int
    dtau : float or none, in an array if more than one column is considered.
    cutoff : int or array of int
    col : array of int
    '''

    scfnorm = scf / scf[0] # just to make sure it's normalised

    if type(col) is str:
        if col == 'all':
            try:
                col = np.arange(np.shape(scf)[1])
            except IndexError:
                col = [0]
        else:
            raise ValueError("col must be int, tuple of int or 'all'")

    if cutoff == None:
        cutoff = []
        for c in col:
            try:
                cutoff.append(scf_cutoff(scfnorm[:, c], scf_err[:, c]))
            except IndexError:
                cutoff.append(scf_cutoff(scfnorm[:], scf_err[:]))
        cutoff = np.array(cutoff)
   
    tau = np.empty(len(col))
    for i in range(len(col)):
        tau[i] = .5 + np.sum(scfnorm[1: cutoff[i]+1])
    # error
    if N != None:
        dtau = np.sqrt((4 * cutoff + 2) * tau**2 / float(N))
    else:
        dtau = None

    if full_output:
        return (tau, dtau, cutoff, col)
    else:
        return (tau, dtau)
#


def error(var, N=1, bsize=1, selfcor=0., data=None):
    '''
    Error based on Andrea Pelissetto's advice: variance and k=1 self-
    correlation correction.
                bsize
        dx^2 = ------- (Var + 2 * C(1))
                  N

    Parameters
    ----------
    var : float or array
        Variance.
    N : int (default = 1)
        Size of the original data before blocking!
    bsize : int or array (default = 1)
        Size of the blocks.
    selfcor : float or array (default = 0) or 'auto'
        Auto-correlation at lag 1, C(1).
        If 'auto' is set, data can be provided and C(1) will be
        estimated by calling scf_k.
    data : array
        The (possibly blocked) data. This is only needed if selfcor is in 'auto'
        mode.
    '''
    N = float(N)

    if selfcor == 'auto' and data != None:
        selfcor = scf_k(data)
        
    return np.sqrt((bsize) / N * (var + 2 * selfcor))
#


def jackknife(data, n=None):
    '''
    Jackknife mean and error.
    The mean is improved to reduce bias following Andrea Pelissetto's advice.
    Calling X_i the Jackknife partial means, 
                        n - 1
        X_jk = n <X> - ------- sum X_i
                          n      n

                   n - 1
        dX^2_jk = ------- sum_i (X_i - X_jk)^2
                     n

    Parameters
    ----------
    data : array like
        Data, possibly blocked.
    n : int 
        len(data). If not provided, it will be calculated.

    Returns
    -------
    (mean, dX)

    mean : float or array.
    dX : float or array.
    '''

    if n == None:
        n = len(data)

    dsum = np.sum(data, axis=0)
    dmean = dsum / float(n)
    try:
        jacksample = np.empty((n, np.shape(data)[1]))
    except IndexError:
        jacksample = np.empty(n)
    for i in range(n):
        jacksample[i] = dsum - data[i]
    jacksample /= float(n - 1)
    
    n = float(n)
    mean = n * dmean - (n - 1.) / n * np.sum(jacksample, axis=0)
    sqerr = (n - 1.) / n * np.sum((jacksample - mean)**2, axis=0)
    return (mean, np.sqrt(sqerr))
#


def scf_analysis(data, N=None,
                 prettyprint=False, plot=False, stream=stdout,
                 *args, **kwargs):
    '''
    Comprehensive auto-correlation analysis based on auto-correlation 
    functions.

    Parameters
    ----------
    data : array like
        Data to analyse.
    N : int
        Length of original dataset, possibly different from len(data).
    prettyprint : boolean (default = False)
        Wether to print a nice summary of results.
    plot : boolean (default = False)
        Wether to plot ACF and its error.
    stream : file object (default = stdot)
        Where to pretty print to.
    *args, **kwargs :
        Additional parameters passed on to scf, scf_err and tauint.

    Returns
    -------
    (tau, ACF, ACF_err)

    tau :
        Auto-correlation time, as returned by tauint.
    ACF, ACF_err :
        Auto-correlation function and error, as returned by 
        scf and scf_err.
    '''
    
    var = np.var(data, ddof=1, axis=0)

    if N == None:
        N = float(len(data))
    acf = scf(data, *args, **kwargs)
    acf_err = scf_err(acf, N, plot=plot, *args, **kwargs)
    tau = tauint(acf, acf_err, N=N, *args, **kwargs)

    err = 2. * tau[0] * var / N
    err = np.sqrt(err)

    derr = var * tau[1] / (N * err)

    if prettyprint == True:
        hrule = "===========================================================\n"
        stream.write("\n")
        stream.write(hrule)
        stream.write("Autocorrelation analysis\n")
        stream.write(hrule)
        stream.write("Integrated autocorrelation times:\n")
        stream.write("%s\n" %(str(tau)))
        stream.write("Errors:\n")
        stream.write("%s\t%s\n" %(err, derr))
    return (tau, acf, acf_err)
#


def check_stationary(data, full_output=False):
    '''
    Checks if a series of data has reached a stationary value.
    Right now, this simply checks for the first maximum of the series.
    This can be used in blocking analyses to pick the right error.

    Parameters
    ----------
    data : array like
        Data to be analysed.
    full_output : boolean (default = False)

    Returns 
    -------
    [res, indx] if full_output is Ture
    res         otherwise.

    res : array
        Value of data at the maximum.
    indx : array of int
        Indices that yield res.
    '''
    #calculate derivative
    deriv = (np.roll(data, -1, axis=0) - data)[:-1]
    #check where it is <= 0
    deriv = np.less_equal(deriv, np.zeros(np.shape(deriv)))
    #find first indices down the rows where it happens
    indx = np.argmax(deriv, axis=0)
    #extract the values
    try:
        cols = len(indx)
        res = np.empty(cols)
        for i in range(cols):
            res[i] = data[indx[i], i]
    except TypeError:
        res = data[indx]

    if full_output:
        return [res, indx]
    else:
        return res
#


def blocking_analysis(data, N=None, nbmin=10, base=2, copy=True,
                      prettyprint=False, plot=False, stream=stdout,
                      method='error', verbose=False):
    '''
    Comprehensive blocking analysis.
    Recursively blocks data and calculate the error by calling the error
    routine.

    Parameters
    ----------
    data : array like
        Initial dataset
    N : int
        Size of initial dataset. It might differ from len(data) if a 
        preliminary blocking already took place.
        If not provided, len(data) is used.
    nbmin : int (default = 10)
        Stop blocking when there are less then nbmin blocks left.
    base : int (default = 2)
        Block size of the i-th blocking will be base**i.
    method : string (default = 'error')
        What kind of error to compute. Options are 'error' or 'jackknife'.
    copy : boolean (default = True)
        Whether to work on a copy of the data, else it will work in place.
    prettyprint : boolean (default = False)
        Pretty summary of results.
    stream : file object (default = stdout)
        Stream to pretty print to.
    plot : boolean (default = False)
        Plot how the error varies with blocking step.
    verbose : boolean (defaul = False)
        Print some status information to stderr.

    Returns
    -------
    table, error_estimate

    table : array like
        Table of errors at each blocking step.
    error_estimate : array like
        Final error estimate.
    '''
    N = float(len(data))
    bsizemax = N // nbmin
    powermax = int(np.log(bsizemax) / np.log(base)) + 1
    
    if method == 'error':
        mean = np.mean(data, axis=0)
    elif method == 'jackknife':
        jkmean = jackknife(data)
        mean = jkmean[0]
    try:
        table = np.empty((powermax, len(mean)))
    except TypeError:
        table = np.empty(powermax)

    if copy == True:
        blocked = np.copy(data)
    else:
        blocked = data
    for i in range(powermax):
        if verbose == True:
            stderr.write("Current block size: %d\n" %base**i)
        if i > 0:
            if copy == True:
                blocked = block(blocked, bsize=base, ignore_discarded='Silent')
            else:
                blocked = block(data, bsize=base**i, ignore_discarded='Silent')
        if method == 'error':
            sc1 = scf_k(blocked, 1, mean=mean)
            var = np.var(blocked, ddof=1, axis=0)
            err = error(var, N=N, bsize=base**i, selfcor=sc1)
        elif method == 'jackknife':
            err = jackknife(blocked)[1]
        table[i] = err
    error_estimate = check_stationary(table, full_output=True)
    bsizes = base**error_estimate[1]
    error_estimate[1] = bsizes

    if prettyprint == True:
        hrule = "===========================================================\n"
        dash  = "-----------------------------------------------------------\n"
        stream.write("\n")
        stream.write(hrule)
        stream.write("Blocking analysis\n")
        stream.write(hrule)
        try:
            stream.write("i\mean\t| %s\n" %("\t".join([str(x) for x in mean])))
        except TypeError:
            stream.write ("i\mean\t| %s\n" %str(mean))
        stream.write(dash)
        for i in range(powermax):
            try:
                outstring = "\t".join(str(x) for x in table[i])
            except TypeError:
                outstring = str(table[i])
            stream.write ("%d\t| %s\n" \
                          %(i, outstring))
        stream.write("\nFinal error estimate: %s\n" %str(error_estimate[0]))
        stream.write(  "   using block sizes: %s\n\n" \
                       %str(bsizes))

    if plot == True:
        import matplotlib.pyplot as plt
        xxx = np.arange(powermax)
        try:
            for i in range(len(mean)):
                plt.plot(xxx, table[:, i])
        except TypeError:
            plt.plot(xxx, table)
        plt.show()

    return table, error_estimate


def estimate_avg_err(data, minblocks=10, plot=False, verbose=False,
                     block_verbose="Warn"):
    '''
    Automatic average and error estimation.
    It blocks with every integer such that at least minblocks are left, 
    calculate the error and check where it is appropriate to stop blocking.
    The final values are computed by jackknife analysis at the block level
    previously determined.

    Parameters
    ----------
    data : 1D array
        Initial dataset
    minblocks : integer (default = 10)
        Stop blocking when less than this number of blocks are left.
    plot : boolean (default = False)
        Plot error vs. block size.
    verbose : boolean (default = False)
        Print some extra information to stderr.
    block_verbose : ("Error" | "Warn" | "Silent")
        What to do in case of BlockingErorr.

    Returns
    -------
    (mean, error) ; 2-tuple of floats
    '''
    meas = len(data)
    max_bsize = meas // minblocks + 1
    avg = np.empty(max_bsize)
    err = np.empty(max_bsize)
    for i in range(1, max_bsize + 1):
        blocked = block(data, bsize=i, ignore_discarded=block_verbose)
        avg[i-1], err[i-1] = jackknife(blocked)

    error, idx = check_stationary(err, full_output=True)
    mean = avg[idx]
    if verbose:
        bsizes = np.arange(1, max_bsize+1)
        stderr.write("Block size = %d\n" %bsizes[idx])

    if plot:
        import matplotlib.pyplot as plt
        bsizes = np.arange(1, max_bsize+1)
        plt.plot(bsizes, err) 
        plt.show()

    return mean, error



