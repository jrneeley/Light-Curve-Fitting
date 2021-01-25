import numpy as np
import re
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from scipy import stats
#import peakutils
from astropy.stats import sigma_clip
import time
import seaborn as sns
import numpy.ma as ma
from gatspy import periodic
import warnings
warnings.simplefilter("ignore")
#warnings.filterwarnings('ignore', category=UserWarning, append=True)

sns.set_theme()

# reads in light curve data from a file
def read_lcv(lcv_file):

    dtype = np.dtype([('filter', 'U4'), ('mjd', float), ('mag', float),
        ('emag', float)])
    data = np.loadtxt(lcv_file, dtype=dtype)

    filters = np.unique(data['filter'])

    return filters, data

# phases a light curve with a given period - produces plot and text file if prompted
def phase_lcv(lcv_data, lcv_name, period, T0=0, output_file=None, plot=1, save=0, \
        error_threshold=0.05, save_dir='', ax=None, errorbars=True):

    phase_all = np.mod((lcv_data['mjd']-T0)/period, 1)
    select_err = lcv_data['emag'] < error_threshold

    phase = phase_all[select_err]
    mag = lcv_data['mag'][select_err]
    mjd = lcv_data['mjd'][select_err]
    err = lcv_data['emag'][select_err]
    band = lcv_data['filter'][select_err]



    if output_file != None:
        phased_file = output_file
        f = open(output_file, 'w')
        f.write('# period = {}\n'.format(period))
        f.write('# band mjd phase mag err\n')
        ph_dtype = np.dtype([('filter', 'U4'), ('mjd', float), ('phase', float),
            ('mag', float), ('emag', float)])
        data_save = np.zeros(len(lcv_data['filter']), dtype=ph_dtype)
        data_save['filter'] = lcv_data['filter']
        data_save['mjd'] = lcv_data['mjd']
        data_save['phase'] = phase_all
        data_save['mag'] = lcv_data['mag']
        data_save['emag'] = lcv_data['emag']

        np.savetxt(f, data_save, fmt='%4s %10.4f %8.6f %6.3f %5.3f')
        f.close()

    if plot == 1:
        if ax != None:
            if errorbars == True:
                ax.errorbar(phase_all, lcv_data['mag'], yerr=lcv_data['emag'],
                    ecolor='black', fmt='none', elinewidth=0.5)
            sns.scatterplot(x=phase_all, y=lcv_data['mag'], hue=lcv_data['filter'],
                style=lcv_data['filter'], ax=ax)

        else:
            fig, ax = plt.subplots(1,1)
            if errorbars == True:
                ax.errorbar(phase_all, lcv_data['mag'], yerr=lcv_data['emag'],
                    ecolor='black', fmt='none', elinewidth=0.5)
            sns.scatterplot(x=phase_all, y=lcv_data['mag'], hue=lcv_data['filter'],
                style=lcv_data['filter'], ax=ax)
            ax.invert_yaxis()


def get_gatspy_period(lcv_data, period_range=[0.2, 0.9], plot=1, fast=True,
    error_threshold=0.1):

    if fast == True:
        model = periodic.LombScargleMultibandFast(fit_period=True)
    else:
        model = periodic.LombScargleMultiband(fit_period=True)
    model.optimizer.period_range=(period_range[0], period_range[1])
    good = lcv_data['emag'] < error_threshold
    model.fit(lcv_data['mjd'][good], lcv_data['mag'][good],
        lcv_data['emag'][good], lcv_data['filter'][good])
    period_step = int((period_range[1] - period_range[0])/0.0001)
    periods = np.linspace(period_range[0], period_range[1], period_step+1)

    P_multi = model.periodogram(periods)

    if plot == 1:
        fig, ax = plt.subplots(1,2, figsize=(14,8))

        phase = np.mod(lcv_data['mjd']/model.best_period, 1)
        sns.scatterplot(x=phase[good], y=lcv_data['mag'][good],
            hue=lcv_data['filter'][good], ax=ax[0])
        ax[0].invert_yaxis()
        sns.lineplot(x=periods, y=P_multi, ax=ax[1])
        x_text = model.best_period
        y_text = np.max(P_multi)
        ax[1].annotate('Best period = {:.4f}'.format(model.best_period),
            xy=(x_text, y_text), xycoords='data', xytext=(30, -15),
            textcoords='offset points', arrowprops=dict(arrowstyle='->',
            color='black'), ha='left', va='center')
        ax[0].set_xlabel('Phase')
        ax[0].set_ylabel('mag')

        ax[1].set_xlabel('Period')
        ax[1].set_ylabel('Power')

    return model.best_period




# Perform GLOESS smoothing of cleaned light curve
def lcv_gloess(lcv_data, period, make_plot=False, band=0, sigma_thresh=5,
    max_delta_phase=0.8, fit_step=0.001, max_amp_err=0.2, min_n_obs=30, ax=None):

    """
    Perform Gloess smoothing and calculate average magnitude of a light curve

    Parameters
    ----------
    lcv_data: array_like, float
        Output from read_lcv. Includes columns for filter, mjd, mag, and emag
        * NEEDS TO BE SINGLE FILTER

    phase: array_like, float
        The array with the variability phase (0 through 1) at each epoch.
        Note that the zero point epoch for the phase has been determined
        randomly for each time series (i.e. phase == 0 is not minimum
        light, but just a random time in the time series).
    w_flag: bool
        It is set True if 50% or more of not rejected epochs in the light
        curve have some AllWISE multiepoch photometry quality flag set to
        a less than optimal value.
    ax: Axes
        Axes class elements generated by a mathplotlib.subplot command,
        specifying where to plot Gloess light curve. If no plot is desired,
        this can be set to any scalar value (e.g. False).
    make_plot: bool, optional
        If true add Gloess smoothed lightcurve to phased time series. Add
        also best fit parameters (mean magnitude and error, amplitude, phase
        of minimum, fit quality flags, etc) to the same plot. Default is False.
    band: int, optional
        Band that is being processed. This is only used for the runtime log
        file messages, so it can be any value (default is '0').

    Returns
    -------
    lcv_params: array-like
        1D array with the Gloess fit parameters.
    lcv_qual: string
        Gloess light curve fit quality flag.


    Notes
    -----
    This function fits the provided phased light curve (processed by the
    routine make_lcv) with the Gloess method [1], and then calculates and
    eturns the following parameters:

        lcv_params['avg']: mean magnitude of the Gloess light curve.
        lcv_params['avg_e']: mean magnitude error.
        lcv_params['amp']: Gloess light curve amplitude.
        lcv_params['chisq']: Gloess fit chi squared.
        lcv_params['sigma']: width of Gaussiam smoothing window.
        lcv_params['T0']: epoch of minimum determined from Gloess light curve.
        lcv_params['ph_min']: phase of minimum light.
        lcv_params['ph_max']: phase of maximum light.

    The routine also returns a quality flag of the Gloess fit (and mean
    magnitude). The flag can have one fo the following values:

        - A: good quality Gloess fit and average magnitude
        - B: successful Gloess fit but scatter around the fit is above
             defined threshold
        - C: poor Gloess fit and average magnitude
        - F: failed Gloess fit
        - U: the source was missing from the AllWISE catalog


    References
    ----------

    [1] Neeley, J.R. et al. 2015, ApJ, 808, 11


    """

    # Set up parameter array
    dtype = np.dtype([('avg', float), ('avg_e', float), ('amp', float),
                      ('chisq', float), ('sigma', float), ('T0', float),
                      ('ph_min', float), ('ph_max', float),('std_res', float)])
    lcv_params = np.zeros(1, dtype=dtype)

    dt = np.dtype([('phase', float), ('mag', float)])
    fit_data = np.zeros(int(1/fit_step), dtype=dt)



    # Check for nan values and remove them
    mjd = lcv_data['mjd'][~np.isnan(lcv_data['emag'])]
    mag = lcv_data['mag'][~np.isnan(lcv_data['emag'])]
    err = lcv_data['emag'][~np.isnan(lcv_data['emag'])]
    phase = np.mod(mjd/period, 1)

    # Set quality of fit to 'A' (will be changed in case of errors)
    lcv_qual = 'A'


    # Remove data with large error bars
    # This removes outliers, i.e. data with much larger error than the rest
    filtered_err = sigma_clip(err, sigma=sigma_thresh, maxiters=2)
    mag_rejected = mag[filtered_err.mask]
    phase_rejected = phase[filtered_err.mask]
    err_rejected = err[filtered_err.mask]
    mjd_rejected = mjd[filtered_err.mask]
    mag = mag[~filtered_err.mask]
    phase = phase[~filtered_err.mask]
    err = err[~filtered_err.mask]
    mjd = mjd[~filtered_err.mask]

    # remind myself what this cut is doing
    from scipy.stats import iqr
    filtered_mag=np.abs(mag-np.median(mag))-err>5*iqr(mag)
    mag_rejected = np.append(mag_rejected, mag[filtered_mag])
    phase_rejected = np.append(phase_rejected, phase[filtered_mag])
    err_rejected = np.append(err_rejected, err[filtered_mag])
    mjd_rejected = np.append(mjd_rejected, mjd[filtered_mag])
    mag = mag[~np.array(filtered_mag)]
    phase = phase[~np.array(filtered_mag)]
    err = err[~np.array(filtered_mag)]
    mjd = mjd[~np.array(filtered_mag)]


    # Skip this star if we don't have enough observations or phase coverage
    n_obs = len(mag)
    #delta_phase = np.max(phase) - np.min(phase)
    # determine largest phase gap in data
    phase_sorted = np.sort(phase)
    phase_diff = phase_sorted[1:] - phase_sorted[:-1]
    delta_phase = np.max(phase_diff)

    if (n_obs < min_n_obs):
        print('  --> {}: Not enough valid datapoint in light curve'.format(lcv_data['filter'][0]))
        lcv_qual = 'F'
        fit_data['mag'][:] == np.nan
        return lcv_params, lcv_qual, fit_data

    if (delta_phase > max_delta_phase):
        print('  --> {}: Not enough phase coverage in light curve'.format(lcv_data['filter'][0]))
        lcv_qual = 'F'
        fit_data['mag'][:] == np.nan
        return lcv_params, lcv_qual, fit_data

    # Bin phases and find average filter width
    hist, bins = np.histogram(phase, bins='auto')
    sigma = 1./len(bins)

    # Extend lcv to 5 periods to avoid edge effects (1st pass)
    phase_copy = np.concatenate((phase-2, phase-1, phase, phase+1, phase+2))
    mag_copy = np.tile(mag, 5)
    err_copy = np.tile(err, 5)
    n_data = len(mag_copy)
    weight = np.zeros(n_data)

    # Create grid for smoothed light curve (over 1 period)
    x = np.arange(0, 1, fit_step)
    n_fit = len(x)
    smoothed_mag = np.zeros(n_fit)

    # Interpolate light curve over fine grid (1st pass)
    for ind, step in enumerate(x):

        dist = phase_copy - step
        closest_phase = np.min(np.abs(dist))
        if closest_phase > sigma: sigma = closest_phase*5
        gauss_exp = dist**2/sigma**2
        gauss_exp[gauss_exp>200] = 200
        weight = err_copy * np.exp(gauss_exp)
        fit = np.polyfit(phase_copy, mag_copy, 2, w=1/weight)
        smoothed_mag[ind] = fit[2] + fit[1]*step + fit[0]*step**2

    # Derive residuals for smoothed light curve (1st pass)
    yy = np.interp(phase, x, smoothed_mag)
    residual = mag - yy

    # Mask datapoints with too large residuals for 2nd pass
    masked_mag = abs(residual/err) > sigma_thresh
    mag_rejected = np.append(mag_rejected, mag[masked_mag])
    phase_rejected = np.append(phase_rejected, phase[masked_mag])
    err_rejected = np.append(err_rejected, err[masked_mag])
    mjd_rejected = np.append(mjd_rejected, mjd[masked_mag])

    mag = ma.masked_where(masked_mag,mag)
    phase = ma.masked_where(masked_mag,phase)
    err = ma.masked_where(masked_mag,err)
    mjd = ma.masked_where(masked_mag,mjd)
    mag = ma.compressed(mag)
    phase = ma.compressed(phase)
    err = ma.compressed(err)
    mjd = ma.compressed(mjd)

    # Extend lcv to 5 periods to avoid edge effects (2nd pass)
    phase_copy = np.concatenate((phase-2, phase-1, phase, phase+1, phase+2))
    mag_copy = np.tile(mag, 5)
    err_copy = np.tile(err, 5)
    n_data = len(mag_copy)
    weight = np.zeros(n_data)

    # Interpolate light curve over fine grid (2nd pass)
    for ind, step in enumerate(x):

        dist = phase_copy - step
        closest_phase = np.min(np.abs(dist))
        if closest_phase > sigma: sigma = closest_phase*5
        gauss_exp = dist**2/sigma**2
        gauss_exp[gauss_exp>200] = 200
        weight = err_copy * np.exp(gauss_exp)
        fit = np.polyfit(phase_copy, mag_copy, 2, w=1/weight)
        smoothed_mag[ind] = fit[2] + fit[1]*step + fit[0]*step**2

    # Extend interpolated lcv over 5 periods (2nd pass)
    smoothed_mag_copy = np.tile(smoothed_mag, 5)
    x_copy = np.concatenate((x-2, x-1, x, x+1, x+2))

    # Derive average flux
    flux = 99*np.power(10,-smoothed_mag/2.5)
    average_flux = np.mean(flux)
    average_mag = -2.5*np.log10(average_flux/99)

    # Derive amplitude from smoothed light curve
    amplitude = np.max(smoothed_mag) - np.min(smoothed_mag)

    #-->>>>> Add error estimate to amplitude?


    # Derive chi2 for smoothed light curve
    yy = np.interp(phase, x, smoothed_mag)
    residual = mag - yy
    chisq = np.sum(residual**2/err**2)
    rchisq = chisq/(len(phase)-2)

    std_res = np.std(residual)

    if std_res > 0.333*amplitude:
        lcv_qual = 'B'
    if std_res > 0.5*amplitude:
        lcv_qual = 'C'
    #Repromoting if good fit
    #if (std_res<0.333*amplitude) and (std_res<max_amp_err):
    #    lcv_qual='A'
    #Putting some restrictions on the maximum amp or max amp_err in the Bailey diagram
    #if (amplitude>MAX_AMP) or (std_res>max_amp_err):
    #    lcv_qual='C'
    # Compare amplitude with 1/3 average residual: bad dataset if smaller
    #avg_res= np.sqrt(np.average(residual**2))
    #if avg_res>0.3333*amplitude:
    #    lcv_qual = 'C'
    #    print('  --> [band: '+b+'] Amplitude too small for residuals')

    # Determine epoch of minimum for this band
    ph_max = x[smoothed_mag.argmin()]
    ph_min = x[smoothed_mag.argmax()]
    T0 = mjd[np.abs(phase - ph_min).argmin()]

    # Estimate average magnitude error
    err_fit = amplitude/np.sqrt(12*len(err))
    average_mag_err = np.sqrt(np.sum(err**2)/len(err)**2 + err_fit**2)

    # Save GLOESS fit parameters
    lcv_params['avg'] = average_mag
    lcv_params['avg_e'] = average_mag_err
    lcv_params['amp'] = amplitude
    lcv_params['chisq'] = chisq
    lcv_params['sigma'] = sigma
    lcv_params['T0'] = T0
    lcv_params['ph_min'] = ph_min
    lcv_params['ph_max'] = ph_max
    lcv_params['std_res'] = std_res

    # save the gloess fit
    fit_data['phase'] = x
    fit_data['mag'] = smoothed_mag


    # Plot fitted light curve
    if make_plot:
        min_mag, max_mag = ax.get_ylim()
        avg_mag_s = "{:<6.3f}".format(average_mag)
        avg_err_s = "{:<5.3f}".format(average_mag_err)
        amplitude_s = "{:<6.3f}".format(amplitude)
        chisq_s = "{:<10.3f}".format(rchisq)
        ax.plot(phase_rejected, mag_rejected, 'x', color='k', mew=1, ms=8, zorder=9) # identify rejected GLOESS points
        ax.plot(x_copy,smoothed_mag_copy, color='g', zorder=10)
        ax.text(0.025, 0.9,
                '$\langle$mag$\\rangle$ = '+avg_mag_s+'$\pm$'+avg_err_s,
                transform=ax.transAxes)
        ax.text(0.675, 0.85, 'A = '+amplitude_s+' (mag)', transform=ax.transAxes)
        ax.text(0.025, 0.85, '$\chi^2$ = '+chisq_s, transform=ax.transAxes)
        #ax.text(0.675, 0.8, '$\sigma$ resid = {:.3f}'.format(std_res), transform=ax.transAxes)
        ax.text(0.025, 0.8, '$Q flag$ = '+str(lcv_qual), transform=ax.transAxes)
        ax.plot([-0.5,1.5],[average_mag, average_mag],
                color='g', linestyle='dashed', linewidth=1)
        ax.plot([-0.5,1.5],
                [average_mag-average_mag_err, average_mag-average_mag_err],
                color='g', linestyle='dotted', linewidth=0.5)
        ax.plot([-0.5,1.5],
                [average_mag+average_mag_err, average_mag+average_mag_err],
                color='g', linestyle='dotted', linewidth=0.5)
        ax.plot([ph_min,ph_min],[average_mag-0.05,average_mag+0.05],
                color='g',linewidth=0.5)

    # Print status message if everything ok
    if lcv_qual == 'A':
        print('  --> {}: Process completed successfully'.format(lcv_data['filter'][0]))
    if lcv_qual == 'B':
        print('  --> {}: Light curve has large scatter'.format(lcv_data['filter'][0]))
    if lcv_qual == 'C':
        print('  --> {}: Very poor light curve'.format(lcv_data['filter'][0]))


    return lcv_params, lcv_qual, fit_data
