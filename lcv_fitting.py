import numpy as np
import re
import matplotlib.pyplot as mp
from astropy.stats import LombScargle
from scipy import stats
import peakutils
from astropy.stats import sigma_clip
from IPython import display


def read_lcv(lcv_file):

    dtype = np.dtype([('filter', 'S2'), ('mjd', float), ('mag', float),
        ('err', float)])
    data = np.loadtxt(lcv_file, dtype=dtype)

    filters = np.unique(data['filter'])

    return filters, data



def phase_lcv(filters, lcv_data, lcv_name, period, T0, save=1, plot=0, error_threshold=0.3):

    if plot == 1:
        fig = mp.figure(figsize=(10,8))
        mp.ylabel('mag')
        mp.xlabel('phase')
        mp.ylim((np.max(lcv_data['mag'])+0.5, np.min(lcv_data['mag'])-0.5))
        mp.xlim(-0.2,1.2)

    for filt in filters:

        mag_all = lcv_data['mag'][lcv_data['filter'] == filt]
        err_all = lcv_data['err'][lcv_data['filter'] == filt]
        mjd_all = lcv_data['mjd'][lcv_data['filter'] == filt]

        # remove NaNs
        err_all = err_all[~np.isnan(mag_all)]
        mjd_all = mjd_all[~np.isnan(mag_all)]
        mag_all = mag_all[~np.isnan(mag_all)]

        # Filter on the uncertainty
        mag = mag_all[err_all < error_threshold]
        mjd = mjd_all[err_all < error_threshold]
        err = err_all[err_all < error_threshold]
        band = np.repeat(filt, len(mag))

        phase = np.mod((mjd - T0)/period, 1)
        if save == 1:
            phased_file = lcv_name+'.ph'
            if filt == filters[0]:
                f_handle = open(phased_file, 'w')
            else:
                f_handle = open(phased_file, 'a')
            data_save = np.array(zip(band, mjd, phase, mag, err), dtype=[('c1', 'S2'),
                ('c2', float), ('c3', float), ('c4', float), ('c5', float)])
            np.savetxt(f_handle, data_save, fmt='%s %10.4f %8.6f %6.3f %5.3f')
            f_handle.close()


        if plot == 1:
            phase = phase[~np.isnan(mag)]
            err = err[~np.isnan(mag)]
            mag = mag[~np.isnan(mag)]
            mp.errorbar(phase, mag, yerr=err, fmt='o', label=filt)
    if plot == 1:
        mp.legend()
        mp.show()


def refine_period(first_band, best_period, second_band=None,
    plot_save=0, error_threshold=0.1, search_window=0.0002, plot=0):

    x = np.array(first_band['mjd'][first_band['err'] < error_threshold], dtype=float)
    y = np.array(first_band['mag'][first_band['err'] < error_threshold], dtype=float)
    er = np.array(first_band['err'][first_band['err'] < error_threshold], dtype=float)
    if second_band is not None:
        x2 = np.array(second_band['mjd'][second_band['err'] < error_threshold], dtype=float)
        y2 = np.array(second_band['mag'][second_band['err'] < error_threshold], dtype=float)
        er2 = np.array(second_band['err'][second_band['err'] < error_threshold], dtype=float)

    # Calculate required precision
    delta_time = np.max(x) - np.min(x)
    approx_p = np.round(best_period, 1)
    n_cycles = np.floor(delta_time/approx_p)
    max_precision = n_cycles * approx_p / (n_cycles - 0.01) - approx_p
    order = np.ceil(np.abs(np.log10(max_precision)))
    precision = 10**order
    best_period = np.round(best_period, decimals=int(order))

    grid_num = search_window*precision

    if grid_num > 100000:
        grid_num = 100000
    min_period = best_period - search_window/2
    max_period = best_period + search_window/2
    periods = np.linspace(min_period, max_period, num=grid_num+1)
    avg_std = np.zeros(len(periods))
    for ind2, trial_period in enumerate(periods):
        phase = np.mod(x/trial_period, 1)
        stds, edges, bin_num = stats.binned_statistic(phase, y, statistic=np.std, bins=100)
        counts, edges, bin_num = stats.binned_statistic(phase, y, statistic='count', bins=100)
        if second_band is not None:
            phase2 = np.mod(x2/trial_period, 1)
            stds2, edges2, bin_num2 = stats.binned_statistic(phase2, y2, statistic=np.std, bins=100)
            counts2, edges2, bin_num2 = stats.binned_statistic(phase2, y2, statistic='count', bins=100)
            avg_std[ind2] = np.mean(stds2[counts2 > 3]) + np.mean(stds[counts > 3])
        else:
            avg_std[ind2] = np.mean(stds[counts > 3])
    order = np.argsort(avg_std)
    new_period = periods[order[0]]
    best_std = avg_std[order[0]]
    mp.plot(periods, avg_std, 'ro')
    mp.axvline(new_period)

    if plot == 1:
        mp.show()
    mp.close()

    return new_period

# Use to do one round of LombScargle and identify possible periods for RRL star
def period_search(data, name, min_period = 0.2, max_period=1.0,
                    error_threshold=0.05, verbose=0):

    x1 = np.array(data['mjd'][data['err'] < error_threshold], dtype=float)
    y1 = np.array(data['mag'][data['err'] < error_threshold], dtype=float)
    er1 = np.array(data['err'][data['err'] < error_threshold], dtype=float)

    freq_max = 1/(min_period)
    freq_min = 1/(max_period)
    frequency, power = LombScargle(x1, y1, er1).autopower(minimum_frequency=freq_min,
                        maximum_frequency=freq_max )

    # Calculate noise level
    median_power = np.median(power)

    # Find peaks
    indices = peakutils.indexes(power, thres=0.5, min_dist=5000 )
    candidate_periods = 1/frequency[indices]
    best_frequency = frequency[np.argmax(power)]
    best_period = 1/best_frequency

    fig = mp.figure(figsize=(12, 6))
    ax1 = mp.subplot2grid((1,2), (0,0))
    ax2 = mp.subplot2grid((1,2), (0,1))
    ax1.plot(1/frequency, power)
    ax1.plot(1/frequency[indices], power[indices], 'rx')
    alias_freq = np.array([best_frequency+1, best_frequency-1, best_frequency+2, best_frequency-2])
    alias_power = np.repeat(np.max(power), 4)
    ax1.plot(1/alias_freq, alias_power, 'kx')
    ax1.set_xlim(min_period, max_period)
    ax1.set_xlabel('Period (days)')
    ax1.set_ylabel('Power')


    # Calculate SNR of peaks
    snr = power[indices]/median_power
    snr_best = power[np.argmax(power)]/median_power
    if verbose == 1:
        print candidate_periods
        print snr
    t_fit = np.linspace(0,1)
    y_fit = LombScargle(x1, y1, er1).model(t_fit, best_frequency)

    phase_data = np.mod(x1*best_frequency, 1)
    ax2.plot(phase_data, y1, 'o')
    ax2.set_ylim(np.max(y1)+0.05, np.min(y1)-0.05)
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('mag')
    mp.show()
    mp.close()
#    return candidate_periods
    return best_period, snr_best


def gloess(phased_lcv_file, clean=1, smoothing_params=None, ask=0, filters='all', master_plot=0):

    # set to 1 if you want to save a single figure for each star with all data
    if master_plot == 1:
        figtosave = mp.figure(figsize=(8,10))
        ax = figtosave.add_subplot(111)

    master_filters = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K',
        'I1', 'I2'], dtype='S2')
    master_markers = np.array(['P', 'v', 'D', '>', 'x', 'p', 'd', '^', 'o', 's'])
    master_offset = np.array([1.0, 0.5, 0.0, -0.25, -0.5, -0.7, -0.9, -1.1, -1.0, -1.5 ])
    master_colors = np.array(['xkcd:violet', 'xkcd:periwinkle', 'xkcd:sapphire',
        'xkcd:sky blue', 'xkcd:emerald', 'xkcd:avocado', 'xkcd:goldenrod',
        'xkcd:orange', 'xkcd:pink', 'xkcd:scarlet'])
    # read in the phased light curve file
    dtype1 = np.dtype([('filter', 'S2'), ('mjd', float), ('phase', float), ('mag', float), ('err', float)])
    data = np.loadtxt(phased_lcv_file, dtype=dtype1, usecols=(0,1,2,3,4))

    # which filters are available
    if filters == 'all': filters = np.unique(data['filter'])
    num_filters = len(filters)


    if smoothing_params is None: smoothing_params = np.repeat(1.0, 10)
        #smoothing_params = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])

    master_avg_mag = np.zeros(10)
    master_amp = np.zeros(10)
    master_sigma = np.zeros(10)
    master_avg_mag_er = np.zeros(10)

    for filt in filters:

        phase = data['phase'][data['filter'] == filt]
        mag = data['mag'][data['filter'] == filt]
        err = data['err'][data['filter'] == filt]
        mjd = data['mjd'][data['filter'] == filt]

        mjd = mjd[~np.isnan(mag)]
        phase = phase[~np.isnan(mag)]
        err = err[~np.isnan(mag)]
        mag = mag[~np.isnan(mag)]

        if clean == 1:
            # remove data with large error bars
            filtered_err = sigma_clip(err, sigma=3, iters=2)
            mag = mag[~filtered_err.mask]
            phase = phase[~filtered_err.mask]
            err = err[~filtered_err.mask]

        # skip this band if we don't have enough observations or phase coverage
        n_obs = len(mag)
        if (filt == 'I1') or (filt == 'I2'): n_obs = 99
        delta_phase = np.max(phase) - np.min(phase)
        if (n_obs < 30) or (delta_phase < 0.7):
            continue
        sigma = float(smoothing_params[master_filters == filt])

        if sigma == 1.0:
            hist, bins = np.histogram(phase, bins='auto')
            sigma = 1./len(bins)

        phase_copy = np.concatenate((phase-2, phase-1, phase, phase+1, phase+2))
        mag_copy = np.tile(mag, 5)
        err_copy = np.tile(err, 5)

        x = np.arange(0, 1, 0.001)

        n_data = len(mag_copy)
        n_fit = len(x)

        happy = 'n'
        while happy == 'n':

            smoothed_mag = np.zeros(n_fit)
            weight = np.zeros(n_data)

            # interpolate
            for ind, step in enumerate(x):

                dist = phase_copy - step
                closest_phase = np.min(np.abs(dist))
                if closest_phase > sigma: sigma = closest_phase*5
                weight = err_copy * np.exp(dist**2/sigma**2)
                fit = np.polyfit(phase_copy, mag_copy, 2, w=1/weight)
                smoothed_mag[ind] = fit[2] + fit[1]*step + fit[0]*step**2
            #    print step, np.min(np.abs(dist))

            smoothed_mag_copy = np.tile(smoothed_mag, 5)
            x_copy = np.concatenate((x-2, x-1, x, x+1, x+2))

            figshow = mp.figure(figsize=(8,6))
            ax2 = figshow.add_subplot(111)
            ax2.errorbar(phase_copy, mag_copy, yerr=err_copy, fmt='o', zorder=1)
            ax2.plot(x_copy, smoothed_mag_copy, 'r-')
            ax2.set_ylim((np.max(mag)+0.2, np.min(mag)-0.2))
            ax2.set_xlim((-0.2, 1.2))
            ax2.set_xlabel('Phase')
            ax2.set_ylabel(filt+' mag')
            display.display(mp.gcf())
            display.clear_output(wait=True)

        #    if smoothing_params[master_filters == filt] != 1.0:
            if ask == 0:
                happy = 'y'
                plot_file = re.sub('.ph', '-'+filt+'fit.pdf', phased_lcv_file)
                mp.savefig(plot_file, format='pdf')
                continue
            check = raw_input('Are you happy with this fit? [y/n]: ')
            if check == 'n':
                sigma = input('Enter new smoothing parameter: ')
            if check == 'y':
                happy = 'y'

        mp.close()
        # Derive light curve parameters
        flux = 99*np.power(10,-smoothed_mag/2.5)
        average_flux = np.mean(flux)
        average_mag = -2.5*np.log10(average_flux/99)

        amplitude = np.max(smoothed_mag) - np.min(smoothed_mag)
        ph_max = x[smoothed_mag == np.min(smoothed_mag)]
        ph_min = x[smoothed_mag == np.max(smoothed_mag)]

        err_fit = amplitude/np.sqrt(12*len(err))
        average_mag_err = np.sqrt(np.sum(err**2)/len(err)**2 + err_fit**2)

        # determine the epoch of maximum using the V band data
        if filt == 'V':
            T0 = ph_max

        master_avg_mag[master_filters == filt] = average_mag
        master_amp[master_filters == filt] = amplitude
        master_sigma[master_filters == filt] = sigma
        master_avg_mag_er[master_filters == filt] = average_mag_err

        fit_file = re.sub('.ph', '.fit', phased_lcv_file)
        if filt == filters[0]:
            f = open(fit_file, 'w')
        else:
            f = open(fit_file, 'a')
        dtype= np.dtype([('filt', 'S2'), ('x', float), ('mag', float)])
        data_save = np.array(zip(np.repeat(filt, len(x)), x, smoothed_mag), dtype=dtype)
        np.savetxt(f, data_save, fmt='%2s %.4f %2.3f')
        f.close()

        if master_plot == 1:

            offset = master_offset[master_filters == filt]
            marker = master_markers[master_filters == filt][0]
            color = master_colors[master_filters == filt][0]
            ax.errorbar(phase, mag+offset, yerr=err, fmt=marker, color=color, zorder=1)
            ax.plot(x_copy, smoothed_mag_copy+offset, 'k-')

    if master_plot == 1:

        max_mag = np.nanmean(data['mag'][data['filter'] == 'V'])+3
        min_mag = np.nanmean(data['mag'][data['filter'] == 'V'])-5
        ax.set_ylim((max_mag, min_mag))
        ax.set_xlim((-0.2, 2.0))
        ax.set_xlabel('Phase')
        ax.set_ylabel('Mag + offset')
        plot_file = re.sub('\.ph', '-fit.pdf', phased_lcv_file)
        mp.savefig(plot_file)

    return master_filters, master_avg_mag, master_avg_mag_er, master_amp, master_sigma
