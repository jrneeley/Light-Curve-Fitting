import lcv_fitting_beta as l

lcv_name = 'test_lcvs/OGLE-BLG-12088'
f, data = l.read_lcv(lcv_name+'.lcv')
V_band = data[data['filter'] == 'I']
period, snr = l.period_search(V_band, error_threshold=0.03)
lcv_params, t0 = l.gloess_auto(data, period, lcv_name, clean=0, error_threshold=0.03, master_plot=0)

l.raw_lcv(data, period, t0, lcv_name, band='I')
