import lcv_fitting_beta as l

f, data = l.read_lcv('test_lcvs/NGC3201_V2.lcv')
V_band = data[data['filter'] == 'V']
period, snr = l.period_search(V_band)
lcv_params, t0 = l.gloess_auto(data, period, 'V2')

l.raw_lcv(data, period, t0, 'V2')
