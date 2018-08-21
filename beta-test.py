import lcv_fitting_beta as l

lcv_name = 'test_lcvs/NGC3201_V29'
f, data = l.read_lcv(lcv_name+'.lcv')
V_band = data[data['filter'] == 'V']
period, snr = l.period_search(V_band)
lcv_params, t0 = l.gloess_auto(data, period, lcv_name)

l.raw_lcv(data, period, t0, lcv_name)
