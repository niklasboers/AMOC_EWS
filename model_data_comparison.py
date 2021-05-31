import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from EWS_functions import *

def funcfit3(x, a, b, c):
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 3)

def funcfit3_jac(x, a, b, c):
    return np.array([np.ones(x.shape[0]), -(x-c) * np.power(-b * (x - c), -2/3) / 3, b * np.power(-b * (x - c), -2/3) / 3]).T

amoc_obs = np.loadtxt('data/amoc_idx_niklas.txt')
amoc_obs = amoc_obs - amoc_obs.mean()

amoc_caesar = np.loadtxt('data/amoc_idx_caesar.txt')
amoc_caesar = amoc_caesar - amoc_caesar.mean()

time_caesar = np.arange(1871, 1871 + amoc_caesar.shape[0])

amoc_amo_obs = np.loadtxt('data/amoc-amo_idx_niklas.txt')

time_obs = np.arange(1870, 1870 + amoc_obs.shape[0])

names = ['CanESM2', 'CCSM4', 'CESM1-BGC', 'CESM1-CAM5', 'CESM1-CAM5-1-FV2', 'CNRM-CM5', 'GFDL-ESM2M', 'GISS-E2-R', 'INMCM4', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'MRI-ESM1', 'Nor-ESM1-M', 'Nor-ESM1-ME']

dat = Dataset('data/ai_Owin_Ensemble_hist-rcp85_r1i1p1.nc')
time = dat.variables['time'][:]


tempano = dat.variables['ai'][:, :]
tempano = tempano.T

dat_v = 'msf'

if dat_v == 'msf':
    psi = dat.variables['msf'][:, :]
elif dat_v == 'msf26':
    psi = dat.variables['msf26'][:, :]

psi = psi.T
sim = psi.shape[0]
for i in range(sim):
    psi[i][psi[i] < 0] = np.mean(psi[i][psi[i] > 0])



idx_c = np.logical_and(time > 1869, time < 2017)
t_c = time[idx_c]
for i in range(sim):
    p = tempano[i, idx_c]
    p0, p1 = np.polyfit(t_c, p, 1)
    print('Trend %s = %.2f K / 100yr'%(names[i], p0 * 100))

    p = psi[i, idx_c]
    p0, p1 = np.polyfit(t_c, p, 1)
    print('Trend %s = %.2f Sv / 100yr'%(names[i], p0 * 100))



idx = np.logical_and(time > 1869, time < 2019)

psi = psi[:, idx]
tempano = tempano[:, idx]
t = time[idx]



psi = psi - np.mean(psi, axis = 1).reshape(15,1)
tempano = tempano - np.mean(tempano, axis = 1).reshape(15,1)






cor_idx_strength = np.zeros(sim)
cor_idx_model_obs = np.zeros(sim)
for i in range(sim):
    cor_idx_strength[i] = np.corrcoef(psi[i], tempano[i])[0,1]
    cor_idx_model_obs[i] = np.corrcoef(tempano[i], amoc_obs)[0,1]
    print('#########')
    print('Cor(%s AMOC idx, strength) = %.3f'%(names[i], cor_idx_strength[i]))
    print('Cor(%s AMOC idx, obs) = %.3f'%(names[i], np.corrcoef(tempano[i], amoc_obs)[0,1]))
    print('Cor(%s AMOC idx, obs AMO) = %.3f'%(names[i], np.corrcoef(tempano[i], amoc_amo_obs)[0,1]))
    print('Cor(%s AMOC idx, obs Caesar) = %.3f'%(names[i], np.corrcoef(tempano[i][:-3], amoc_caesar)[0,1]))
print(np.mean(cor_idx_model_obs))

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
for i in range(sim):
    ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'o', ms = 12, label = '%s'%names[i])
plt.legend()
plt.grid()
ax.set_xlim((-.5, .8))
ax.set_xlabel('Cor(AMOC Strength, SST-based AMOC Index)')
ax.set_ylabel('Cor(modelled and observed SST-based AMOC Index)')
fig.savefig('plots/CMIP5_correlations_%s.pdf'%dat_v, bbox_inches = 'tight')






fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111)
for i in range(sim):
    p = tempano[i]
    p0, p1 = np.polyfit(t, p, 1)
    ax.plot(t, p, color = plt.cm.jet(i / 15.), alpha = .7, label = '%s: %.2f K / 100yr'%(names[i], p0 * 100))

    ax.plot(t, p0 * t + p1, color = plt.cm.jet(i / 15.), alpha = .7)
    print('Trend %s = %.2f'%(names[i], p0 * 100))


# ax.axvline(x = 2020, color = 'k')
# ax.set_ylim(10,30)
# plt.legend(loc = 2)
# ax2 = ax.twinx()

ax.plot(time_obs, amoc_obs, label = 'SST-based index', color = 'k', lw = 2)
p0, p1 = np.polyfit(time_obs, amoc_obs, 1)
ax.plot(time_obs, p0 * time_obs + p1, color = 'k', lw = 2)
ax.plot(time_obs, amoc_amo_obs, label = 'SST-based index corrected for AMO', color = 'k', ls = '--', lw = 2)
p0, p1 = np.polyfit(time_obs, amoc_amo_obs, 1)
ax.plot(time_obs, p0 * time_obs + p1, color = 'k', ls = '--', lw = 2)

plt.legend(loc = 2, ncol = 3)
ax.set_xlabel('Time [yr]')
ax.set_ylabel('SST-based AMOC index anomaly [K]')
plt.savefig('plots/CMIP5_amoc_idx.pdf', bbox_inches = 'tight')

fig = plt.figure(figsize = (15, 10))
ax = fig.add_subplot(111)
for i in range(sim):

    p = psi[i]
    print(t[0], t[-1])
    p0, p1 = np.polyfit(t, p, 1)
    ax.plot(t, p, color = plt.cm.jet(i / 15.), alpha = .7, label = '%s: %.2f Sv / 100yr'%(names[i], p0 * 100))

    ax.plot(t, p0 * t + p1, color = plt.cm.jet(i / 15.), alpha = .7)
    print('Trend %s = %.2f Sv / 100yr'%(names[i], p0 * 100))
# ax.axvline(x = 2020, color = 'k')
# ax.set_ylim(10,30)
# plt.legend(loc = 2)
# ax2 = ax.twinx()

# ax.plot(time_obs, amoc_obs, label = 'SST-based index', color = 'k', lw = 2)
# p0, p1 = np.polyfit(time_obs, amoc_obs, 1)
# ax.plot(time_obs, p0 * time_obs + p1, color = 'k', lw = 2)
# ax.plot(time_obs, amoc_amo_obs, label = 'SST-based index - AMO', color = 'k', ls = '--', lw = 2)
# p0, p1 = np.polyfit(time_obs, amoc_amo_obs, 1)
# ax.plot(time_obs, p0 * time_obs + p1, color = 'k', ls = '--', lw = 2)

plt.legend(loc = 2, ncol = 3)
ax.set_xlabel('Time [yr]')
ax.set_ylabel('AMOC strength Psi [Sv]')

plt.savefig('plots/CMIP5_amoc_strength_%s.pdf'%dat_v, bbox_inches = 'tight')



for ws in [70]:
    bound = ws // 2
    # fig = plt.figure(figsize = (8,12))
    # ax = fig.add_subplot(211)
    # # ax.plot(time_obs, runstd(amoc_obs, ws), 'k-', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 'k-', lw = 1, alpha = .4, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 1, alpha = .4)
    #
    # # ax.plot(time_obs, runstd(amoc_amo_obs, ws), 'k--', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_amo_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 1, alpha = .4, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 1, alpha = .4)
    #
    #  # ax.plot(time_caesar, runstd(amoc_caesar, ws), 'k-.', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_caesar, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index - Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .4, label = "SST-based index - Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 1, alpha = .4)
    #
    # ax.set_ylabel('STD')
    # plt.legend(loc = 2)
    # ax.set_xlim((1870, 2020))
    # ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax = fig.add_subplot(212)
    # # ax.plot(time_obs, runac(amoc_obs, ws), 'k-', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 'k-', lw = 1, alpha = .4, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 1, alpha = .4)
    #
    # # ax.plot(time_obs, runac(amoc_amo_obs, ws), 'k--', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_amo_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 1, alpha = .4, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 1, alpha = .4)
    #
    # # ax.plot(time_caesar, runac(amoc_caesar, ws), 'k-.', lw = 2, alpha = .3)
    #
    # p0, p1 = np.polyfit(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_caesar, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index - Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 1, alpha = .4, label = "SST-based index - Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 1, alpha = .4)
    # plt.legend(loc = 2)
    # ax.set_ylabel('AR(1)')
    # ax.set_xlabel('Time [yr]')
    # ax.set_xlim((1870, 2020))
    # ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # fig.savefig('plots/EWS_SST_index_ws%d.pdf'%ws, bbox_inches = 'tight')
    #
    #
    #
    #
    #
    # fig = plt.figure(figsize = (10,14))
    # ax = fig.add_subplot(211)
    # psi_sig_std = np.zeros(sim)
    # psi_sig_ac = np.zeros(sim)
    # for i in range(sim):
    #     p = psi[i]
    #
    #
    #     p0, p1 = np.polyfit(t[bound : -bound], runstd(p, ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(runstd(p, ws)[bound : -bound], 1000, p0)
    #     print ("p value AMOC std %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         psi_sig_std[i] = 1
    #         ax.plot(t[bound : -bound], runstd(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s (p = %.2f)'%(names[i], pv))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], runstd(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    # plt.legend(loc = 2, ncol = 3)
    #
    # ax.axvspan(t[0], t[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(t[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax.set_xlim((1870, 2020))
    # ax.set_ylim((.4, 3.3))
    # ax.set_ylabel('STD')
    #
    # ax = fig.add_subplot(212)
    # for i in range(sim):
    #     p = psi[i]
    #
    #     p0, p1 = np.polyfit(t[bound : -bound], runac(p, ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(runac(p, ws)[bound : -bound], 1000, p0)
    #     print ("p value AMOC ac %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         psi_sig_ac[i] = 1
    #         ax.plot(t[bound : -bound], runac(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s (p = %.2f)'%(names[i], pv))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], runac(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    #
    # plt.legend(loc = 2, ncol = 3)
    #
    # ax.axvspan(t[0], t[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(t[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax.set_xlim((1870, 2020))
    #
    # ax.set_ylim((-.3, 1.4))
    # ax.set_ylabel('AR(1)')
    #
    # ax.set_xlabel('Time [yr]')
    #
    #
    #
    # fig.savefig('plots/CMIP5_amoc_strength_EWS_ws%d.pdf'%ws, bbox_inches = 'tight')
    #
    # fig = plt.figure(figsize = (10,14))
    # ax = fig.add_subplot(211)
    # tempano_sig_std = np.zeros(sim)
    # tempano_sig_ac = np.zeros(sim)
    # for i in range(sim):
    #     p = tempano[i]
    #
    #     p0, p1 = np.polyfit(t[bound : -bound], runstd(p, ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(runstd(p, ws)[bound : -bound], 1000, p0)
    #     print ("p value AMOC std %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         tempano_sig_std[i] = 1
    #         ax.plot(t[bound : -bound], runstd(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s (p = %.2f)'%(names[i], pv))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], runstd(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    #
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_obs, ws)[bound : -bound], 'k-', alpha = .3, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_amo_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runstd(amoc_amo_obs, ws)[bound : -bound], 'k--', alpha = .3, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', alpha = .3)
    #
    # p0, p1 = np.polyfit(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runstd(amoc_caesar, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_caesar[bound : -bound], runstd(amoc_caesar, ws)[bound : -bound], 'k-.', alpha = .3, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', alpha = .3)
    #
    # plt.legend(loc = 2, ncol = 3)
    #
    # print ("p value AMOC SST-based index std: ", kendall_tau_test(runstd(amoc_obs, ws)[bound : -bound], 1000, p0))
    # print ("p value AMOC SST-based index - AMO std: ", kendall_tau_test(runstd(amoc_amo_obs, ws)[bound : -bound], 1000, p0))
    # print ("p value AMOC SST-based index Caesar std: ", kendall_tau_test(runstd(amoc_caesar, ws)[bound : -bound], 1000, p0))
    #
    # ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax.set_xlim((1870, 2020))
    # ax.set_ylim((.1, .7))
    # ax.set_ylabel('STD')
    #
    # ax = fig.add_subplot(212)
    # for i in range(sim):
    #     p = tempano[i]
    #
    #     p0, p1 = np.polyfit(t[bound : -bound], runac(p, ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(runac(p, ws)[bound : -bound], 1000, p0)
    #     print ("p value AMOC ac %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         tempano_sig_ac[i] = 1
    #         ax.plot(t[bound : -bound], runac(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s (p = %.2f)'%(names[i], pv))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], runac(p, ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    #
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_obs, ws)[bound : -bound], 'k-', alpha = .3, label = "SST-based index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', alpha = .3)
    #
    # p0, p1 = np.polyfit(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_amo_obs, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], runac(amoc_amo_obs, ws)[bound : -bound], 'k--', alpha = .3, label = "SST-based index - AMO")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', alpha = .3)
    #
    # p0, p1 = np.polyfit(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(runac(amoc_caesar, ws)[bound : -bound], 1000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_caesar[bound : -bound], runac(amoc_caesar, ws)[bound : -bound], 'k-.', alpha = .3, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', alpha = .3)
    #
    # plt.legend(loc = 2, ncol = 3)
    # ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    #
    # ax.set_ylabel('AR(1)')
    #
    # ax.set_xlabel('Time [yr]')
    #
    # print ("p value AMOC SST-based index AR1: ", kendall_tau_test(runac(amoc_obs, ws)[bound : -bound], 1000, p0))
    # print ("p value AMOC SST-based index - AMO AR1: ", kendall_tau_test(runac(amoc_amo_obs, ws)[bound : -bound], 1000, p0))
    # print ("p value AMOC SST-based index Caesar AR1: ", kendall_tau_test(runac(amoc_caesar, ws)[bound : -bound], 1000, p0))
    #
    # plt.legend(loc = 2, ncol = 3)
    # ax.set_xlim((1870, 2020))
    # ax.set_ylim((-.2, 1.3))
    # fig.savefig('plots/CMIP5_amoc_index_EWS_ws%d.pdf'%ws, bbox_inches = 'tight')
    #
    # psi_sig = psi_sig_std * psi_sig_ac
    # tempano_sig = tempano_sig_std * tempano_sig_ac
    #
    # sig = psi_sig * tempano_sig
    #
    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(111)
    # for i in range(sim):
    #     if psi_sig[i] == 1 and tempano_sig[i] == 1:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = '*', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 1 and tempano_sig[i] == 0:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 's', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 0 and tempano_sig[i] == 1:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'o', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 0 and tempano_sig[i] == 0:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'x', ms = 12, label = '%s'%names[i])
    #
    #
    # plt.legend()
    # ax.set_xlim((-.5, .8))
    # plt.grid()
    # ax.set_xlabel('Cor(modelled AMOC Strength, modelled SST-based AMOC Index)')
    # ax.set_ylabel('Cor(modelled SST-based AMOC Index, observed SST-based AMOC Index)')
    # fig.savefig('plots/CMIP5_correlations_EWS_ws%d.pdf'%ws, bbox_inches = 'tight')

    psi_sig = np.zeros(sim)
    tempano_sig = np.zeros(sim)

    # fig = plt.figure(figsize = (6, 8))
    # ax = fig.add_subplot(211)
    # psi_sig_std = np.zeros(sim)
    # psi_sig_ac = np.zeros(sim)
    # for i in range(sim):
    #     p = psi[i]
    #     popt, cov = curve_fit(funcfit3, t, p, p0 = [-1,.1,2020], maxfev = 1000000000)
    #     p0, p1 = np.polyfit(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(run_fit_a_ar1(p- funcfit3(t, *popt), ws)[bound : -bound], 10000, p0)
    #     print ("p value AMOC lambda %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         psi_sig[i] = 1
    #         ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s'%(names[i]))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    # # plt.legend(loc = 2, ncol = 3)
    # plt.legend(bbox_to_anchor=(1.04, .99), fontsize = 'small')
    #
    # ax.axvspan(t[0], t[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(t[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax.set_xlim((1870, 2020))
    # # ax.set_ylim((.4, 3.3))
    # ax.set_ylabel(r'Restoring rate $\lambda$ modelled AMOC strength')
    # ax.set_xlabel('Time [yr]')
    #
    #
    # ax = fig.add_subplot(212)
    # for i in range(sim):
    #     p = tempano[i]
    #     popt, cov = curve_fit(funcfit3, t, p, p0 = [-1,.1,2020], maxfev = 1000000000)
    #     p0, p1 = np.polyfit(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 1)
    #     pv = kendall_tau_test(run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 10000, p0)
    #     print ("p value AMOC std %s: %.4f"%(names[i], pv))
    #     if p0 > 0 and pv < .05:
    #         tempano_sig[i] = 1
    #         ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '%s'%(names[i]))
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
    #     else:
    #         ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .5)
    #         ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .5)
    #
    # popt, cov = curve_fit(funcfit3, time_obs, amoc_obs, p0 = [-1,.1,2020], maxfev = 1000000000)
    # p0, p1 = np.polyfit(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 1)
    # pv = kendall_tau_test(run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 10000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = "observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k-', alpha = .5, label = "observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', alpha = .5)
    #
    # popt, cov = curve_fit(funcfit3, time_obs, amoc_amo_obs, p0 = [-1,.1,2020], maxfev = 1000000000)
    # p0, p1 = np.polyfit(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 1)
    # pv = kendall_tau_test(run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 10000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "corrected observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k--', alpha = .5, label = "corrected observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', alpha = .5)
    #
    # # p0, p1 = np.polyfit(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 1)
    # # pv = kendall_tau_test(run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 10000, p0)
    # # if p0 > 0 and pv < .05:
    # #     ax.plot(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index Caesar")
    # #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # # else:
    # #     ax.plot(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 'k-.', alpha = .5, label = "SST-based index Caesar")
    # #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', alpha = .5)
    #
    # # plt.legend(loc = 2, ncol = 3)
    # plt.legend(bbox_to_anchor=(1.04, 1.05), fontsize = 'small')
    #
    # ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    # ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
    #
    # ax.set_xlim((1870, 2020))
    # # ax.set_ylim((.1, .7))
    # ax.set_ylabel('Restoring rate $\lambda$ SST-based AMOC index')
    # ax.set_xlabel('Time [yr]')
    #
    # fig.savefig('plots/CMIP5_amoc_index_Lambda_ws%d.pdf'%ws, bbox_inches = 'tight')
    #
    #
    # sig = psi_sig * tempano_sig
    #
    # fig = plt.figure(figsize = (6,8))
    # ax = fig.add_subplot(111)
    # for i in range(sim):
    #     if psi_sig[i] == 1 and tempano_sig[i] == 1:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = '*', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 1 and tempano_sig[i] == 0:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 's', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 0 and tempano_sig[i] == 1:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'o', ms = 12, label = '%s'%names[i])
    #     elif psi_sig[i] == 0 and tempano_sig[i] == 0:
    #         ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'x', ms = 12, label = '%s'%names[i])
    #
    #
    # plt.legend()
    # ax.set_xlim((-.5, .8))
    # plt.grid()
    # ax.set_xlabel('Cor(modelled AMOC Strength, modelled SST-based AMOC Index)')
    # ax.set_ylabel('Cor(modelled SST-based AMOC Index, observed SST-based AMOC Index)')
    # fig.savefig('plots/CMIP5_correlations_Lambda_ws%d.pdf'%ws, bbox_inches = 'tight')


    for i in range(sim):
        fig = plt.figure(figsize = (12, 12))
        ps = psi[i]
        popt_ps, cov = curve_fit(funcfit3, t, ps, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
        print(popt_ps)
        temp = tempano[i]
        popt_temp, cov = curve_fit(funcfit3, t, temp, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
        print(popt_temp)
        popt_amoc_obs, cov = curve_fit(funcfit3, time_obs, amoc_obs, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
        popt_amoc_amo_obs, cov = curve_fit(funcfit3, time_obs, amoc_amo_obs, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
        ax = fig.add_subplot(321)
        ax.plot(t, ps, 'k-', label = 'AMOC strength %s'%names[i])
        ax.plot(t, funcfit3(t, *popt_ps), 'r-')
        plt.legend()
        ax = fig.add_subplot(322)
        ax.plot(t, temp, 'k-', label = 'AMOC index %s'%names[i])
        ax.plot(t, funcfit3(t, *popt_temp), 'r-')
        ax.plot(time_obs, amoc_obs, 'b-', label = 'observed AMOC index')
        ax.plot(time_obs, amoc_amo_obs, 'b--', label = 'corrected observed AMOC index')
        plt.legend()
        ax = fig.add_subplot(323)
        ax.plot(t, ps - funcfit3(t, *popt_ps), 'k-', label = 'fluctuations')
        plt.legend()
        ax = fig.add_subplot(324)
        ax.plot(t, temp - funcfit3(t, *popt_temp), 'k-', label = 'fluctuations')
        plt.legend()
        ax = fig.add_subplot(325)
        ax.plot(t[bound:-bound], run_fit_a_ar1(ps - funcfit3(t, *popt_ps), ws)[bound:-bound], 'k-', label = r'$\lambda$')
        plt.legend()
        ax = fig.add_subplot(326)
        ax.plot(t[bound:-bound], run_fit_a_ar1(temp - funcfit3(t, *popt_temp), ws)[bound:-bound], 'k-', label = r'$\lambda$')
        ax.plot(time_obs[bound:-bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt_amoc_obs), ws)[bound:-bound], 'b-', label = r'$\lambda$ obs')
        ax.plot(time_obs[bound:-bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt_amoc_amo_obs), ws)[bound:-bound], 'b--', label = r'$\lambda$ obs mod')
        plt.legend()
        plt.savefig('plots/models/AMOC_%s_%s.pdf'%(names[i], dat_v), bbox_iches = 'tight')


    fig = plt.figure(figsize = (6, 12))
    ax = fig.add_subplot(311)
    ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    psi_sig_std = np.zeros(sim)
    psi_sig_ac = np.zeros(sim)
    for i in range(sim):
        p = psi[i]
        popt, cov = curve_fit(funcfit3, t, p, p0 = [-1,.1,2020], maxfev = 1000000000, jac = funcfit3_jac)
        p0, p1 = np.polyfit(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 1)
        pv = kendall_tau_test(run_fit_a_ar1(p- funcfit3(t, *popt), ws)[bound : -bound], 10000, p0)
        print ("p value AMOC lambda %s: %.4f"%(names[i], pv))
        if p0 > 0 and pv < .05:
            psi_sig[i] = 1
            ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '*%s'%(names[i]))
            ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
        else:
            ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
            ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .3)
    # plt.legend(loc = 2, ncol = 3)
    plt.legend(bbox_to_anchor=(1.04, 1.), fontsize = 'small')

    ax.axvspan(t[0], t[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    ax.axvspan(t[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')

    ax.set_xlim((1870, 2020))
    # ax.set_ylim((.4, 3.3))
    ax.set_ylabel(r'Restoring rate $\lambda$ (modelled AMOC strength)')
    ax.set_xlabel('Time [yr]')


    ax = fig.add_subplot(312)
    ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    for i in range(sim):
        p = tempano[i]
        popt, cov = curve_fit(funcfit3, t, p, p0 = [-1,.1,2020], maxfev = 1000000000, jac = funcfit3_jac)
        p0, p1 = np.polyfit(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 1)
        pv = kendall_tau_test(run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], 10000, p0)
        print ("p value AMOC std %s: %.4f"%(names[i], pv))
        if p0 > 0 and pv < .05:
            tempano_sig[i] = 1
            ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), lw = 2, label = '*%s'%(names[i]))
            ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), lw = 2)
        else:
            ax.plot(t[bound : -bound], run_fit_a_ar1(p - funcfit3(t, *popt), ws)[bound : -bound], color = plt.cm.jet(i / 15.), label = '%s'%names[i], alpha = .3)
            ax.plot(t[bound : -bound], p0 * t[bound : -bound] + p1, color = plt.cm.jet(i / 15.), ls = ':', alpha = .5)

    popt, cov = curve_fit(funcfit3, time_obs, amoc_obs, p0 = [-1,.1,2020], maxfev = 1000000000, jac = funcfit3_jac)
    p0, p1 = np.polyfit(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 1)
    pv = kendall_tau_test(run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 10000, p0)
    if p0 > 0 and pv < .05:
        ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k-', lw = 2, alpha = .9, label = r"observed SST$_{SG-GM}$")
        ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', lw = 2, alpha = .9)
    else:
        ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k-', alpha = .5, label = r"observed SST$_{SG-GM}$")
        ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k-', alpha = .5)

    popt, cov = curve_fit(funcfit3, time_obs, amoc_amo_obs, p0 = [-1,.1,2020], maxfev = 1000000000, jac = funcfit3_jac)
    p0, p1 = np.polyfit(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 1)
    pv = kendall_tau_test(run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 10000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k--', lw = 2, alpha = .9, label = "corrected observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_obs[bound : -bound], run_fit_a_ar1(amoc_amo_obs - funcfit3(time_obs, *popt), ws)[bound : -bound], 'k--', alpha = .5, label = "corrected observed index")
    #     ax.plot(time_obs[bound : -bound], p0 * time_obs[bound : -bound] + p1, 'k--', alpha = .5)

    # p0, p1 = np.polyfit(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 1)
    # pv = kendall_tau_test(run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 10000, p0)
    # if p0 > 0 and pv < .05:
    #     ax.plot(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 'k-.', lw = 2, alpha = .9, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', lw = 2, alpha = .9)
    # else:
    #     ax.plot(time_caesar[bound : -bound], run_fit_a_ar1(amoc_caesar, ws)[bound : -bound], 'k-.', alpha = .5, label = "SST-based index Caesar")
    #     ax.plot(time_caesar[bound : -bound], p0 * time_caesar[bound : -bound] + p1, 'k-.', alpha = .5)

    # plt.legend(loc = 2, ncol = 3)
    plt.legend(bbox_to_anchor=(1.04, 1.05), fontsize = 'small')

    ax.axvspan(time_obs[0], time_obs[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    ax.axvspan(time_obs[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')

    ax.set_xlim((1870, 2020))
    # ax.set_ylim((.1, .7))
    ax.set_ylabel('Restoring rate $\lambda$ (AMOC index)')
    ax.set_xlabel('Time [yr]')




    sig = psi_sig * tempano_sig


    ax = fig.add_subplot(313)
    ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
    for i in range(sim):
        if psi_sig[i] == 1 and tempano_sig[i] == 1:
            ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = '*', linestyle = 'None', ms = 12, label = '%s'%names[i])
        elif psi_sig[i] == 1 and tempano_sig[i] == 0:
            ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 's', linestyle = 'None', ms = 12, label = '%s'%names[i])
        elif psi_sig[i] == 0 and tempano_sig[i] == 1:
            ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'o', linestyle = 'None', ms = 12, label = '%s'%names[i])
        elif psi_sig[i] == 0 and tempano_sig[i] == 0:
            ax.plot(cor_idx_strength[i], cor_idx_model_obs[i], color = plt.cm.jet(i / 15.), marker = 'x', linestyle = 'None', ms = 12, label = '%s'%names[i])


    plt.legend(bbox_to_anchor=(1.04, 1.), fontsize = 'small')
    # ax.set_xlim((-.5, .8))
    plt.grid()
    ax.set_xlabel('Cor(modelled AMOC Strength, modelled AMOC Index)')
    ax.set_ylabel('Cor(modelled AMOC Index, observed AMOC Index)')
    fig.savefig('plots/CMIP5_correlations_Lambda_ws%d_3panels_cor%s.pdf'%(ws, dat_v), bbox_inches = 'tight')
