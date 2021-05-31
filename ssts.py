from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.regression.linear_model as sm
import scipy.stats as st
import statsmodels.api as sm2
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat

from lagcorr import lagcorr
from EWS_functions import *

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point

from shiftgrid import shiftgrid

def runmean(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xs[i] = np.mean(x[: i + w // 2 + 1])
   for i in range(n - w // 2, n):
      xs[i] = np.mean(x[i - w // 2 + 1:])

   for i in range(w // 2, n - w // 2):
      xs[i] = np.mean(x[i - w // 2 : i + w // 2 + 1])
   return xs

def coordinate_indices_from_ra(lat, lon, lat_max, lat_min, lon_max, lon_min):
    la = len(lat)
    lo = len(lon)
    n = la * lo
    indices = np.arange(n).reshape((la, lo))
    lat_max = lat[np.argmin(np.abs(lat - lat_max))]
    lat_min = lat[np.argmin(np.abs(lat - lat_min))]
    lon_max = lon[np.argmin(np.abs(lon - lon_max))]
    lon_min = lon[np.argmin(np.abs(lon - lon_min))]
    ra_indices = indices[np.where(lat == lat_max)[0][0] : np.where(lat == lat_min)[0][0] + 1 , np.where(lon == lon_min)[0][0] : np.where(lon == lon_max)[0][0] + 1 ]
    return np.unique(ra_indices.flatten())


def trmm_indices_for_area(area_coords, trmm_lat, trmm_lon):
   la = len(trmm_lat)
   lo = len(trmm_lon)
   n = la * lo
   indices = np.arange(n).reshape((la, lo))
   trmm_coords = np.transpose([np.repeat(trmm_lat, len(trmm_lon)), np.tile(trmm_lon, len(trmm_lat))])
   lat_max = trmm_lat[np.argmin(np.abs(trmm_lat - np.max(area_coords[:, 0])))]
   lat_min = trmm_lat[np.argmin(np.abs(trmm_lat - np.min(area_coords[:, 0])))]
   lon_max = trmm_lon[np.argmin(np.abs(trmm_lon - np.max(area_coords[:, 1])))]
   lon_min = trmm_lon[np.argmin(np.abs(trmm_lon - np.min(area_coords[:, 1])))]
   ra_indices = indices[np.where(trmm_lat == lat_max)[0][0] : np.where(trmm_lat == lat_min)[0][0] + 1 , np.where(trmm_lon == lon_min)[0][0] : np.where(trmm_lon == lon_max)[0][0] + 1 ]
   ra_indices = ra_indices.flatten()
   trmm_ra_coords = trmm_coords[ra_indices]
   d = np.zeros((len(area_coords), len(ra_indices)))
   for i in range(len(area_coords)):
      for j in range(len(ra_indices)):
         d[i, j] = np.sum(np.abs(area_coords[i] - trmm_ra_coords[j]))
   trmm_indices_area = ra_indices[np.argmin(d, axis = 1)]
   return trmm_indices_area


def funcfit2(x, a, b, c):
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 2)

def funcfit3(x, a, b, c):
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 3)

def funcfit3_jac(x, a, b, c):
    return np.array([np.ones(x.shape[0]), -(x-c) * np.power(-b * (x - c), -2/3) / 3, b * np.power(-b * (x - c), -2/3) / 3]).T


rapid = Dataset('data/moc_transports_yearly.nc')
rapid_time = rapid.variables['time'][:]
rapid_moc = rapid.variables['moc_mar_hc10'][:]
rapid_time = np.arange(2004, 2019,1)

glosea5 = Dataset('data/GloSea5_timeseries_amoc/moc26.nc')
# print(glosea5.variables)
glosea5_time = glosea5['t_GLOSEA5'][:]
glosea5_amoc = glosea5['M26_GLOSEA5'][:]
# print(glosea5_time)

glosea_years = np.arange(1993, 2017, 1)
glosea_amoc_ann = np.zeros(glosea_years.shape[0])
for i in range(glosea_years.shape[0]):
    glosea_amoc_ann[i] = np.nanmean(glosea5_amoc[i * 12 : (i + 1) * 12])

glosea5_amoc_rm = runmean(glosea5_amoc, 12)

fm_amoc = loadmat('data/Frajka-Williams_MOCproxy_for_figshare_v1.0.mat')
# print(fm_amoc.keys())
# print(fm_amoc['recon'][0][0])
# print(fm_amoc['mocgrid'][0][0][4])
fm_time = np.arange(1993, 2015, 1/12)

fm_years = np.arange(1993, 2015, 1)
fm_amoc_ann = np.zeros(fm_years.shape[0])
for i in range(fm_years.shape[0]):
    fm_amoc_ann[i] = np.nanmean(fm_amoc['recon'][0][0][3][0][i * 12 : (i + 1) * 12])



chentung = np.loadtxt('data/chen_tung_2018.txt')
chentung_time = chentung[:, 0]
chentung_amoc1 = chentung[:, 1]
chentung_amoc2 = chentung[:, 2]
chentung_amoc1[chentung_amoc1 == -100] = np.nan

dth = 300
sal = np.load('data/EN421_NA_salinity_am_d%d.npy'%dth)
sal_global = np.load('data/EN421_GLOBAL_salinity_d%d.npy'%dth)
print(sal_global.shape)
sal_time = np.arange(1900, 2020)
print(sal_time.shape)
sal_lat = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['lat'][:]
sal_lon = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['lon'][:]

sal_lo = len(sal_lon)
sal_la = len(sal_lat)
sal_n = sal_la * sal_lo


plt.figure()
plt.plot(rapid_time, rapid_moc)
# plt.plot(fm_time, fm_amoc['recon'][0][0][3][0])
plt.plot(fm_years, fm_amoc_ann)
plt.plot(glosea_years, glosea_amoc_ann)
#+ plt.plot(glosea5_time, glosea5_amoc_rm)
plt.savefig('plots/fm_rapid_test.pdf', bbox_inches = 'tight')

data = 'ERSST'
if data == 'HadISST':
    dat = Dataset('data/HadISST_sst.nc')
    # print(dat.variables)
    time = dat.variables['time'][:]
    lat = dat.variables['latitude'][:]
    lon = dat.variables['longitude'][:]
    sst = dat.variables['sst'][:,:,:]
    # print(lat)
    # print(lon)
elif data == 'ERSST':
    dat = Dataset('data/sst.mnmean.nc')
    # print(dat.variables)
    time = dat.variables['time'][15*12:-4]
    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    sst = dat.variables['sst'][15*12:-4,:,:]
    sst, lon = shiftgrid(180., sst, lon, start=False)
    # print(lat)
    # print(lon)



la = lat.shape[0]
lo = lon.shape[0]
n = la * lo



area_ceaser_coords = np.loadtxt('data/area_ceaser.txt')
area_ceaser_indices = trmm_indices_for_area(area_ceaser_coords, lat, lon)

area_ceaser = np.ones(n)
area_ceaser[area_ceaser_indices] = -1
area_ceaser = area_ceaser.reshape((lat.shape[0], lon.shape[0]))

ny = int(sst.shape[0] / 12)

tidx0 = np.array([10, 11, 12, 13, 14, 15, 16], dtype = 'int')
tidx = np.zeros((ny, 7))
ssty = np.zeros((ny, la, lo))
sstay = np.zeros((ny, la, lo))
for i in range(ny):
    tidx = np.array([10, 11, 12, 13, 14, 15, 16]) + i * 12
    tidx = np.array(tidx, dtype = 'int')
    ssty[i] = np.nanmean(sst[tidx], axis = 0)
    sstay[i] = np.nanmean(sst[i * 12 : (i + 1) * 12], axis = 0)

sal_ny = int(sal_global.shape[0] / 12) - 1

saly = np.zeros((sal_ny, sal_la, sal_lo))
salay = np.zeros((sal_ny, sal_la, sal_lo))
for i in range(sal_ny):
    tidx = np.array([10, 11, 12, 13, 14, 15, 16]) + i * 12
    tidx = np.array(tidx, dtype = 'int')
    saly[i] = np.nanmean(sal_global[tidx], axis = 0)
    salay[i] = np.nanmean(sal_global[i * 12 : (i + 1) * 12], axis = 0)




time = np.arange(1870, 1870 + ny)
time2 = np.arange(time[0], 2020)
print(time)
mssty = np.mean(ssty, axis = 0)
ssty[:, mssty <= 0] = np.nan

msstay = np.mean(sstay, axis = 0)
sstay[:, msstay <= 0] = np.nan

msal = np.mean(sal_global, axis = 0)
sal_global[:, msal <= 0] = np.nan


sst_trends = np.zeros((lat.shape[0], lon.shape[0]))
for i in range(lat.shape[0]):
    for j in range(lon.shape[0]):
        if mssty[i, j] > 0:
            sst_trends[i, j] = np.polyfit(np.arange(ny), ssty[:, i, j], 1)[0] * 100
        else:
            sst_trends[i, j] = np.nan

sal_trends = np.zeros((sal_lat.shape[0], sal_lon.shape[0]))
for i in range(sal_lat.shape[0]):
    for j in range(sal_lon.shape[0]):
        if msal[i, j] > 0:
            sal_trends[i, j] = np.polyfit(np.arange(sal_ny), saly[:, i, j], 1)[0] * 100
        else:
            sal_trends[i, j] = np.nan





amosidx =  coordinate_indices_from_ra(lat, lon, 80, 0, 20, -80)
gloidx =  coordinate_indices_from_ra(lat, lon, 60, -60, 179.5, -179.5)

nhidx =  coordinate_indices_from_ra(lat, lon, 80, 0, 179.5, -179.5)

nsst_trends = sst_trends / np.nanmean(sst_trends.flatten()[gloidx])


# spidx =  coordinate_indices_from_ra(lat, lon, 66, 42, -10, -80)
# bzi = np.where(nsst_trends.flatten() < 1)[0]
# sgi = np.intersect1d(spidx, bzi)

sgi = area_ceaser_indices

area = np.ones(n)
area[sgi] = -1
area = area.reshape((lat.shape[0], lon.shape[0]))

amo_area = np.ones(n)
amo_area[amosidx] = -1
amo_area = amo_area.reshape((lat.shape[0], lon.shape[0]))
print(lat)

weights = np.cos(np.radians(lat))
weights = weights / np.sum(weights)
print(weights)
weights = np.tile(weights, (lon.shape[0], 1)).T
sstay = sstay * weights
ssty = ssty * weights


# gl_mean_ay = np.nanmean(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, gloidx], axis = 1)
#
# gl_mean = np.nanmean(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, gloidx], axis = 1)
# nh_mean = np.nanmean(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, nhidx], axis = 1)
#
# amoc1 = np.nanmean(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, sgi], axis = 1)

gl_mean_ay = np.nansum(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, gloidx], axis = 1) / np.sum(weights.flatten()[gloidx])

gl_mean = np.nansum(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, gloidx], axis = 1) / np.sum(weights.flatten()[gloidx])
nh_mean = np.nansum(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, nhidx], axis = 1) / np.sum(weights.flatten()[nhidx])

amoc1 = np.nansum(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, sgi], axis = 1) / np.sum(weights.flatten()[sgi])


amoc2 = amoc1 - gl_mean
amoc2 = (amoc2 - np.mean(amoc2))
np.savetxt('data/amoc_idx_niklas.txt', amoc2)


sm_w = 50
rmw = 70
bound = rmw // 2

sst_var_spatial = np.zeros_like(ssty)
sst_ar1_spatial = np.zeros_like(ssty)
sst_lambda_spatial = np.zeros_like(ssty)

var_sst_trend = np.zeros((la, lo))
ar1_sst_trend = np.zeros((la, lo))
lambda_sst_trend = np.zeros((la, lo))

# count = 0
# for i in range(la):
#     for j in range(lo):
#         if np.sum(np.isnan(ssty[:, i, j])) == 0:
#             ts_temp = ssty[:, i, j] - runmean(ssty[:, i, j], sm_w)
#
#             sst_var_spatial[:, i, j] = runstd(ts_temp, rmw)**2
#             p1, p0 = np.polyfit(time[bound : - bound], sst_var_spatial[:, i, j][bound : - bound], 1)
#             var_sst_trend[i, j] = p1
#
#             sst_ar1_spatial[:, i, j] = runac(ts_temp, rmw)
#             p1, p0 = np.polyfit(time[bound : - bound], sst_ar1_spatial[:, i, j][bound : - bound], 1)
#             ar1_sst_trend[i, j] = p1
#
#             # sst_lambda_spatial[:, i, j] = run_fit_a(ts_temp, rmw)
#             # p1, p0 = np.polyfit(time[bound : - bound], sst_lambda_spatial[:, i, j][bound : - bound], 1)
#             # lambda_sst_trend[i, j] = p1
#             count += 1
#             print(count)
#
# np.save('data/am_sst_global_runstd.npy', sst_var_spatial)
# np.save('data/am_sst_global_runac.npy', sst_ar1_spatial)
# # np.save('data/am_sst_global_runlambda.npy', sst_lambda_spatial)
# np.save('data/trend_am_sst_global_runstd.npy', var_sst_trend)
# np.save('data/trend_am_sst_global_runac.npy', ar1_sst_trend)
# # np.save('data/trend_am_sst_global_runlambda.npy', lambda_sst_trend)
#
#
#
# sst_var_spatial = np.load('data/am_sst_global_runstd.npy')
# sst_ar1_spatial = np.load('data/am_sst_global_runac.npy')
# # sst_lambda_spatial = np.load('data/am_sst_global_runlambda.npy')
#
# m_sst_var_spatial = np.mean(sst_var_spatial, axis = 0)
# m_sst_ar1_spatial = np.mean(sst_ar1_spatial, axis = 0)
# # m_sst_lambda_spatial = np.mean(sst_lambda_spatial, axis = 0)
#
# var_sst_trend = np.load('data/trend_am_sst_global_runstd.npy')
# ar1_sst_trend = np.load('data/trend_am_sst_global_runac.npy')
# # lambda_sst_trend = np.load('data/trend_am_sst_global_runlambda.npy')
#
#
#
# data_crs = ccrs.PlateCarree()
#
# fig = plt.figure(figsize = (12,6 ))
# m_sst_var_spatial, lonc = add_cyclic_point(m_sst_var_spatial, coord=lon)
# m_sst_ar1_spatial, lonc = add_cyclic_point(m_sst_ar1_spatial, coord=lon)
# m_sst_lambda_spatial, lonc = add_cyclic_point(m_sst_lambda_spatial, coord=lon)
# lon2d, lat2d = np.meshgrid(lon, lat)
# ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
# ax.coastlines(resolution = '50m')
# ax.contourf(lonc, lat, m_am_sal_global_runstd, transform = data_crs)
# ax.gridlines(linestyle=":", draw_labels=True)
#
# ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson())
# ax.coastlines(resolution = '50m')
# ax.contourf(lonc, lat, m_am_sal_global_runac, transform = data_crs)
# ax.gridlines(linestyle=":", draw_labels=True)
#
# # ax = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson())
# # ax.coastlines(resolution = '50m')
# # ax.contourf(lonc, lat, data, transform = data_crs)
# # ax.gridlines(linestyle=":", draw_labels=True)
#
# fig.savefig('plots/SST_meanEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')
#
# fig = plt.figure(figsize = (12,6 ))
# var_sst_trend, lonc = add_cyclic_point(var_sst_trend, coord=lon)
# ar1_sst_trend, lonc = add_cyclic_point(ar1_sst_trend, coord=lon)
# lambda_sst_trend, lonc = add_cyclic_point(lambda_sst_trend, coord=lon)
# lon2d, lat2d = np.meshgrid(lon, lat)
# ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
# ax.coastlines(resolution = '50m')
# ax.contourf(lonc, lat, var_sst_trend, transform = data_crs)
# ax.gridlines(linestyle=":", draw_labels=True)
#
# ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson())
# ax.coastlines(resolution = '50m')
# ax.contourf(lonc, lat, ar1_sst_trend, transform = data_crs)
# ax.gridlines(linestyle=":", draw_labels=True)
#
# # ax = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson())
# # ax.coastlines(resolution = '50m')
# # ax.contourf(lonc, lat, lambda_sst_trend, transform = data_crs)
# # ax.gridlines(linestyle=":", draw_labels=True)
#
# fig.savefig('plots/SST_trendEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')




# amo = np.nanmean(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, amosidx], axis = 1) - gl_mean_ay
amo = np.nansum(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, amosidx], axis = 1) / np.sum(weights.flatten()[amosidx])  - gl_mean_ay
amo = amo - np.mean(amo[np.where(time == 1901)[0][0] : np.where(time == 1970)[0][0]])
amo = (amo - np.mean(amo))

amoc_caesar = np.loadtxt('data/amoc_idx_caesar.txt')
amoc_caesar = (amoc_caesar - np.mean(amoc_caesar))
time_caesar = np.arange(1870, 1870 + amoc_caesar.shape[0])

X = sm.add_constant(amo)
model = sm2.OLS(amoc2, X)
results = model.fit()
print("regresssion coeff AMO AMOC", results.params)
print("correlation coeff AMO AMOC", np.corrcoef(amoc2, amo)[0,1])
amoc3 = amoc2 - results.params[1] * amo - results.params[0]
np.savetxt('data/amoc-amo_idx_niklas.txt', amoc3)


dipole1 = coordinate_indices_from_ra(lat, lon, 80, 45, 30, -70)
dipole2 = coordinate_indices_from_ra(lat, lon, 0, -45, 30, -70)

dipole1_area = np.ones(n)
dipole1_area[dipole1] = -1
dipole1_area = dipole1_area.reshape((lat.shape[0], lon.shape[0]))

dipole2_area = np.ones(n)
dipole2_area[dipole2] = -1
dipole2_area = dipole2_area.reshape((lat.shape[0], lon.shape[0]))

# amoc4 = np.nanmean(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, dipole1], axis = 1) - np.nanmean(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, dipole2], axis = 1)
amoc4 = np.nansum(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, dipole1], axis = 1)  / np.sum(weights.flatten()[dipole1]) - np.nansum(sstay.reshape(sstay.shape[0], sstay.shape[1] * sstay.shape[2])[:, dipole2], axis = 1)  / np.sum(weights.flatten()[dipole2])
amoc4 = amoc4 - amoc4.mean()
np.savetxt('data/amoc-amo_idx_dipole.txt', amoc4)
# amoc5 = np.nanmean(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, sgi], axis = 1) - nh_mean
amoc5 = np.nansum(ssty.reshape(ssty.shape[0], ssty.shape[1] * ssty.shape[2])[:, sgi], axis = 1) / np.sum(weights.flatten()[sgi]) - nh_mean
amoc5 = amoc5 - amoc5.mean()
np.savetxt('data/amoc-amo_idx_rahmstorf.txt', amoc5)

#
#
#
# fig = plt.figure(figsize = (10,6))
# ax = fig.add_subplot(111)
# ax.plot(time_caesar, amoc_caesar, 'k-', alpha = .8, label = 'Caesar et al. 2018')
# p0, p1 = np.polyfit(time_caesar, amoc_caesar, 1)
# ax.plot(time_caesar, p0 * time_caesar + p1, 'k--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
# ax.plot(time, amoc2, 'b-', alpha = .8, label = 'SG - gl. mean')
# p0, p1 = np.polyfit(time, amoc2, 1)
# ax.plot(time, p0 * time + p1, 'b--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
# ax.plot(time, amoc3, 'r-', alpha = .8, label = 'SG - gl mean regr AMO')
# p0, p1 = np.polyfit(time, amoc3, 1)
# ax.plot(time, p0 * time + p1, 'r--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
#
# ax.plot(time, amoc4, 'g-', alpha = .8, label = 'SST Dipole (Roberts et al., 2013)')
# p0, p1 = np.polyfit(time, amoc4, 1)
# ax.plot(time, p0 * time + p1, 'g--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
#
# ax.plot(time, amoc5, 'm-', alpha = .8, label = 'SG - NH mean (Rahmstorf et al., 2015)')
# p0, p1 = np.polyfit(time, amoc5, 1)
# ax.plot(time, p0 * time + p1, 'm--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
#
# ax.set_xlabel('time [yr]')
# ax.set_ylabel('SST-based AMOC indices [K]')
# ax.grid()
# ax.legend(loc = 3, ncol = 2)
#
# ax2 = ax.twinx()
# ax2.plot(sal_time, sal, 'c-', alpha = .8, label = 'NA Salinity (Chen & Tung, 2018)')
# p0, p1 = np.polyfit(sal_time, sal, 1)
# ax2.plot(time, p0 * time + p1, 'c--', alpha = .8, label = 'slope = %.2f K/100yr'%(p0 * 100))
# # ax2.plot(chentung_time, chentung_amoc2, 'y-', alpha = .8, label = 'AMOC index (Salinity, CT2018)')
# ax2.set_ylabel('Salinity-based AMOC index')
# ax2.legend(loc = 1)
#
# plt.savefig('plots/amoc_idx_minus_vs_mean_%s_3.pdf'%data, bbox_index = 'tight')
#
#
# area = np.random.randn(180, 360)
data_crs = ccrs.PlateCarree()

np.save('data/sst_trends', sst_trends)
np.save('data/nsst_trends', nsst_trends)
np.save('data/sal_trends', sal_trends)





# sst_trends, lonc = add_cyclic_point(sst_trends, coord=lon)
#
# nsst_trends, lonc = add_cyclic_point(nsst_trends, coord=lon)
# lon2d, lat2d = np.meshgrid(lon, lat)
#
# area, lonc = add_cyclic_point(area, coord=lon)
# area_ceaser, lonc = add_cyclic_point(area_ceaser, coord=lon)
# dipole1_area, lonc = add_cyclic_point(dipole1_area, coord=lon)
# dipole2_area, lonc = add_cyclic_point(dipole2_area, coord=lon)
#
# sal_trends, sal_lonc = add_cyclic_point(sal_trends, coord=sal_lon)
#
#
# area_nn = np.ones((sal_la,sal_lo))
# area_nn[np.ix_(np.logical_and(sal_lat > 44, sal_lat < 66), np.logical_or(sal_lon > 289, sal_lon < 30))] = -1
# area_nn, sal_lonc = add_cyclic_point(area_nn, coord=sal_lon)
#
# area_n = np.ones((sal_la, sal_lo))
# area_n[np.ix_(np.logical_and(sal_lat > 10, sal_lat < 40), np.logical_or(sal_lon > 289, sal_lon < 30))] = -1
# area_n, sal_lonc = add_cyclic_point(area_n, coord=sal_lon)
#
# area_s = np.ones((sal_la, sal_lo))
# area_s[np.ix_(np.logical_and(sal_lat > -34, sal_lat < -10), np.logical_or(sal_lon > 289, sal_lon < 30))] = -1
# area_s, sal_lonc = add_cyclic_point(area_s, coord=sal_lon)
#
#
# fig = plt.figure(figsize = (10,10))
# # ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
# ax = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=340))
# # ax.coastlines(resolution = '50m')
# cf = ax.contourf(lonc, lat, sst_trends, levels = np.linspace(-.8,.8,20), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = [-.8,-.4,0,.4,.8])
# cbar.set_label(r'SST trend [K / 100yr]')
# cs = ax.contour(lonc, lat, area, [0], linestyles = 'solid', linewidths = 1., colors = 'r', transform = data_crs)
# cs = ax.contour(lonc, lat, area_ceaser, [0], linestyles = 'solid', linewidths = 2., colors = 'b', transform = data_crs)
# cs = ax.contour(lonc, lat, dipole1_area, [0], linestyles = 'solid', linewidths = 2., colors = 'c', transform = data_crs)
# cs = ax.contour(lonc, lat, dipole2_area, [0], linestyles = 'solid', linewidths = 2., colors = 'm', transform = data_crs)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
# ax.add_feature(cartopy.feature.LAND, color='k')
# # ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson())
# ax = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=340))
# # ax.coastlines(resolution = '10m')
# # cf = ax.contourf(lonc, lat, nsst_trends, levels = np.linspace(-1,3, 101), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# cf = ax.contourf(sal_lonc, sal_lat, sal_trends, levels = np.linspace(-.15,.15, 20), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = [-.15,-.075,0,.075,.15])
# # cbar.set_label(r'normalized SST trend')
# cbar.set_label(r'Salinity trends [psu / 100yr]')
# cs = ax.contour(sal_lonc, sal_lat, area_nn, [0], linestyles = 'solid', linewidths = 2., colors = 'k', transform = data_crs)
# cs = ax.contour(sal_lonc, sal_lat, area_n, [0], linestyles = 'solid', linewidths = 2., colors = 'orange', transform = data_crs)
# cs = ax.contour(sal_lonc, sal_lat, area_s, [0], linestyles = 'solid', linewidths = 2., colors = 'r', transform = data_crs)
#
# ax.gridlines(linestyle=":", draw_labels=False)
# ax.add_feature(cartopy.feature.LAND, color='k')
# fig.savefig('plots/SST_SAL_global_trends_%s.pdf'%(data), bbox_inches = 'tight')



# fig = plt.figure(figsize = (12,6 ))
# ax = fig.add_subplot(211)
# ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# basemap(sst_trends, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., area1 = area, line_color1 = 'r', area2 = area_ceaser, line_color2 = 'b', area3 = dipole1_area, line_color3 = 'c', area4 = dipole2_area, line_color4 = 'm', contours = np.linspace(-1, 1, 11), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'SST trends [K / 100yr]')
# ax = fig.add_subplot(212)
# ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# basemap(nsst_trends, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., area1 = area, line_color1 = 'r', area2 = area_ceaser, line_color2 = 'b', area3 = dipole1_area, line_color3 = 'c', area4 = dipole2_area, line_color4 = 'm', contours = np.linspace(-1,  3, 9), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'SST  trends normalized to global mean')
# fig.savefig('plots/SSTs_globally_%s.pdf'%data, bbox_inches = 'tight')
#
# ws = 10
#
# fig = plt.figure(figsize = (6,8))
#
# ax = fig.add_subplot(311)
# ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc1, 'b-', alpha = .6)
# ax.plot(time, gaussian_filter1d(amoc1, ws), 'b-', lw = 2., label = "SG SSTs")
# plt.legend(loc = 2)
# ax.set_xlim((1870, 2020))
# ax.set_ylabel('SST [°C]')
#
# ax2 = ax.twinx()
# ax2.plot(time, gl_mean, 'r-', alpha = .8)
# ax2.plot(time, gaussian_filter1d(gl_mean, ws), 'r-', lw = 2., label = "Global mean SSTs")
# plt.legend(loc = 1)
# ax2.set_xlim((1870, 2020))
# ax2.set_ylabel('SST [°C]')
#
#
# ax = fig.add_subplot(312)
# ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc2, 'b-', alpha = .6)
# ax.plot(time, gaussian_filter1d(amoc2, ws), 'b-', lw = 2., label = "AMOC index (SSTs)")
# plt.legend(loc = 2)
# ax.set_xlim((1870, 2020))
# ax.set_ylabel('SST anomaly [K]')
#
# ax2 = ax.twinx()
# ax2.plot(time, amo, 'r-', alpha = .8)
# ax2.plot(time, gaussian_filter1d(amo, ws), 'r-', lw = 2., label = "AMO index (SSTs)")
# plt.legend(loc = 1)
# ax2.set_xlim((1870, 2020))
# ax2.set_ylabel('SST anomaly [K]')
#
#
# ax = fig.add_subplot(313)
# ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc2, 'b-', alpha = .6)
# ax.plot(time, gaussian_filter1d(amoc2, ws), 'b-', lw = 2., label = "AMOC index (SSTs)")
# # plt.legend(loc = 2)
# ax.set_xlim((1870, 2020))
# ax .set_ylabel('SST anomaly [K]')
# ax.set_xlabel('Time [yr AD]')
#
# ax.plot(time, amoc3, 'r-', alpha = .8)
# ax.plot(time, gaussian_filter1d(amoc3, ws), 'r-', lw = 2., label = "AMOC index (SSTs, modified)")
# plt.legend(loc = 3)
# ax.set_xlim((1870, 2020))
# ax.set_ylabel('SST anomaly [K]')
#
# ax2 = ax.twinx()
# # ax2.plot(chentung_time, chentung_amoc1, 'k-', lw = 1, label = 'CT1')
# # ax2.plot(chentung_time, chentung_amoc2, 'k-', lw = 1, label = 'AMOC index (Salinity, CT2018)')
# p0, p1 = np.polyfit(sal_time, sal, 1)
# sal = sal - p0 * sal_time - p1
# ax2.plot(sal_time, sal, 'k-', lw = 1, label = 'AMOC index (Salinity, EN4.2.1)')
# ax2.set_ylim(-2.5, 3.5)
# ax2.set_ylabel('normalized salinity index')
# ax2.legend(loc = 1)
# # ax2 = ax.twinx()
# # ax2.plot(time, amoc3, 'r-', alpha = .8)
# # ax2.plot(time, gaussian_filter1d(amoc3, ws), 'r-', lw = 2., label = "AMOC index (modified)")
# # plt.legend(loc = 1)
# # ax2.set_xlim((1870, 2020))
# # ax2.set_ylabel('SST anomaly [K]')
#
# fig.savefig('plots/SG_SSTs_AMOC_index_%s.pdf'%data, bbox_inches = 'tight')
#
#
ws = 70
bound = ws // 2
tt_samples = 100000

fig = plt.figure(figsize = (6,8))
ax = fig.add_subplot(311)
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(time, amoc2, 'k-', label = r"AMOC index SST$_{SG-GM}$")

if data == 'HadISST':
    # popt31, cov = curve_fit(funcfit3, time[time > 1880], amoc2[time > 1880], p0 = [-8.33097773e-01,  1.05507897e-02,  2018], maxfev = 1000000000, jac = funcfit3_jac)
    popt31, cov = curve_fit(funcfit3, time[time > 1880], amoc2[time > 1880], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
elif data == 'ERSST':
    popt31, cov = curve_fit(funcfit3, time[time > 1880], amoc2[time > 1880], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)

# ax.plot(time,  funcfit3(time, *popt31), 'r-', label = '3rd order FP')
ax.plot(time2,  funcfit3(time2, *popt31), 'r-', label = 'Fixed Point')
plt.legend(loc = 3)
# ax.set_xlim((1870, 2020))
ax.set_xlim((1870, 2020))
plt.grid()

ax.set_ylabel('SST anomaly [K]')

# ax2 = ax.twinx()
# ax2.plot(rapid_time[1:], rapid_moc[:-1], 'b-', alpha = .5, label = 'RAPID')
# ax2.plot(fm_years[1:], fm_amoc_ann[:-1], 'b--', alpha = .5, label = 'FW')
# ax2.plot(glosea_years[1:], glosea_amoc_ann[:-1], 'b:', alpha = .5, label = 'Glosea5')
# ax2.legend(loc = 1)
# ax2.set_ylim(15, 23)
# ax2.set_ylabel('AMOC strength [Sv]', color = 'b')

print('cor amoc2 rapid = ', np.corrcoef(rapid_moc, amoc2[-rapid_moc.shape[0]:])[0,1])


ax = fig.add_subplot(312)
ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], run_fit_a(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], color = 'm', label = r'$\lambda$')
# p0, p1 = np.polyfit(time[bound : -bound], run_fit_a(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], 10000, p0)
# ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'm', ls = '--', label = 'Linear fit (p = %.4f)'%pv)

ax.plot(time[bound : -bound], run_fit_a_ar1(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], color = 'r', label = r'$\lambda$ SST$_{SG-GM}$')

p0, p1 = np.polyfit(time[bound : -bound], run_fit_a_ar1(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], 1)
pv = kendall_tau_test(run_fit_a_ar1(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], tt_samples, p0)
# ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
if pv >= .001:
    ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')


plt.legend(loc = 2)

ax.set_ylabel(r'Restoring rate $\lambda$ (ws = %d yr)'%ws, color = 'r')
# ax.set_xlim((1870, 2020))
ax.set_xlim((1870, 2020))
ax.set_ylim((-1, 0))
# ax2.set_ylim((., -.2))
ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')

ax2 = ax.twinx()
# fit = funcfit3(time, *popt31)
fit = funcfit3(time2, *popt31)
dfitdt = fit[1:] - fit[:-1]
# ax2.plot(time[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
ax2.plot(time2[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
ax2.set_ylabel(r'Inverse sensitivity $\left(\frac{dx^{\ast}}{dt}\right)^{-1}$', color = 'b')
# if data == 'HadISST':
#     ax2.set_ylim(-195, -180)
if data == 'HadISST':
    ax2.set_ylim(-246, -244)
elif data == 'ERSST':
    ax2.set_ylim(-260, -120)

plt.legend(loc = 4)



ax = fig.add_subplot(313)
ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(time[bound : -bound], runstd(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound]**2, color = 'r', label = 'Variance SST$_{SG-GM}$')

p0, p1 = np.polyfit(time[bound : -bound], runstd(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound]**2, 1)
pv = kendall_tau_test(runstd(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound]**2, tt_samples, p0)
# ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'p = %.5f'%pv)
if pv >= .001:
    ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')


ax.set_xlabel('Time [yr AD]')
ax.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
plt.legend(loc = 2)
# ax.set_xlim((1870, 2020))
ax.set_xlim((1870, 2020))
ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')

ax2 = ax.twinx()
ax2.plot(time[bound : -bound], runac(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], color = 'b', label = 'AC1 SST$_{SG-GM}$')
p0, p1 = np.polyfit(time[bound : -bound], runac(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], 1)
pv = kendall_tau_test(runac(amoc2 - funcfit3(time, *popt31), ws)[bound : -bound], tt_samples, p0)
# ax2.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'b', ls = '--', label = 'p = %.5f'%pv)
if pv >= .001:
    ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')

plt.legend(loc = 4)
ax2.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')

fig.savefig('plots/AMOC_SG_index_model_fits_vs_time_ws%d_all_%s_long_t.pdf'%(ws, data), bbox_inches = 'tight')

#
#
# fig = plt.figure(figsize = (6,8))
# ax = fig.add_subplot(311)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc3, 'k-', label = r"AMOC index SST$_{SG-GM-AMO}$")
# a_range = np.arange(amoc3.min() - 10, amoc3.max() + 10, .01)
# if data == 'HadISST':
#     popt32, cov = curve_fit(funcfit3, time[time > 1880], amoc3[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# elif data == 'ERSST':
#     popt32, cov = curve_fit(funcfit3, time[time > 1880], amoc3[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
#
# # ax.plot(time,  funcfit3(time, *popt32), 'r-', label = '3rd order FP')
# ax.plot(time2,  funcfit3(time2, *popt32), 'r-', label = 'Fixed Point')
# plt.legend(loc = 3)
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# plt.grid()
#
# ax.set_ylabel('SST anomaly [K]')
#
# # ax2 = ax.twinx()
# # ax2.plot(rapid_time[1:], rapid_moc[:-1], 'b-', alpha = .5, label = 'RAPID')
# # ax2.plot(fm_years[1:], fm_amoc_ann[:-1], 'b--', alpha = .5, label = 'FW')
# # ax2.plot(glosea_years[1:], glosea_amoc_ann[:-1], 'b:', alpha = .5, label = 'Glosea5')
# #
# # ax2.legend(loc = 1)
# # ax2.set_ylim(15, 24)
# # ax2.set_ylabel('AMOC strength [Sv]', color = 'b')
#
# ax = fig.add_subplot(312)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], run_fit_a_ar1(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], color = 'r', label = r'$\lambda$ SST$_{SG-GM-AMO}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], run_fit_a_ar1(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# plt.legend(loc = 2)
#
# ax.set_ylabel(r'Restoring rate $\lambda$ (ws = %d yr)'%ws, color = 'r')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_ylim((-1, 0))
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
#
# ax2 = ax.twinx()
# # fit = funcfit3(time, *popt32)
# fit = funcfit3(time2, *popt32)
# dfitdt = fit[1:] - fit[:-1]
# # ax2.plot(time[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.plot(time2[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.set_ylabel(r'Inverse sensitivity $\left(\frac{dx^{\ast}}{dt}\right)^{-1}$', color = 'b')
# plt.legend(loc = 4)
# if data == 'HadISST':
#     ax2.set_ylim(-312.24, -310.)
#
#
#
# ax = fig.add_subplot(313)
# ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], runstd(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound]**2, color = 'r', label = 'Variance SST$_{SG-GM-AMO}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], runstd(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_xlabel('Time [yr AD]')
# ax.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
# plt.legend(loc = 2)
#
# ax2 = ax.twinx()
# ax2.plot(time[bound : -bound], runac(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], color = 'b', label = 'AC1 SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time[bound : -bound], runac(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc3 - funcfit3(time, *popt32), ws)[bound : -bound], tt_samples, p0)
# # ax2.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'b', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
#
# ax2.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')
# plt.legend(loc = 4)
#
# fig.savefig('plots/AMOC-AMO_SG_index_model_fits_vs_time_ws%d_all_%s_long.pdf'%(ws, data), bbox_inches = 'tight')
#
#
# fig = plt.figure(figsize = (6,8))
# ax = fig.add_subplot(311)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc4, 'k-', label = r"AMOC index SST$_{DIPOLE}$")
# a_range = np.arange(amoc4.min() - 10, amoc4.max() + 10, .01)
# if data == 'HadISST':
#     popt42, cov = curve_fit(funcfit3, time[time > 1880], amoc4[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# elif data == 'ERSST':
#     popt42, cov = curve_fit(funcfit3, time[time > 1880], amoc4[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
#
# # ax.plot(time,  funcfit3(time, *popt32), 'r-', label = '3rd order FP')
# ax.plot(time2,  funcfit3(time2, *popt42), 'r-', label = 'Fixed Point')
# plt.legend(loc = 3)
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# plt.grid()
#
# ax.set_ylabel('SST anomaly [K]')
#
# # ax2 = ax.twinx()
# # ax2.plot(rapid_time[1:], rapid_moc[:-1], 'b-', alpha = .5, label = 'RAPID')
# # ax2.plot(fm_years[1:], fm_amoc_ann[:-1], 'b--', alpha = .5, label = 'FW')
# # ax2.plot(glosea_years[1:], glosea_amoc_ann[:-1], 'b:', alpha = .5, label = 'Glosea5')
# #
# # ax2.legend(loc = 1)
# # ax2.set_ylim(15, 24)
# # ax2.set_ylabel('AMOC strength [Sv]', color = 'b')
#
# ax = fig.add_subplot(312)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], run_fit_a_ar1(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], color = 'r', label = r'$\lambda$ SST$_{DIPOLE}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], run_fit_a_ar1(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# plt.legend(loc = 2)
#
# ax.set_ylabel(r'Restoring rate $\lambda$ (ws = %d yr)'%ws, color = 'r')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_ylim((-1, 0))
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
#
# ax2 = ax.twinx()
# # fit = funcfit3(time, *popt32)
# fit = funcfit3(time2, *popt42)
# dfitdt = fit[1:] - fit[:-1]
# # ax2.plot(time[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.plot(time2[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.set_ylabel(r'Inverse sensitivity $\left(\frac{dx^{\ast}}{dt}\right)^{-1}$', color = 'b')
# plt.legend(loc = 4)
# if data == 'HadISST':
#     ax2.set_ylim(-390.6, -389.9)
#
# ax = fig.add_subplot(313)
# ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], runstd(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound]**2, color = 'r', label = 'Variance SST$_{DIPOLE}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], runstd(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_xlabel('Time [yr AD]')
# ax.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
# plt.legend(loc = 2)
#
# ax2 = ax.twinx()
# ax2.plot(time[bound : -bound], runac(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], color = 'b', label = 'AC1 SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time[bound : -bound], runac(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc4 - funcfit3(time, *popt42), ws)[bound : -bound], tt_samples, p0)
# # ax2.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'b', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
#
# ax2.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')
# plt.legend(loc = 4)
#
# fig.savefig('plots/AMOC_DIPOLE_index_model_fits_vs_time_ws%d_all_%s_long.pdf'%(ws, data), bbox_inches = 'tight')
#
# fig = plt.figure(figsize = (6,8))
# ax = fig.add_subplot(311)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time, amoc5, 'k-', label = "AMOC index SST$_{SG-NH}$")
# a_range = np.arange(amoc5.min() - 10, amoc5.max() + 10, .01)
# if data == 'HadISST':
#     popt52, cov = curve_fit(funcfit3, time[time > 1880], amoc5[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# elif data == 'ERSST':
#     popt52, cov = curve_fit(funcfit3, time[time > 1880], amoc5[time > 1880], p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
#
# # ax.plot(time,  funcfit3(time, *popt32), 'r-', label = '3rd order FP')
# ax.plot(time2,  funcfit3(time2, *popt52), 'r-', label = 'Fixed Point')
# plt.legend(loc = 3)
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# plt.grid()
#
# ax.set_ylabel('SST anomaly [K]')
#
# # ax2 = ax.twinx()
# # ax2.plot(rapid_time[1:], rapid_moc[:-1], 'b-', alpha = .5, label = 'RAPID')
# # ax2.plot(fm_years[1:], fm_amoc_ann[:-1], 'b--', alpha = .5, label = 'FW')
# # ax2.plot(glosea_years[1:], glosea_amoc_ann[:-1], 'b:', alpha = .5, label = 'Glosea5')
# #
# # ax2.legend(loc = 1)
# # ax2.set_ylim(15, 24)
# # ax2.set_ylabel('AMOC strength [Sv]', color = 'b')
#
# ax = fig.add_subplot(312)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], run_fit_a_ar1(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], color = 'r', label = r'$\lambda$ SST$_{SG-NH}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], run_fit_a_ar1(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# plt.legend(loc = 2)
#
# ax.set_ylabel(r'Restoring rate $\lambda$ (ws = %d yr)'%ws, color = 'r')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_ylim((-1, 0))
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
#
# ax2 = ax.twinx()
# # fit = funcfit3(time, *popt32)
# fit = funcfit3(time2, *popt52)
# dfitdt = fit[1:] - fit[:-1]
# # ax2.plot(time[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.plot(time2[:-1], dfitdt**-1, color = 'b', ls = '-', label = r'Sensitivity $\frac{dx^{\ast}}{dt}$')
# ax2.set_ylabel(r'Inverse sensitivity $\left(\frac{dx^{\ast}}{dt}\right)^{-1}$', color = 'b')
# plt.legend(loc = 4)
# if data == 'HadISST':
#     ax2.set_ylim(-312.6, -311.2)
#
# ax = fig.add_subplot(313)
# ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time[bound : -bound], runstd(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound]**2, color = 'r', label = 'Variance SST$_{SG-NH}$')
#
# p0, p1 = np.polyfit(time[bound : -bound], runstd(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'r', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')
#
#
# ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
# ax.axvspan(time[:-bound][-1], 2020, facecolor = 'none', edgecolor = 'k', hatch = '/')
# # ax.set_xlim((1870, 2020))
# ax.set_xlim((1870, 2020))
# ax.set_xlabel('Time [yr AD]')
# ax.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
# plt.legend(loc = 2)
#
# ax2 = ax.twinx()
# ax2.plot(time[bound : -bound], runac(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], color = 'b', label = 'AC1 SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time[bound : -bound], runac(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc5 - funcfit3(time, *popt52), ws)[bound : -bound], tt_samples, p0)
# # ax2.plot(time[bound : -bound], p0 * time[bound : -bound] + p1, color = 'b', ls = '--', label = 'Linear fit (p = %.5f)'%pv)
# if pv >= .001:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
# else:
#     ax2.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
#
# ax2.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')
# plt.legend(loc = 4)
#
# fig.savefig('plots/AMOC_SGNH_index_model_fits_vs_time_ws%d_all_%s_long.pdf'%(ws, data), bbox_inches = 'tight')
#
#
#


ws2 = 30

gl_mean_sort = np.sort(gl_mean)
gmt_range = np.linspace(gl_mean.min(), gl_mean.max(), 150)

# amoc2_sort = amoc2[np.argsort(gl_mean)]
# popt21, cov21 = curve_fit(funcfit2, gl_mean_sort, amoc2_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt31, cov31 = curve_fit(funcfit3, gl_mean_sort, amoc2_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc2_o2_std = np.zeros_like(gl_mean_sort)
# amoc2_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc2_o2_std[i] = np.std(amoc2_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt21))**2
#     amoc2_o3_std[i] = np.std(amoc2_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt31))**2
#
#
# fig = plt.figure(figsize = (6,6))
#
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc2, c = time, cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt31), 'r-', label = r'Fixed Point SST$_{SG-GM}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc2_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance SST$_{SG-GM}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc2_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc2_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     ax.set_ylim(0.02, .2)
# ax.set_ylabel(r'Variance SST$_{SG-GM}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt31)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     ax2.set_ylim(.4, 1.2)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 2)
# plt.legend(loc = 4)
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
#
# plt.subplots_adjust(wspace = .4, hspace = .2)
# fig.savefig('plots/AMOC_SG_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
#
# amoc3_sort = amoc3[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc3_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc3_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc3_o2_std = np.zeros_like(gl_mean_sort)
# amoc3_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc3_o2_std[i] = np.std(amoc3_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc3_o3_std[i] = np.std(amoc3_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc3, c = time, cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point SST$_{SG-GM-AMO}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc3_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = 'Variance SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc3_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc3_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(.025, .075)
# ax.set_ylabel(r'Variance SST$_{SG-GM-AMO}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     # ax2.set_ylim(.4, 1.6)
#     ax2.set_ylim(.605, .615)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC-AMO_SG_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
#
# amoc4_sort = amoc4[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc4_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc4_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc4_o2_std = np.zeros_like(gl_mean_sort)
# amoc4_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc4_o2_std[i] = np.std(amoc4_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc4_o3_std[i] = np.std(amoc4_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc4, c = time, cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point SST$_{DIPOLE}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc4_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc4_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc4_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(0., .065)
# ax.set_ylabel(r'Variance SST$_{DIPOLE}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# # if data == 'HadISST':
# #     # ax2.set_ylim(.4, 1.6)
# #     ax2.set_ylim(.605, .615)
# # elif data == 'ERSST':
# #     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_DIPOLE_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
# amoc5_sort = amoc5[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc5_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc5_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc5_o2_std = np.zeros_like(gl_mean_sort)
# amoc5_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc5_o2_std[i] = np.std(amoc5_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc5_o3_std[i] = np.std(amoc5_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc5, c = time, cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point SST$_{SG-NH}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc5_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance SST$_{SG-NH}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc5_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc5_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(.04, .14)
# ax.set_ylabel(r'Variance SST$_{SG-NH}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     # ax2.set_ylim(.4, 1.6)
#     ax2.set_ylim(.4, 1.)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_SGNH_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')



sal = np.load('data/EN421_NA_salinity_d%d.npy'%dth)
sal = (sal - sal.mean())# / sal.std()

sal_n = np.load('data/EN421_SSS_N_d%d.npy'%dth)
sal_s = np.load('data/EN421_SSS_S_d%d.npy'%dth)
sal_sg = np.load('data/EN421_SG_d%d.npy'%dth)
sal_klus = np.load('data/EN421_KLUS_d%d.npy'%dth)
print(sal_klus)
sal_n = (sal_n - sal_n.mean())
sal_s = (sal_s - sal_s.mean())
sal_sg = (sal_sg - sal_sg.mean())
sal_klus = (sal_klus - sal_klus.mean())
print(sal_klus)
years = np.arange(1900, 2020)
sal_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_am[i] = np.mean(sal[i * 12 : (i + 1) * 12])
np.save('data/EN421_NA_salinity_am_d%d.npy'%dth, sal_am)

sal_n_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_n_am[i] = np.mean(sal_n[i * 12 : (i + 1) * 12])
np.save('data/EN421_NA_salinity_n_am_d%d.npy'%dth, sal_n_am)


sal_s_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_s_am[i] = np.mean(sal_s[i * 12 : (i + 1) * 12])
np.save('data/EN421_NA_salinity_s_am_d%d.npy'%dth, sal_s_am)

sal_klus_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_klus_am[i] = np.nanmean(sal_klus[i * 12 : (i + 1) * 12])
np.save('data/EN421_NA_salinity_klus_am_d%d.npy'%dth, sal_klus_am)

print(sal_klus_am)

time_sal = np.arange(1900, 2020)


gmt_range = np.linspace(gl_mean.min(), gl_mean.max(), 150)



amoc6 = -sal_am[:-1]
amoc7 = -sal_n_am[:-1]
amoc8 = -sal_s_am[:-1]
amoc9 = -sal_klus_am[:-1]


np.savetxt('data/SST_SG-GM.txt', amoc2)
np.savetxt('data/SST_SG-GM-AMO.txt', amoc3)
np.savetxt('data/SST_DIPOLE.txt', amoc4)
np.savetxt('data/SST_SG-NH.txt', amoc5)
np.savetxt('data/S_NN1.txt', amoc6)
np.savetxt('data/S_N.txt', amoc7)
np.savetxt('data/S_S.txt', amoc8)
np.savetxt('data/S_NN2.txt', amoc9)
np.savetxt('data/GM_SST.txt', gl_mean)


# print(sal_klus_am)
# print(sal_am.shape)
# gl_mean = gl_mean[30:]
# print(gl_mean.shape)
# gl_mean_sort = np.sort(gl_mean)
# gmt_range = np.linspace(gl_mean.min(), gl_mean.max(), 150)
#
# amoc6_sort = amoc6[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc6_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc6_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc6_o2_std = np.zeros_like(gl_mean_sort)
# amoc6_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc6_o2_std[i] = np.std(amoc6_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc6_o3_std[i] = np.std(amoc6_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc6, c = time_sal[:-1], cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point S$_{NN1}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc6_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance S$_{NN1}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc6_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc6_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# # if data == 'HadISST':
# #     # ax.set_ylim(.02, .1)
# #     ax.set_ylim(.025, .075)
# ax.set_ylabel(r'Variance S$_{NN1}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# # if data == 'HadISST':
# #     # ax2.set_ylim(.4, 1.6)
# #     ax2.set_ylim(.605, .615)
# # elif data == 'ERSST':
# #     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_SAL_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
# amoc7_sort = amoc7[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc7_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc7_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc7_o2_std = np.zeros_like(gl_mean_sort)
# amoc7_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc7_o2_std[i] = np.std(amoc7_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc7_o3_std[i] = np.std(amoc7_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc7, c = time_sal[:-1], cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point S$_{N}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc7_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance S$_{N}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc7_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc7_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(-.0004, .0025)
# ax.set_ylabel(r'Variance S$_{N}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     # ax2.set_ylim(.4, 1.6)
#     ax2.set_ylim(0, .3)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_SAL_N_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
# amoc8_sort = amoc8[np.argsort(gl_mean)]
# popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc8_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc8_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# amoc8_o2_std = np.zeros_like(gl_mean_sort)
# amoc8_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     amoc8_o2_std[i] = np.std(amoc8_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc8_o3_std[i] = np.std(amoc8_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc8, c = time_sal[:-1], cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point S$_{S}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc8_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance S$_{S}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc8_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc8_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(-.0005, .0032)
# ax.set_ylabel(r'Variance S$_{S}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     # ax2.set_ylim(.4, 1.6)
#     ax2.set_ylim(0., .5)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_SAL_S_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
#
#
# print(amoc9)
# amoc9_sort = amoc9[np.argsort(gl_mean)]
# # popt22, cov22 = curve_fit(funcfit2, gl_mean_sort, amoc9_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
# popt32, cov32 = curve_fit(funcfit3, gl_mean_sort, amoc9_sort, p0 = [-1, 3, 30], maxfev = 100000000, jac = funcfit3_jac)
#
# # amoc9_o2_std = np.zeros_like(gl_mean_sort)
# amoc9_o3_std = np.zeros_like(gl_mean_sort)
# for i in range(ws2 // 2, gl_mean_sort.shape[0] - ws2 // 2):
#     # amoc9_o2_std[i] = np.std(amoc9_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit2(gl_mean_sort[i], *popt22))**2
#     amoc9_o3_std[i] = np.std(amoc9_sort[i - ws2 // 2 : i + ws2 // 2] - funcfit3(gl_mean_sort[i], *popt32))**2
#
#
# fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(211)
# ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# sc = ax.scatter(gl_mean, amoc9, c = time_sal[:-1], cmap = 'viridis')
# plt.colorbar(sc, orientation = 'horizontal', label = 'time [yr]', pad = 0.2)
# ax.plot(gmt_range,  funcfit3(gmt_range, *popt32), 'r-', label = r'Fixed Point S$_{NN2}$')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# plt.legend(loc = 1)
# plt.grid()
# ax.set_ylabel('SST anomaly [K]')
# ax.set_xlabel('GMT [°C]')
#
# ax = fig.add_subplot(212)
# ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc9_o3_std[ws2 // 2 : - ws2 // 2], color = 'r', ls = 'None', marker = 'o', alpha = .7, label = r'Variance S$_{NN2}$')
# p0, p1 = np.polyfit(gl_mean_sort[ws2 // 2 : - ws2 // 2], amoc9_o3_std[ws2 // 2 : - ws2 // 2], 1)
# pv = kendall_tau_test(amoc9_o3_std[ws2 // 2 : - ws2 // 2], 10000, p0)
# # ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit (p = %.4f)'%pv)
# ax.plot(gl_mean_sort[ws2 // 2 : - ws2 // 2], p0 * gl_mean_sort[ws2 // 2 : - ws2 // 2] + p1, 'r--', lw = 2, label = 'Linear fit')
# ax.set_xlim((gl_mean_sort.min(), gl_mean_sort.max()))
# if data == 'HadISST':
#     # ax.set_ylim(.02, .1)
#     ax.set_ylim(0., .008)
# ax.set_ylabel(r'Variance S$_{NN2}$', color = 'r')
# ax.set_xlabel('GMT [°C]')
# plt.legend(loc = 2)
# plt.grid()
#
# fp = funcfit3(gmt_range, *popt32)
# dxdT = (fp[1:] - fp[:-1]) / (gmt_range[1:] - gmt_range[:-1])
#
# ax2 = ax.twinx()
# ax2.plot(gmt_range[:-1], -dxdT, 'b-', label = r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$')
# if data == 'HadISST':
#     # ax2.set_ylim(.4, 1.6)
#     ax2.set_ylim(0., .4)
# elif data == 'ERSST':
#     ax2.set_ylim(0, 3)
#
# ax2.set_ylabel(r'Sensitivity $\left|\frac{dx^{\ast}}{dT}\right|$', color = 'b')
# plt.legend(loc = 4)
# fig.savefig('plots/AMOC_SAL_KLUS_index_model_fits_vs_T2_ws%d_%s.pdf'%(ws2, data), bbox_inches = 'tight')
