import numpy as np
import scipy.stats as st
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from EWS_functions import *

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
data_crs = ccrs.PlateCarree()

def funcfit3(x, a, b, c):
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 3)

def funcfit3_jac(x, a, b, c):
    return np.array([np.ones(x.shape[0]), -(x-c) * np.power(-b * (x - c), -2/3) / 3, b * np.power(-b * (x - c), -2/3) / 3]).T


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
   ra_indices = indices[np.where(trmm_lat == lat_min)[0][0] : np.where(trmm_lat == lat_max)[0][0] + 1 , np.where(trmm_lon == lon_min)[0][0] : np.where(trmm_lon == lon_max)[0][0] + 1 ]
   ra_indices = ra_indices.flatten()
   trmm_ra_coords = trmm_coords[ra_indices]
   d = np.zeros((len(area_coords), len(ra_indices)))
   for i in range(len(area_coords)):
      for j in range(len(ra_indices)):
         d[i, j] = np.sum(np.abs(area_coords[i] - trmm_ra_coords[j]))
   trmm_indices_area = ra_indices[np.argmin(d, axis = 1)]
   return trmm_indices_area


def trmm_indices_for_area2(area_coords, trmm_lat, trmm_lon):
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

def coordinate_indices_from_ra2(lat, lon, lat_max, lat_min, lon_max, lon_min):
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



years = np.arange(1900, 2020)

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


tlen = int(years.shape[0] * 12)

lat = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['lat'][:]
lon = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['lon'][:]
depth = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['depth'][:]




la = len(lat)
lo = len(lon)
n = la * lo

area = np.ones((la, lo))
area[np.ix_(np.logical_and(lat > 44, lat < 66), np.logical_or(lon > 289, lon < 11))] = -1

salt = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.201811.nc').variables['salinity'][0, 0, :,:]



sal = np.zeros((tlen))
sal_s = np.zeros((tlen))
sal_n = np.zeros((tlen))
sal_sg = np.zeros((tlen))
sal_klus = np.zeros((tlen))

sal_global = np.zeros((tlen, la, lo))

dth = 300

weights = np.cos(lat * 2 * np.pi / 360)


dlen = np.where(depth < dth)[0].shape[0]

area_ceaser_coords = np.loadtxt('data/area_ceaser.txt')
area_ceaser_indices_sal = trmm_indices_for_area(area_ceaser_coords, lat, lon)

# i = 0
# for y in years:
#     if y < 2020:
#         for m in range(12):
#             dat = Dataset('data/EN421/EN.4.2.1.f.analysis.g10.%s%s.nc'%(y, months[m]))
#             # print(dat.variables)
#             # sal[i] = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 44, lat < 66), np.logical_or(lon > 289, lon < 11)])
#             # sal_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 44, lat < 66), np.logical_or(lon > 289, lon < 11)], axis = 2)
#             sal_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 44, lat < 66), np.logical_or(lon > 289, lon < 30)], axis = 2)
#             sal_temp = np.average(sal_temp, weights = weights[np.logical_and(lat > 44, lat < 66)], axis = 1)
#             # sal_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 50, lat < 60), np.logical_and(lon > 310, lon < 330)], axis = 2)
#             # sal_temp = np.average(sal_temp, weights = weights[np.logical_and(lat > 50, lat < 60)], axis = 1)
#             sal[i] = sal_temp.mean()
#
#             ## sal_n_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 10, lat < 40), np.logical_or(lon > 280, lon < 11)], axis = 2)
#             # sal_n_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 10, lat < 40), np.logical_or(lon > 289, lon < 30)], axis = 2)
#             # sal_n_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 10, lat < 40), np.logical_or(lon > 280, lon < 30)], axis = 2)
#             # sal_n_temp = np.average(sal_n_temp, weights = weights[np.logical_and(lat > 10, lat < 40)], axis = 1)
#             # sal_n[i] = sal_n_temp.mean()
#
#
#             ## sal_s_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > -34, lat < -10), np.logical_or(lon > 280, lon < 11)], axis = 2)
#             # sal_s_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > -34, lat < -10), np.logical_or(lon > 289, lon < 30)], axis = 2)
#             # sal_s_temp = np.average(sal_s_temp, weights = weights[np.logical_and(lat > -34, lat < -10)], axis = 1)
#             # sal_s[i] = sal_s_temp.mean()
#
#
#             sal_klus_temp = np.nanmean(dat.variables['salinity'][0, depth < dth, np.logical_and(lat > 55, lat < 62), np.logical_and(lon > 298, lon < 334)], axis = 2)
#             sal_klus_temp = np.average(sal_klus_temp, weights = weights[np.logical_and(lat > 55, lat < 62)], axis = 1)
#             sal_klus[i] = sal_klus_temp.mean()
#
#
#             # sal_sg[i] = np.nanmean(dat.variables['salinity'][0, depth < dth, :, :].reshape(dlen, la * lo)[:, area_ceaser_indices_sal])
#
#
#
# #             sal_global[i] = np.nanmean(dat.variables['salinity'][0, depth < dth], axis = 0)
#
#              i += 1

# np.save('data/EN421_NA_salinity_d%d.npy'%dth, sal)
# np.save('data/EN421_SSS_N_d%d.npy'%dth, sal_n)
# np.save('data/EN421_SSS_S_d%d.npy'%dth, sal_s)
# np.save('data/EN421_SG_d%d.npy'%dth, sal_sg)
# np.save('data/EN421_KLUS_d%d.npy'%dth, sal_klus)

#
# np.save('data/EN421_GLOBAL_salinity_d%d.npy'%dth, sal_global)

sal_global = np.load('data/EN421_GLOBAL_salinity_d%d.npy'%dth)
sal_global_temp = np.reshape(sal_global, (sal_global.shape[0], sal_global.shape[1] * sal_global.shape[2]))

norm_sal_global = np.nanmean(sal_global_temp, axis = 1)

m_sal_global = np.mean(sal_global, axis = 0)



tt_samples = 100000
sm_w = 50
rmw = 70
ws = rmw
bound = rmw // 2


am_sal_global = np.zeros((years.shape[0], la, lo))
for i in range(years.shape[0]):
    am_sal_global[i] = np.mean(sal_global[i * 12 : (i + 1) * 12], axis = 0)
np.save('data/EN421_GLOBAL_salinity_am_d%d.npy'%dth, am_sal_global)

sst_data = 'HadISST'
if sst_data == 'HadISST':
    sst_dat = Dataset('data/HadISST_sst.nc')
    sst_time = sst_dat.variables['time'][:]
    sst_lat = sst_dat.variables['latitude'][:]
    sst_lon = sst_dat.variables['longitude'][:]
    sst = sst_dat.variables['sst'][:,:,:]

elif data == 'ERSST':
    sst_dat = Dataset('data/sst.mnmean.nc')
    sst_time = sst_dat.variables['time'][15*12:-4]
    sst_lat = sst_dat.variables['lat'][:]
    sst_lon = sst_dat.variables['lon'][:]
    sst_sst = sst_dat.variables['sst'][15*12:-4,:,:]
    sst, lon = shiftgrid(180., sst, sst_dat, start=False)


sst[sst == -1000] = np.nan

sst_la = sst_lat.shape[0]
sst_lo = sst_lon.shape[0]
sst_n = sst_la * sst_lo

sst_ny = int(sst.shape[0] / 12)
sst_years = np.arange(1870, 1870 + sst_ny)

ssty = np.zeros((sst_ny, sst_la, sst_lo))
sstay = np.zeros((sst_ny, sst_la, sst_lo))
for i in range(sst_ny):
    tidx = np.array([10, 11, 12, 13, 14, 15, 16]) + i * 12
    tidx = np.array(tidx, dtype = 'int')
    ssty[i] = np.nanmean(sst[tidx], axis = 0)
    sstay[i] = np.nanmean(sst[i * 12 : (i + 1) * 12], axis = 0)




# # am_sal_global_runstd = np.ones((years.shape[0], la, lo)) * np.nan
# # am_sal_global_runac = np.ones((years.shape[0], la, lo)) * np.nan
# # am_sal_global_runlambda_cor = np.ones((years.shape[0], la, lo)) * np.nan
# #
# # var_en4 = np.ones((la, lo)) * np.nan
# # ar_en4 = np.ones((la, lo)) * np.nan
# # lambda_cor_en4 = np.ones((la, lo)) * np.nan
#
# var_en4_p = np.ones((la, lo)) * np.nan
# ar_en4_p = np.ones((la, lo)) * np.nan
# # lambda_cor_en4_p = np.ones((la, lo)) * np.nan
#
#
# am_sal_global_runstd = np.load('data/am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sal_global_runac = np.load('data/am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sal_global_runlambda_cor = np.load('data/am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# var_en4 = np.load('data/trend_am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_en4 = np.load('data/trend_am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_cor_en4 = np.load('data/trend_am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# count = 0
# for i in range(la):
#     for j in range(lo):
#         if np.sum(np.isnan(am_sal_global[:, i, j])) == 0:
#             ts_temp = am_sal_global[:, i, j] - runmean(am_sal_global[:, i, j], sm_w)
#
#             # am_sal_global_runstd[:, i, j] = runstd(ts_temp, rmw)**2
#             # var_en4[i, j] = st.linregress(years[bound : - bound], am_sal_global_runstd[:, i, j][bound : - bound])[0]
#             var_en4_p[i, j] = kendall_tau_test(am_sal_global_runstd[:, i, j][bound : - bound], 1000, var_en4[i, j])
#
#             # am_sal_global_runac[:, i, j] = runac(ts_temp, rmw)
#             # ar_en4[i, j] = st.linregress(years[bound : - bound], am_sal_global_runac[:, i, j][bound : - bound])[0]
#             ar_en4_p[i, j] = kendall_tau_test(am_sal_global_runac[:, i, j][bound : - bound], 1000, ar_en4[i, j])
#
#             # am_sal_global_runlambda_cor[:, i, j] = run_fit_a_ar1(ts_temp, rmw)
#             # lambda_cor_en4[i, j] = st.linregress(years[bound : - bound], am_sal_global_runlambda_cor[:, i, j][bound : - bound])[0]
#             # lambda_cor_en4_p[i, j] = kendall_tau_test(am_sal_global_runlambda_cor[:, i, j][bound : - bound], 1000, lambda_cor_en4[i, j])
#             count += 1
#             print(count)
#
# # np.save('data/am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sal_global_runstd)
# # np.save('data/am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sal_global_runac)
# # np.save('data/am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sal_global_runlambda_cor)
# #
# # np.save('data/trend_am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), var_en4)
# # np.save('data/trend_am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), ar_en4)
# # np.save('data/trend_am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), lambda_cor_en4)
#
# np.save('data/trend_p_am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), var_en4_p)
# np.save('data/trend_p_am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), ar_en4_p)
# np.save('data/trend_p_am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), lambda_cor_en4_p)
#
#
#
#
# am_sst_global_runstd = np.ones((sst_ny, sst_la, sst_lo)) * np.nan
# am_sst_global_runac = np.ones((sst_ny, sst_la, sst_lo)) * np.nan
# am_sst_global_runlambda_cor = np.ones((sst_ny, sst_la, sst_lo)) * np.nan
#
# var_sst = np.ones((sst_la, sst_lo)) * np.nan
# ar_sst = np.ones((sst_la, sst_lo)) * np.nan
# lambda_cor_sst = np.ones((sst_la, sst_lo)) * np.nan
#
#
# ar_sst_p = np.ones((sst_la, sst_lo)) * np.nan
# var_sst_p = np.ones((sst_la, sst_lo)) * np.nan
# lambda_cor_sst_p = np.ones((sst_la, sst_lo)) * np.nan
#
#
# am_sst_global_runstd = np.load('data/am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sst_global_runac = np.load('data/am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sst_global_runlambda_cor = np.load('data/am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# var_sst = np.load('data/trend_am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_sst = np.load('data/trend_am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_cor_sst = np.load('data/trend_am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# count = 0
# for i in range(la):
#     for j in range(lo):
#         if np.sum(np.isnan(ssty[:, i, j])) == 0 and np.where(ssty[:, i, j] <= 0)[0].shape[0] == 0:
#             ts_temp = ssty[:, i, j] - runmean(ssty[:, i, j], sm_w)
#             if np.where(ts_temp == 0)[0].shape[0] == 0:
#                 # am_sst_global_runstd[:, i, j] = runstd(ts_temp, rmw)**2
#                 # var_sst[i, j] = st.linregress(sst_years[bound : - bound], am_sst_global_runstd[:, i, j][bound : - bound])[0]
#                 var_sst_p[i, j] = kendall_tau_test(am_sst_global_runstd[:, i, j][bound : - bound], 1000, var_sst[i, j])
#
#                 # am_sst_global_runac[:, i, j] = runac(ts_temp, rmw)
#                 # ar_sst[i, j] = st.linregress(sst_years[bound : - bound], am_sst_global_runac[:, i, j][bound : - bound])[0]
#                 # ar_sst_p[i, j] = kendall_tau_test(am_sst_global_runac[:, i, j][bound : - bound], 1000, ar_sst[i, j])
#
#                 # am_sst_global_runlambda_cor[:, i, j] = run_fit_a_ar1(ts_temp, rmw)
#                 # lambda_cor_sst[i, j] = st.linregress(sst_years[bound : - bound], am_sst_global_runlambda_cor[:, i, j][bound : - bound])[0]
#                 # lambda_cor_sst_p[i, j] = kendall_tau_test(am_sst_global_runlambda_cor[:, i, j][bound : - bound], 1000, lambda_cor_sst[i, j])
#
#                 count += 1
#                 print(count)
#
# np.save('data/am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sst_global_runstd)
# np.save('data/am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sst_global_runac)
# np.save('data/am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), am_sst_global_runlambda_cor)
#
# np.save('data/trend_am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), var_sst)
# np.save('data/trend_am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), ar_sst)
# np.save('data/trend_am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), lambda_cor_sst)

# np.save('data/trend_p_am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), lambda_cor_sst_p)
# np.save('data/trend_p_am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), var_sst_p)
# np.save('data/trend_p_am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw), ar_sst_p)






# am_sal_global = np.load('data/EN421_GLOBAL_salinity_am_d%d.npy'%dth)
#
#
#
# am_sal_global_runstd = np.load('data/am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sal_global_runac = np.load('data/am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sal_global_runlambda = np.load('data/am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# m_am_sal_global_runstd = np.nanmean(am_sal_global_runstd, axis = 0)
# m_am_sal_global_runac = np.nanmean(am_sal_global_runac, axis = 0)
# m_am_sal_global_runlambda = np.nanmean(am_sal_global_runlambda, axis = 0)
#
# var_en4 = np.load('data/trend_am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_en4 = np.load('data/trend_am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_en4 = np.load('data/trend_am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# var_en4_p = np.load('data/trend_p_am_sal_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_en4_p = np.load('data/trend_p_am_sal_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_en4_p = np.load('data/trend_p_am_sal_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# m_am_sal_global_runstd, lonc = add_cyclic_point(m_am_sal_global_runstd, coord=lon)
# m_am_sal_global_runac, lonc = add_cyclic_point(m_am_sal_global_runac, coord=lon)
# m_am_sal_global_runlambda, lonc = add_cyclic_point(m_am_sal_global_runlambda, coord=lon)
#
# m_am_sal_global_runstd[m_am_sal_global_runstd == 0] = np.nan
# m_am_sal_global_runac[m_am_sal_global_runac == 0] = np.nan
# m_am_sal_global_runlambda[m_am_sal_global_runlambda == 0] = np.nan
#
# var_en4[var_en4 == 0] = np.nan
# ar_en4[ar_en4 == 0] = np.nan
# lambda_en4[lambda_en4 == 0] = np.nan
#
#
# var_en4, lonc = add_cyclic_point(var_en4, coord=lon)
# ar_en4, lonc = add_cyclic_point(ar_en4, coord=lon)
# lambda_en4, lonc = add_cyclic_point(lambda_en4, coord=lon)
#
# var_en4_p, lonc = add_cyclic_point(var_en4_p, coord=lon)
# ar_en4_p, lonc = add_cyclic_point(ar_en4_p, coord=lon)
# lambda_en4_p, lonc = add_cyclic_point(lambda_en4_p, coord=lon)
#
#
#
# am_sst_global_runstd = np.load('data/am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sst_global_runac = np.load('data/am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# am_sst_global_runlambda = np.load('data/am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# m_am_sst_global_runstd = np.nanmean(am_sst_global_runstd, axis = 0)
# m_am_sst_global_runac = np.nanmean(am_sst_global_runac, axis = 0)
# m_am_sst_global_runlambda = np.nanmean(am_sst_global_runlambda, axis = 0)
#
# var_sst = np.load('data/trend_am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_sst = np.load('data/trend_am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_sst = np.load('data/trend_am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
# var_sst_p = np.load('data/trend_p_am_sst_global_runstd_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# ar_sst_p = np.load('data/trend_p_am_sst_global_runac_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
# lambda_sst_p = np.load('data/trend_p_am_sst_global_runlambda_cor_d%d_sm%d_rmw%d.npy'%(dth, sm_w, rmw))
#
#
# m_am_sst_global_runstd, sst_lonc = add_cyclic_point(m_am_sst_global_runstd, coord=sst_lon)
# m_am_sst_global_runac, sst_lonc = add_cyclic_point(m_am_sst_global_runac, coord=sst_lon)
# m_am_sst_global_runlambda, sst_lonc = add_cyclic_point(m_am_sst_global_runlambda, coord=sst_lon)
#
# var_sst, sst_lonc = add_cyclic_point(var_sst, coord=sst_lon)
# ar_sst, sst_lonc = add_cyclic_point(ar_sst, coord=sst_lon)
# lambda_sst, sst_lonc = add_cyclic_point(lambda_sst, coord=sst_lon)
#
# var_sst_p, sst_lonc = add_cyclic_point(var_sst_p, coord=sst_lon)
# ar_sst_p, sst_lonc = add_cyclic_point(ar_sst_p, coord=sst_lon)
# lambda_sst_p, sst_lonc = add_cyclic_point(lambda_sst_p, coord=sst_lon)
#
#
# var_sst = var_sst * 100
# ar_sst = ar_sst * 100
# lambda_sst = lambda_sst * 100
#
# var_en4 = var_en4 * 100
# ar_en4 = ar_en4 * 100
# lambda_en4 = lambda_en4 * 100


#
# fig = plt.figure(figsize = (15,10))
#
# ax = fig.add_subplot(1, 3, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,20), extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.9,-.3,7))
#
# gl = ax.gridlines(linestyle=":", draw_labels=False)
#
#
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# # ax = fig.add_subplot(1, 3, 1)
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda)
#
# ax = fig.add_subplot(1, 3, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runstd, levels = np.linspace(0., .012   ,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runstd, levels = np.linspace(0., .012   ,20), extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(0., .012   ,7))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'Var')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(1, 3, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runac, levels = np.linspace(.2, .8,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runac, levels = np.linspace(.2, .7,20), extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(.2, .7, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'AC1')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# fig.savefig('plots/Sal_meanEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')
#
# fig = plt.figure(figsize = (15,10))
# ax = fig.add_subplot(1, 3, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.8, .8, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.8, .8, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
# ax = fig.add_subplot(1, 3, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, var_en4, levels = np.linspace(-.01, .01, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, var_en4, levels = np.linspace(-.01, .01, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.01, .01, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ Var / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
# ax = fig.add_subplot(1, 3, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(lonc, lat, ar_en4, levels = np.linspace(-.6, .6,6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, ar_en4, levels = np.linspace(-.8, .8,20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.8, .8,6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ AC1 / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# fig.savefig('plots/Sal_trendEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')
#
#
# fig = plt.figure(figsize = (15,10))
#
# ax = fig.add_subplot(1, 3, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,13), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,20), extend= 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-1.1,-.5,7))
# gl = ax.gridlines(linestyle=":", draw_labels=False)
#
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# # ax = fig.add_subplot(1, 3, 1)
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda)
#
# ax = fig.add_subplot(1, 3, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runstd, levels = np.linspace(0., .6,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runstd, levels = np.linspace(0., .6,20), extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(0., .6,7))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'Var')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(1, 3, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runac, levels = np.linspace(-.1, .5,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runac, levels = np.linspace(-.1, .5,20), extend = 'both', transform = data_crs)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.1, .5,7))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'AC1')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# fig.savefig('plots/SST_meanEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')
#
# fig = plt.figure(figsize = (15,10))
# ax = fig.add_subplot(1, 3, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.6, .6, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
# ax = fig.add_subplot(1, 3, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, var_sst, levels = np.linspace(-.2, .2, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, var_sst, levels = np.linspace(-.2, .2, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.2, .2, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ Var / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
# ax = fig.add_subplot(1, 3, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, ar_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, ar_sst, levels = np.linspace(-.6, .6, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.6, .6, 6))
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ AC1 / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# fig.savefig('plots/SST_trendEWS_spatial_d%d_sm%d_ws%d.pdf'%(dth, sm_w, rmw), bbox_inches = 'tight')






area_ceaser_indices = trmm_indices_for_area2(area_ceaser_coords, sst_lat, sst_lon)

area_ceaser = np.ones(sst_la * sst_lo)
area_ceaser[area_ceaser_indices] = -1
area_ceaser = area_ceaser.reshape((sst_lat.shape[0], sst_lon.shape[0]))

sgi = area_ceaser_indices

area = np.ones(sst_la * sst_lo)
area[sgi] = -1
area = area.reshape((sst_lat.shape[0], sst_lon.shape[0]))

dipole1 = coordinate_indices_from_ra(sst_lat, sst_lon, 80, 45, 30, -70)
dipole2 = coordinate_indices_from_ra(sst_lat, sst_lon, 0, -45, 30, -70)




dipole1_area = np.ones(sst_la * sst_lo)
dipole1_area[dipole1] = -1
dipole1_area = dipole1_area.reshape((sst_lat.shape[0], sst_lon.shape[0]))

dipole2_area = np.ones(sst_la * sst_lo)
dipole2_area[dipole2] = -1
dipole2_area = dipole2_area.reshape((sst_lat.shape[0], sst_lon.shape[0]))


sst_trends = np.load('data/sst_trends.npy')
nsst_trends = np.load('data/nsst_trends.npy')
sal_trends = np.load('data/sal_trends.npy')

sst_trends, sst_lonc = add_cyclic_point(sst_trends, coord=sst_lon)

nsst_trends, sst_lonc = add_cyclic_point(nsst_trends, coord=sst_lon)


# area, sst_lonc = add_cyclic_point(area, coord=sst_lon)
# area_ceaser, sst_lonc = add_cyclic_point(area_ceaser, coord=sst_lon)
# dipole1_area, sst_lonc = add_cyclic_point(dipole1_area, coord=sst_lon)
# dipole2_area, sst_lonc = add_cyclic_point(dipole2_area, coord=sst_lon)
#
# sal_trends, lonc = add_cyclic_point(sal_trends, coord=lon)
#
#
# area_nn = np.ones((la,lo))
# area_nn[np.ix_(np.logical_and(lat > 44, lat < 66), np.logical_or(lon > 289, lon < 30))] = -1
# area_nn, lonc = add_cyclic_point(area_nn, coord=lon)
#
# area_n = np.ones((la, lo))
# area_n[np.ix_(np.logical_and(lat > 10, lat < 40), np.logical_or(lon > 289, lon < 30))] = -1
# area_n, lonc = add_cyclic_point(area_n, coord=lon)
#
# area_s = np.ones((la, lo))
# area_s[np.ix_(np.logical_and(lat > -34, lat < -10), np.logical_or(lon > 289, lon < 30))] = -1
# area_s, lonc = add_cyclic_point(area_s, coord=lon)
#
#
# area_klus = np.ones((la, lo))
# area_klus[np.ix_(np.logical_and(lat > 54, lat < 62), np.logical_and(lon > 298, lon < 334))] = -1
# area_klus, lonc = add_cyclic_point(area_klus, coord=lon)
#
#
#
# ## fig = plt.figure(figsize = (9,15))
# fig = plt.figure(figsize = (10,11))
# # ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
# ax = fig.add_subplot(3, 2, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Sea-surface temperatures', fontweight="bold")
# # ax.coastlines()
# cf = ax.contourf(sst_lonc, sst_lat, sst_trends, levels = np.linspace(-.8,.8,20), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = [-.8,-.4,0,.4,.8])
# cbar = plt.colorbar(cf, ax = ax, ticks = [-.8,-.4,0,.4,.8], shrink = .8)
# cbar.set_label(r'SST trend [K / 100yr]')
# # cs = ax.contour(sst_lonc, sst_lat, area, [0], linestyles = 'solid', linewidths = 2., colors = 'r', transform = data_crs)
# cs = ax.contour(sst_lonc, sst_lat, area_ceaser, [0], linestyles = 'solid', linewidths = 2., colors = 'b', transform = data_crs)
# cs = ax.contour(sst_lonc, sst_lat, dipole1_area, [0], linestyles = 'solid', linewidths = 2., colors = 'c', transform = data_crs)
# cs = ax.contour(sst_lonc, sst_lat, dipole2_area, [0], linestyles = 'solid', linewidths = 2., colors = 'm', transform = data_crs)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
# ax.add_feature(cartopy.feature.LAND, color='.3')
# # ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson())
#
#
# ax = fig.add_subplot(3, 2, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Salinity', fontweight="bold")
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, nsst_trends, levels = np.linspace(-1,3, 101), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# cf = ax.contourf(lonc, lat, sal_trends, levels = np.linspace(-.15,.15, 20), cmap = plt.cm.RdBu_r, extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = [-.15,-.075,0,.075,.15])
# cbar = plt.colorbar(cf, ax = ax, ticks = [-.15,-.075,0,.075,.15], shrink = .8)
# # cbar.set_label(r'normalized SST trend')
# cbar.set_label(r'Salinity trends [psu / 100yr]')
# cs = ax.contour(lonc, lat, area_nn, [0], linestyles = 'solid', linewidths = 2., colors = 'k', transform = data_crs)
# cs = ax.contour(lonc, lat, area_n, [0], linestyles = 'solid', linewidths = 2., colors = 'orange', transform = data_crs)
# cs = ax.contour(lonc, lat, area_s, [0], linestyles = 'solid', linewidths = 2., colors = 'r', transform = data_crs)
# cs = ax.contour(lonc, lat, area_klus, [0], linestyles = 'solid', linewidths = 2., colors = 'y', transform = data_crs)
#
# ax.gridlines(linestyle=":", draw_labels=False)
# ax.add_feature(cartopy.feature.LAND, color='.3')
#
#
#
# ax = fig.add_subplot(3, 2, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,13), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,20), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-1.1,-.5,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-1.1,-.5,7), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='.3')
#
#
# ax = fig.add_subplot(3, 2, 4, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,20), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.9,-.3,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.9,-.3,7), shrink = .8)
#
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='.3')
#
# ax = fig.add_subplot(3, 2, 5, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(sst_lonc, sst_lat, lambda_sst_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.6, .6, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.6, .6, 6), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='.3')
#
# ax = fig.add_subplot(3, 2, 6, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.8, .8, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(lonc, lat, lambda_en4_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.8, .8, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.8, .8, 6), shrink = .8)
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='.3')
# plt.subplots_adjust(wspace = .02, hspace = .1)
# fig.savefig('plots/SST_SAL_EWS_global_trends_%s_sm%d_rmw%d_klus.pdf'%(sst_data, sm_w, rmw), bbox_inches = 'tight')

#
#
#
#
# fig = plt.figure(figsize = (10,11))
# ax = fig.add_subplot(3, 2, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Sea-surface temperatures', fontweight="bold")
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,13), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,20), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-1.1,-.5,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-1.1,-.5,7), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
# ax = fig.add_subplot(3, 2, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Salinity', fontweight="bold")
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,20), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.9,-.3,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.9,-.3,7), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\lambda$')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
#
# ax = fig.add_subplot(3, 2, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,13), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runstd, levels = np.linspace(0.,.6,20), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(0.,.6,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(0.,.6,7), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'Var')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
# ax = fig.add_subplot(3, 2, 4, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runstd, levels = np.linspace(0.,.012,20), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(0,.012,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(0,.012,7), shrink = .8)
#
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'Var')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
# ax = fig.add_subplot(3, 2, 5, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runlambda, levels = np.linspace(-1.1,-.5,13), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, m_am_sst_global_runac, levels = np.linspace(-.1,.5,20), extend= 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.1,.5,7))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.1,.5,7), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'AC1')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
# ax = fig.add_subplot(3, 2, 6, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, m_am_sal_global_runlambda, levels = np.linspace(-.9,-.3,13), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, m_am_sal_global_runac, levels = np.linspace(.2,.7,20), extend = 'both', transform = data_crs)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(.2,.7,6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(.2,.7,6), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'AC1')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
#
# plt.subplots_adjust(wspace = .02, hspace = .1)
# fig.savefig('plots/SST_SAL_all_EWS_global_trends_%s_sm%d_rmw%d.pdf'%(sst_data, sm_w, rmw), bbox_inches = 'tight')
#
#
#
# fig = plt.figure(figsize = (10,11))
# # ax = fig.add_subplot(2, 1, 1, projection=ccrs.Robinson())
# ax = fig.add_subplot(3, 2, 1, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Sea-surface temperatures', fontweight="bold")
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(sst_lonc, sst_lat, lambda_sst_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.6, .6, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.6, .6, 6), shrink = .8)
#
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(3, 2, 2, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.set_title('Salinity', fontweight="bold")
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.8, .8, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(lonc, lat, lambda_en4_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
#
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.8, .8, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.8, .8, 6), shrink = .8)
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta \lambda$ / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(3, 2, 3, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, var_sst, levels = np.linspace(-.2, .2, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(sst_lonc, sst_lat, var_sst_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
#
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.2, .2, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.2, .2, 6), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\Delta$ Var / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(3, 2, 4, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, var_en4, levels = np.linspace(-.01, .01, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(lonc, lat, var_en4_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
#
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.01, .01, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.01, .01, 6), shrink = .8)
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ Var / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
#
# ax = fig.add_subplot(3, 2, 5, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(sst_lonc, sst_lat, lambda_sst, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(sst_lonc, sst_lat, ar_sst, levels = np.linspace(-.6, .6, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(sst_lonc, sst_lat, ar_sst_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
#
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.6, .6, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.6, .6, 6), shrink = .8)
# gl = ax.gridlines(linestyle=":", draw_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
# gl.ylocator = mticker.FixedLocator(np.arange(-90,90,30))
#
# cbar.set_label(r'$\Delta$ AC1 / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# ax = fig.add_subplot(3, 2, 6, projection=ccrs.Orthographic(central_longitude=340))
# ax.text(0, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.coastlines()
# # cf = ax.contourf(lonc, lat, lambda_en4, levels = np.linspace(-.6, .6, 6), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01)
# cf = ax.contourf(lonc, lat, ar_en4, levels = np.linspace(-.8, .8, 20), extend = 'both', transform = data_crs, cmap = plt.cm.RdBu_r)
# hatch = ax.contourf(lonc, lat, ar_en4_p, levels = [0, .05, 2], transform = data_crs, colors = 'none', hatches = ['.....', None])
#
# # cbar = plt.colorbar(cf, ax = ax, orientation = 'horizontal', pad = .01, ticks = np.linspace(-.8, .8, 6))
# cbar = plt.colorbar(cf, ax = ax, ticks = np.linspace(-.8, .8, 6), shrink = .8)
# ax.gridlines(linestyle=":", draw_labels=False)
# cbar.set_label(r'$\Delta$ AC1 / 100yr')
# ax.add_feature(cartopy.feature.LAND, color='k')
#
# plt.subplots_adjust(wspace = .02, hspace = .1)
# fig.savefig('plots/SST_SAL_trends_EWS_global_trends_%s_sm%d_rmw%d.pdf'%(sst_data, sm_w, rmw), bbox_inches = 'tight')
#
#
# # # print(var_en4)
# # # print(ar_en4)
# # # fig = plt.figure(figsize = (12,6 ))
# # # ax = fig.add_subplot(311)
# # # ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(m_am_sal_global_runstd, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(0, .02, 11), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'Variance')
# # # ax = fig.add_subplot(312)
# # # ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(m_am_sal_global_runac, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(m_am_sal_global_runac.min(), m_am_sal_global_runac.max(), 11), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'AC1')
# # # ax = fig.add_subplot(313)
# # # ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(m_am_sal_global_runlambda, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(m_am_sal_global_runlambda.min(), m_am_sal_global_runlambda.max(), 11), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = r'$\lambda$')
# # #
# # # fig.savefig('plots/EN4_EWSmean.pdf', bbox_inches = 'tight')
# # #
# # # fig = plt.figure(figsize = (12,6 ))
# # # ax = fig.add_subplot(311)
# # # ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(var_en4, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(-.0002, .0002, 10), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'Variance')
# # # ax = fig.add_subplot(312)
# # # ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(ar_en4, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(-.01, .01, 10), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = 'AC1')
# # # ax = fig.add_subplot(313)
# # # ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # basemap(lambda_en4, lat, lon, res = 'c', lms = 2., proj = 'moll', shift = 0., contours = np.linspace(-.01, .01, 10), color = plt.cm.RdBu_r, alpha = .7, colorbar = True, extend = 'both', meridians = np.arange(0,360, 40), parallels = np.arange(-80,80,40), cbar_title = r'$\lambda$')
# # #
# # # fig.savefig('plots/EN4_EWS_global.pdf', bbox_inches = 'tight')
# #
# #
# # #
# #
#
sal = np.load('data/EN421_NA_salinity_d%d.npy'%dth)
sal = (sal - sal.mean())# / sal.std()

sal_n = np.load('data/EN421_SSS_N_d%d.npy'%dth)
sal_s = np.load('data/EN421_SSS_S_d%d.npy'%dth)
sal_sg = np.load('data/EN421_SG_d%d.npy'%dth)
sal_klus = np.load('data/EN421_KLUS_d%d.npy'%dth)

sal_n = (sal_n - sal_n.mean())
sal_s = (sal_s - sal_s.mean())
sal_sg = (sal_sg - sal_sg.mean())
sal_klus = (sal_klus - sal_klus.mean())

time = np.arange(1900, 2020 + 3/12, 1/12)

chentung = np.loadtxt('data/chen_tung_2018.txt')
chentung_time = chentung[:, 0]
chentung_amoc1 = chentung[:, 1]
chentung_amoc2 = chentung[:, 2]
chentung_amoc1[chentung_amoc1 == -100] = np.nan




amoc_sst = np.loadtxt('data/amoc_idx_niklas.txt')
amoc_sst_amo = np.loadtxt('data/amoc-amo_idx_niklas.txt')
amoc_sst_dipole = np.loadtxt('data/amoc-amo_idx_dipole.txt')
amoc_sst_sg = np.loadtxt('data/amoc-amo_idx_rahmstorf.txt')



time_sst = np.arange(1870, 2019)

# amoc_sst_am = np.zeros(time_sst.shape[0])
# for i in range(years.shape[0]):
#     amoc_sst_am[i] = np.mean(amoc_sst[i * 12 : (i + 1) * 12])


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

sal_sg_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_sg_am[i] = np.mean(sal_sg[i * 12 : (i + 1) * 12])

sal_klus_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    sal_klus_am[i] = np.mean(sal_klus[i * 12 : (i + 1) * 12])

norm_sal_global_am = np.zeros(years.shape[0])
for i in range(years.shape[0]):
    norm_sal_global_am[i] = np.mean(norm_sal_global[i * 12 : (i + 1) * 12])

norm_sal_global = norm_sal_global_am

rm_amoc_sst = runmean(amoc_sst, sm_w)
h_amoc_sst = amoc_sst - rm_amoc_sst

rm_amoc_sst_amo = runmean(amoc_sst_amo, sm_w)
h_amoc_sst_amo = amoc_sst_amo - rm_amoc_sst_amo

rm_amoc_sst_dipole = runmean(amoc_sst_dipole, sm_w)
h_amoc_sst_dipole = amoc_sst_dipole - rm_amoc_sst_dipole

rm_amoc_sst_sg = runmean(amoc_sst_sg, sm_w)
h_amoc_sst_sg = amoc_sst_sg - rm_amoc_sst_sg


rm_sal = runmean(sal_am, sm_w)
h_sal = sal_am - rm_sal


rm_sal_n = runmean(sal_n_am, sm_w)
h_sal_n = sal_n_am - rm_sal_n

rm_sal_s = runmean(sal_s_am, sm_w)
h_sal_s = sal_s_am - rm_sal_s

rm_sal_sg = runmean(sal_sg_am, sm_w)
h_sal_sg = sal_sg_am - rm_sal_sg

rm_sal_klus = runmean(sal_klus_am, sm_w)
h_sal_klus = sal_klus_am - rm_sal_klus

ir_sst = np.linspace(-1, 1, 100)
ir_sal = np.linspace(-.15, .15, 100)

kde1 = st.gaussian_kde(h_amoc_sst)
kde1 = kde1.evaluate(ir_sst)
kde2 = st.gaussian_kde(h_amoc_sst_amo)
kde2 = kde2.evaluate(ir_sst)
kde3 = st.gaussian_kde(h_amoc_sst_sg)
kde3 = kde3.evaluate(ir_sst)
kde4 = st.gaussian_kde(h_amoc_sst_dipole)
kde4 = kde4.evaluate(ir_sst)

kde5 = st.gaussian_kde(h_sal)
kde5 = kde5.evaluate(ir_sal)
kde6 = st.gaussian_kde(h_sal_klus)
kde6 = kde6.evaluate(ir_sal)
kde7 = st.gaussian_kde(h_sal_n)
kde7 = kde7.evaluate(ir_sal)
kde8 = st.gaussian_kde(h_sal_s)
kde8 = kde8.evaluate(ir_sal)


fig = plt.figure(figsize = (8,4))
ax = fig.add_subplot(121)
ax.plot(ir_sst, kde1, 'b-', lw = 2, label = r'SST$_{SG-GM}$, s = %.2f'%(st.skew(h_amoc_sst)))
ax.plot(ir_sst, kde2, 'r-', lw = 2, label = r'SST$_{SG-GM-AMO}$, s = %.2f'%(st.skew(h_amoc_sst_amo)))
ax.plot(ir_sst, kde3, 'c-', lw = 2, label = r'SST$_{SG-NH}$, s = %.2f'%(st.skew(h_amoc_sst_sg)))
ax.plot(ir_sst, kde4, 'm-', lw = 2, label = r'SST$_{DIPOLE}$, s = %.2f'%(st.skew(h_amoc_sst_dipole)))
ax.legend()
ax.grid()
ax.set_ylabel('PDF')
ax.set_xlabel(r'SST anomaly [${}^{\circ}$C]')

ax = fig.add_subplot(122)
ax.plot(ir_sal, kde5, 'k-', lw = 2, label = r'S$_{NN1}$, s = %.2f'%(st.skew(h_sal)))
ax.plot(ir_sal, kde6, 'y-', lw = 2, label = r'S$_{NN2}$, s = %.2f'%(st.skew(h_sal_klus)))
ax.plot(ir_sal, kde7, color = 'orange', lw = 2, ls = '-', label = r'S$_{N}$, s  = %.2f'%(st.skew(h_sal_n)))
ax.plot(ir_sal, kde8, 'r-', lw = 2, label = r'S$_{S}$, s = %.2f'%(st.skew(h_sal_s)))
ax.legend()
ax.grid()
ax.set_ylabel('PDF')
ax.set_xlabel(r'Salinity anomaly [psu]')
fig.savefig('plots/AMOC_indices_KDEs_ws%d.pdf'%ws, bbox_inches = 'tight')
#
# # fig = plt.figure(figsize = (8,8))
# # ax = fig.add_subplot(211)
# # ax.text(-.13, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # ax.plot(time, sal)
# # ax.plot(time_sst, amoc_sst, color = 'b', label = r'SST$_{SG-GM}$')
# # ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 2)
# #
# # ax.plot(time_sst, amoc_sst_amo, color = 'r', label = r'SST$_{SG-GM-AMO}$')
# # ax.plot(time_sst, rm_amoc_sst_amo, color = 'r', lw = 2)
# #
# # ax.plot(time_sst, amoc_sst_sg, color = 'c', label = r'SST$_{SG-NH}$')
# # ax.plot(time_sst, rm_amoc_sst_sg, color = 'c', lw = 2)
# #
# # ax.plot(time_sst, amoc_sst_dipole, color = 'm', label = r'SST$_{DIPOLE}$')
# # ax.plot(time_sst, rm_amoc_sst_dipole, color = 'm', lw = 2)
# # ax.grid(True)
# #
# # ax.set_xlim(1870, 2020)
# # ax.legend(loc = 2)
# # ax.set_ylabel('SST-based AMOC index [°C]')
# #
# # ax = fig.add_subplot(212)
# # ax.text(-.13, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # ax.plot(time, sal)
# # # ax.plot(time_sst, amoc_sst, color = 'b', label = 'SST AMOC index')
# # # ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 2)
# # # ax.set_xlim(1870, 2020)
# # # ax.legend(loc = 2)
# # # ax.set_ylabel('SST anomaly [°C]', color = 'b')
# # # ax2 = ax.twinx()
# # ax.plot(years, -sal_am, color = 'k', label = 'S$_{NN}$')
# # ax.plot(years, -rm_sal, color = 'k', lw = 2)
# # # ax2.plot(years, sal_sg_am, color = 'c', label = 'S$_{SG}$')
# # # ax2.plot(years, rm_sal_sg, color = 'c', lw = 2)
# #
# # ax.plot(years, -sal_n_am, color = 'orange', label = 'S$_{N}$')
# # ax.plot(years, -rm_sal_n, color = 'orange', lw = 2)
# # ax.plot(years, -sal_s_am, color = 'r', label = 'S$_{S}$')
# # ax.plot(years, -rm_sal_s, color = 'r', lw = 2)
# # # ax2.plot(chentung_time, chentung_amoc2, color = 'g', label = 'CT2018')
# # ax.legend(loc = 2)
# # ax.set_ylabel('Salinity-based AMOC index [psu]')
# # ax.set_xlim(1870, 2020)
# # ax.set_xlabel('Time [yr AD]')
# # ax.grid(True)
# # fig.savefig('plots/AMOC_indices_d%d_rm%d.pdf'%(dth, sm_w), bbox_inches = 'tight')
# #
# # fig = plt.figure(figsize = (12,8))
# #
# # ax = fig.add_subplot(211)
# # ax.text(-.13, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time_sst, h_amoc_sst, color = 'b', label = r'SST$_{SG-GM}$')
# #
# # ax.plot(time_sst, h_amoc_sst_amo, color = 'r', label = r'SST$_{SG-GM-AMO}$')
# #
# # ax.plot(time_sst, h_amoc_sst_sg, color = 'c', label = r'SST$_{SG-NH}$')
# #
# # ax.plot(time_sst, h_amoc_sst_dipole, color = 'm', label = r'SST$_{DIPOLE}$')
# # ax.grid(True)
# # ax.set_xlim(1870, 2020)
# # ax.legend(loc = 2)
# # ax.set_ylabel('Detrended SST-based AMOC index [°C]')
# #
# # ax = fig.add_subplot(212)
# # ax.text(-.13, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(years, h_sal, color = 'k', label = r'S$_{NN}$')
# # ax.plot(years, -h_sal_n, color = 'orange', label = r'S$_{N}$')
# # ax.plot(years, -h_sal_s, color = 'r', label = 'S$_{S}$')
# # # ax2.plot(years, h_sal_sg, color = 'c', label = 'S$_{SG}$')
# # # ax2.plot(chentung_time, chentung_amoc2, color = 'g', label = 'CT2018')
# # ax.legend(loc = 2)
# # ax.grid(True)
# # ax.set_xlim(1870, 2020)
# # ax.set_xlabel('Time [yr AD]')
# # ax.set_ylabel('Detrended salinity-based AMOC index [psu]')
# # fig.savefig('plots/AMOC_indices_detr_d%d_rm%d.pdf'%(dth, sm_w), bbox_inches = 'tight')
#
#
#
# fig = plt.figure(figsize = (11,12))
# ax = fig.add_subplot(421)
# ax.set_title('SST-based AMOC indices', fontweight="bold")
# ax.text(-.13, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time, sal)
# ax.plot(time_sst, amoc_sst, color = 'b', alpha = .8, label = r'SST$_{SG-GM}$')
# ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 3)
#
# ax.plot(time_sst, amoc_sst_amo, color = 'r', alpha = .8, label = r'SST$_{SG-GM-AMO}$')
# ax.plot(time_sst, rm_amoc_sst_amo, color = 'r', lw = 3)
#
# ax.plot(time_sst, amoc_sst_sg, color = 'c', alpha = .8, label = r'SST$_{SG-NH}$')
# ax.plot(time_sst, rm_amoc_sst_sg, color = 'c', lw = 3)
#
# ax.plot(time_sst, amoc_sst_dipole, color = 'm', alpha = .8, label = r'SST$_{DIPOLE}$')
# ax.plot(time_sst, rm_amoc_sst_dipole, color = 'm', lw = 3)
# ax.grid(True)
#
# ax.set_xlim(1870, 2020)
# ax.legend(loc = 3, fontsize = 8)
# ax.set_ylabel('SSTs [°C]')
#
# ax = fig.add_subplot(422)
# ax.set_title('Salinity-based AMOC indices', fontweight="bold")
# ax.text(-.13, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time, sal)
# # ax.plot(time_sst, amoc_sst, color = 'b', label = 'SST AMOC index')
# # ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 2)
# # ax.set_xlim(1870, 2020)
# # ax.legend(loc = 2)
# # ax.set_ylabel('SST anomaly [°C]', color = 'b')
# # ax2 = ax.twinx()
# ax.plot(years, -sal_am, color = 'k', alpha = .8, label = 'S$_{NN1}$')
# ax.plot(years, -rm_sal, color = 'k', lw = 3)
# # ax2.plot(years, sal_sg_am, color = 'c', label = 'S$_{SG}$')
# # ax2.plot(years, rm_sal_sg, color = 'c', lw = 2)
#
# ax.plot(years, -sal_klus_am, color = 'y', alpha = .8, label = 'S$_{NN2}$')
# ax.plot(years, -rm_sal_klus, color = 'y', lw = 3)
#
#
# ax.plot(years, -sal_n_am, color = 'orange', alpha = .8, label = 'S$_{N}$')
# ax.plot(years, -rm_sal_n, color = 'orange', lw = 3)
# ax.plot(years, -sal_s_am, color = 'r', alpha = .8, label = 'S$_{S}$')
# ax.plot(years, -rm_sal_s, color = 'r', lw = 3)
#
#
# # ax2.plot(chentung_time, chentung_amoc2, color = 'g', label = 'CT2018')
# ax.legend(loc = 3, fontsize = 8)
# ax.set_ylabel('Salinity [psu]')
# ax.set_xlim(1900, 2020)
#
# ax.grid(True)
#
#
# # amoc_sst = h_amoc_sst
# # amoc_sst_sg = h_amoc_sst_sg
# # amoc_sst_amo = h_amoc_sst_amo
# # amoc_sst_dipole = h_amoc_sst_dipole
# #
# # sal_am = h_sal
# # sal_klus_am = h_sal_klus
# # sal_n_am = -h_sal_n
# # sal_s_am = -h_sal_s
#
#
#
#
# p0, p1 = np.polyfit(time_sst, amoc_sst, 1)
# # amoc_sst = amoc_sst - p0 * time_sst - p1
#
# p0, p1 = np.polyfit(years, sal_am, 1)
# # sal_am = sal_am - p0 * years - p1
#
#
# ax = fig.add_subplot(423)
# ax.text(-.12, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(h_amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(h_amoc_sst, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(h_amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(h_amoc_sst_amo, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(h_amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(h_amoc_sst_sg, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(h_amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(h_amoc_sst_dipole, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# ax.set_xlim(1870, 2020)
# ax.set_ylim((-1, 0))
# ax.legend(loc = 2, fontsize = 8)
# ax.grid(True)
#
# ax = fig.add_subplot(425)
# ax.text(-.12, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], runstd(h_amoc_sst, ws)[bound : - bound]**2, color = 'b', label = 'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(h_amoc_sst, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_amoc_sst, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(h_amoc_sst_amo, ws)[bound : - bound]**2, color = 'r', label = 'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(h_amoc_sst_amo, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_amoc_sst_amo, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(h_amoc_sst_sg, ws)[bound : - bound]**2, color = 'c', label = 'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(h_amoc_sst_sg, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_amoc_sst_sg, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(h_amoc_sst_dipole, ws)[bound : - bound]**2, color = 'm', label = 'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(h_amoc_sst_dipole, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_amoc_sst_dipole, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.set_xlim(1870, 2020)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# ax.grid(True)
#
# ax = fig.add_subplot(427)
# ax.text(-.12, 1, s = 'g', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], runac(h_amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(h_amoc_sst, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(h_amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(h_amoc_sst_amo, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(h_amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(h_amoc_sst_sg, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(h_amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(h_amoc_sst_dipole, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# ax.set_xlabel('Time [yr AD]')
# ax.set_xlim(1870, 2020)
# ax.set_ylim(0,1)
# ax.grid(True)
#
#
# ax = fig.add_subplot(424)
# ax.text(-.12, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], run_fit_a_ar1(h_sal, ws)[bound : - bound], color = 'k', label = r'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(h_sal, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(h_sal, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(h_sal_klus, ws)[bound : - bound], color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(h_sal_klus, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(h_sal_klus, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(h_sal_n, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(h_sal_n, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(h_sal_n, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(h_sal_s, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(h_sal_s, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(h_sal_s, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.set_ylim((-1, 0))
# ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_xlim(1900, 2020)
# ax.grid(True)
#
#
#
# ax = fig.add_subplot(426)
# ax.text(-.12, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], runstd(h_sal, ws)[bound : - bound]**2, color = 'k', label = r'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(h_sal, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_sal, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runstd(h_sal_klus, ws)[bound : - bound]**2, color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(h_sal_klus, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_sal_klus, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(years[bound : - bound], runstd(h_sal_n, ws)[bound : - bound]**2, color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(h_sal_n, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_sal_n, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runstd(h_sal_s, ws)[bound : - bound]**2, color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(h_sal_s, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(h_sal_s, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# ax.set_xlim(1900, 2020)
# ax.grid(True)
#
#
# ax = fig.add_subplot(428)
# ax.text(-.12, 1, s = 'h', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], runac(h_sal, ws)[bound : - bound], color = 'k', label  = 'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(h_sal, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_sal, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runac(h_sal_klus, ws)[bound : - bound], color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(h_sal_klus, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_sal_klus, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(years[bound : - bound], runac(h_sal_n, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(h_sal_n, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_sal_n, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runac(h_sal_s, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(h_sal_s, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(h_sal_s, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.grid(True)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# ax.set_xlim(1900, 2020)
# ax.set_ylim(0,1)
# ax.set_xlabel('Time [yr AD]')
#
# plt.subplots_adjust(hspace = .2)
# fig.savefig('plots/AMOC_indices_EWS_%s_d%d_sm%d_rmw%d_klus.pdf'%(sst_data, dth, sm_w, rmw), bbox_inches = 'tight')
#
#
#
# sal_am = - sal_am
# sal_n_am = - sal_n_am
# sal_s_am = - sal_s_am
# sal_klus_am = - sal_klus_am
#
# popt, cov = curve_fit(funcfit3, time_sst, amoc_sst, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_amoc_sst = funcfit3(time_sst, *popt)
#
# popt, cov = curve_fit(funcfit3, time_sst, amoc_sst_amo, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_amoc_sst_amo = funcfit3(time_sst, *popt)
#
# popt, cov = curve_fit(funcfit3, time_sst, amoc_sst_sg, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_amoc_sst_sg = funcfit3(time_sst, *popt)
#
# popt, cov = curve_fit(funcfit3, time_sst, amoc_sst_dipole, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_amoc_sst_dipole = funcfit3(time_sst, *popt)
#
# popt, cov = curve_fit(funcfit3, years, sal_am, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_sal = funcfit3(years, *popt)
#
# popt, cov = curve_fit(funcfit3, years, sal_n_am, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_sal_n = funcfit3(years, *popt)
#
# popt, cov = curve_fit(funcfit3, years, sal_s_am, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_sal_s = funcfit3(years, *popt)
#
# popt, cov = curve_fit(funcfit3, years, sal_klus_am, p0 = [-1,.1,2030], maxfev = 1000000000, jac = funcfit3_jac)
# rm_sal_klus = funcfit3(years, *popt)
#
#
# fig = plt.figure(figsize = (11,12))
# ax = fig.add_subplot(421)
# ax.set_title('SST-based AMOC indices', fontweight="bold")
# ax.text(-.13, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time, sal)
# ax.plot(time_sst, amoc_sst, color = 'b', alpha = .8, label = r'SST$_{SG-GM}$')
# ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 3)
#
# ax.plot(time_sst, amoc_sst_amo, color = 'r', alpha = .8, label = r'SST$_{SG-GM-AMO}$')
# ax.plot(time_sst, rm_amoc_sst_amo, color = 'r', lw = 3)
#
# ax.plot(time_sst, amoc_sst_sg, color = 'c', alpha = .8, label = r'SST$_{SG-NH}$')
# ax.plot(time_sst, rm_amoc_sst_sg, color = 'c', lw = 3)
#
# ax.plot(time_sst, amoc_sst_dipole, color = 'm', alpha = .8, label = r'SST$_{DIPOLE}$')
# ax.plot(time_sst, rm_amoc_sst_dipole, color = 'm', lw = 3)
# ax.grid(True)
#
# ax.set_xlim(1870, 2020)
# ax.legend(loc = 3, fontsize = 8)
# ax.set_ylabel('SSTs [°C]')
#
# ax = fig.add_subplot(422)
# ax.set_title('Salinity-based AMOC indices', fontweight="bold")
# ax.text(-.13, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time, sal)
# # ax.plot(time_sst, amoc_sst, color = 'b', label = 'SST AMOC index')
# # ax.plot(time_sst, rm_amoc_sst, color = 'b', lw = 2)
# # ax.set_xlim(1870, 2020)
# # ax.legend(loc = 2)
# # ax.set_ylabel('SST anomaly [°C]', color = 'b')
# # ax2 = ax.twinx()
# ax.plot(years, sal_am, color = 'k', alpha = .8, label = 'S$_{NN1}$')
# ax.plot(years, rm_sal, color = 'k', lw = 3)
# # ax2.plot(years, sal_sg_am, color = 'c', label = 'S$_{SG}$')
# # ax2.plot(years, rm_sal_sg, color = 'c', lw = 2)
#
# ax.plot(years, sal_klus_am, color = 'y', alpha = .8, label = 'S$_{NN2}$')
# ax.plot(years, rm_sal_klus, color = 'y', lw = 3)
#
# ax.plot(years, sal_n_am, color = 'orange', alpha = .8, label = 'S$_{N}$')
# ax.plot(years, rm_sal_n, color = 'orange', lw = 3)
# ax.plot(years, sal_s_am, color = 'r', alpha = .8, label = 'S$_{S}$')
# ax.plot(years, rm_sal_s, color = 'r', lw = 3)
#
#
#
# # ax2.plot(chentung_time, chentung_amoc2, color = 'g', label = 'CT2018')
# ax.legend(loc = 3, fontsize = 8)
# ax.set_ylabel('Salinity [psu]')
# ax.set_xlim(1900, 2020)
#
# ax.grid(True)
#
#
# amoc_sst = amoc_sst - rm_amoc_sst
# amoc_sst_sg = amoc_sst_sg - rm_amoc_sst_sg
# amoc_sst_amo = amoc_sst_amo - rm_amoc_sst_amo
# amoc_sst_dipole = amoc_sst_dipole - rm_amoc_sst_dipole
#
# sal_am = sal_am - rm_sal
# sal_n_am = sal_n_am - rm_sal_n
# sal_s_am = sal_s_am - rm_sal_s
# sal_klus_am = sal_klus_am - rm_sal_klus
#
#
#
#
#
# ax = fig.add_subplot(423)
# ax.text(-.12, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_amo, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_sg, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_dipole, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# ax.set_xlim(1870, 2020)
# ax.set_ylim((-1, 0))
# ax.legend(loc = 2, fontsize = 8)
# ax.grid(True)
#
# ax = fig.add_subplot(425)
# ax.text(-.12, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], runstd(amoc_sst, ws)[bound : - bound]**2, color = 'b', label = 'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc_sst, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(amoc_sst_amo, ws)[bound : - bound]**2, color = 'r', label = 'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_amo, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc_sst_amo, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(amoc_sst_sg, ws)[bound : - bound]**2, color = 'c', label = 'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_sg, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc_sst_sg, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runstd(amoc_sst_dipole, ws)[bound : - bound]**2, color = 'm', label = 'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_dipole, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(amoc_sst_dipole, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.set_xlim(1870, 2020)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# ax.grid(True)
#
# ax = fig.add_subplot(427)
# ax.text(-.12, 1, s = 'g', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(time_sst[bound : - bound], runac(amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_amo, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_sg, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(time_sst[bound : - bound], runac(amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_dipole, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# ax.set_xlabel('Time [yr AD]')
# ax.set_xlim(1870, 2020)
# ax.set_ylim(0,1)
# ax.grid(True)
#
#
# ax = fig.add_subplot(424)
# ax.text(-.12, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], run_fit_a_ar1(sal_am, ws)[bound : - bound], color = 'k', label = r'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(sal_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(sal_klus_am, ws)[bound : - bound], color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_klus_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(sal_klus_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_n_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], run_fit_a_ar1(sal_s_am, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_s_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(run_fit_a_ar1(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.set_ylim((-1, 0))
# ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_xlim(1900, 2020)
# ax.grid(True)
#
#
#
# ax = fig.add_subplot(426)
# ax.text(-.12, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], runstd(sal_am, ws)[bound : - bound]**2, color = 'k', label = r'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_am, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(sal_am, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(years[bound : - bound], runstd(sal_klus_am, ws)[bound : - bound]**2, color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_klus_am, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(sal_klus_am, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runstd(sal_n_am, ws)[bound : - bound]**2, color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_n_am, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(sal_n_am, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runstd(sal_s_am, ws)[bound : - bound]**2, color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_s_am, ws)[bound : -bound]**2, 1)
# pv = kendall_tau_test(runstd(sal_s_am, ws)[bound : -bound]**2, tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
#
#
#
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# ax.set_xlim(1900, 2020)
#
# ax.grid(True)
#
#
# ax = fig.add_subplot(428)
# ax.text(-.12, 1, s = 'h', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# ax.plot(years[bound : - bound], runac(sal_am, ws)[bound : - bound], color = 'k', label  = 'S$_{NN1}$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(sal_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(sal_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(years[bound : - bound], runac(sal_klus_am, ws)[bound : - bound], color = 'y', label = r'S$_{NN2}$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(sal_klus_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(sal_klus_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'y', ls = '--', label = r'$p = %.3f$'%pv)
#
#
# ax.plot(years[bound : - bound], runac(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(sal_n_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.plot(years[bound : - bound], runac(sal_s_am, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# p0, p1 = np.polyfit(years[bound : -bound], runac(sal_s_am, ws)[bound : -bound], 1)
# pv = kendall_tau_test(runac(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# if pv < .001:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p < 10^{-3}$')
# else:
#     ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = r'$p = %.3f$'%pv)
#
# ax.grid(True)
# ax.legend(loc = 2, fontsize = 8)
# ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# ax.set_xlim(1900, 2020)
# ax.set_ylim(0,1)
# ax.set_xlabel('Time [yr AD]')
#
# plt.subplots_adjust(hspace = .2)
# fig.savefig('plots/AMOC_indices_FP_EWS_%s_d%d_sm%d_rmw%d_klus.pdf'%(sst_data, dth, sm_w, rmw), bbox_inches = 'tight')
#
#
#
#
#
#
#
#
# # fig = plt.figure(figsize = (8, 12))
# # ax = fig.add_subplot(311)
# # ax.text(-.12, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_amo, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_sg, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst_dipole, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.4f'%pv)
# #
# #
# # ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# # ax.set_xlim(1870, 2020)
# # ax.set_ylim((-1, 0))
# # ax.legend(loc = 2, fontsize = 11)
# # ax.grid(True)
# #
# # ax = fig.add_subplot(312)
# # ax.text(-.12, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time_sst[bound : - bound], runstd(amoc_sst, ws)[bound : - bound]**2, color = 'b', label = 'SST$_{SG-GM}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(amoc_sst, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runstd(amoc_sst_amo, ws)[bound : - bound]**2, color = 'r', label = 'SST$_{SG-GM-AMO}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_amo, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(amoc_sst_amo, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runstd(amoc_sst_sg, ws)[bound : - bound]**2, color = 'c', label = 'SST$_{SG-NH}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_sg, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(amoc_sst_sg, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runstd(amoc_sst_dipole, ws)[bound : - bound]**2, color = 'm', label = 'SST$_{DIPOLE}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst_dipole, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(amoc_sst_dipole, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.4f'%pv)
# # ax.set_xlim(1870, 2020)
# # ax.legend(loc = 2, fontsize = 11)
# # ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# # ax.grid(True)
# #
# # ax = fig.add_subplot(313)
# # ax.text(-.12, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(time_sst[bound : - bound], runac(amoc_sst, ws)[bound : - bound], color = 'b', label = r'SST$_{SG-GM}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runac(amoc_sst_amo, ws)[bound : - bound], color = 'r', label = r'SST$_{SG-GM-AMO}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_amo, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_amo, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runac(amoc_sst_sg, ws)[bound : - bound], color = 'c', label = r'SST$_{SG-NH}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_sg, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_sg, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'c', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(time_sst[bound : - bound], runac(amoc_sst_dipole, ws)[bound : - bound], color = 'm', label = r'SST$_{DIPOLE}$')
# # p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst_dipole, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(amoc_sst_dipole, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.4f'%pv)
# #
# #
# # ax.legend(loc = 2, fontsize = 11)
# # ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# # ax.set_xlabel('Time [yr AD]')
# # ax.set_xlim(1870, 2020)
# # ax.grid(True)
# #
# #
# # fig.savefig('plots/EWS_SST_detrended_d%d_rm%d_ws%d.pdf'%(dth, sm_w, ws), bbox_inches = 'tight')
# #
# #
# # fig = plt.figure(figsize = (8, 12))
# # ax = fig.add_subplot(311)
# # ax.text(-.12, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(years[bound : - bound], run_fit_a_ar1(sal_am, ws)[bound : - bound], color = 'k', label = r'S$_{NN}$')
# # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(run_fit_a_ar1(sal_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], run_fit_a_ar1(sal_s_am, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_s_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(run_fit_a_ar1(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], run_fit_a_ar1(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_n_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(run_fit_a_ar1(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.set_ylim((-1, 0))
# # ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# # ax.legend(loc = 2, fontsize = 11)
# # ax.set_xlim(1870, 2020)
# # ax.grid(True)
# #
# #
# #
# # ax = fig.add_subplot(312)
# # ax.text(-.12, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(years[bound : - bound], runstd(sal_am, ws)[bound : - bound]**2, color = 'k', label = r'S$_{NN}$')
# # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_am, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(sal_am, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], runstd(sal_s_am, ws)[bound : - bound]**2, color = 'r', label = r'S$_S$')
# # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_s_am, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(sal_s_am, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], runstd(sal_n_am, ws)[bound : - bound]**2, color = 'orange', label = r'S$_N$')
# # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_n_am, ws)[bound : -bound]**2, 1)
# # pv = kendall_tau_test(runstd(sal_n_am, ws)[bound : -bound]**2, tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.legend(loc = 2, fontsize = 11)
# # ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# # ax.set_xlim(1870, 2020)
# # ax.grid(True)
# #
# #
# # ax = fig.add_subplot(313)
# # ax.text(-.12, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # ax.plot(years[bound : - bound], runac(sal_am, ws)[bound : - bound], color = 'k', label  = 'S$_{NN}$')
# # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(sal_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'k', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], runac(sal_s_am, ws)[bound : - bound], color = 'r', label = r'S$_S$')
# # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_s_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.plot(years[bound : - bound], runac(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'S$_N$')
# # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_n_am, ws)[bound : -bound], 1)
# # pv = kendall_tau_test(runac(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# # ax.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.4f'%pv)
# #
# # ax.grid(True)
# # ax.legend(loc = 2, fontsize = 11)
# # ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# # ax.set_xlim(1870, 2020)
# # ax.set_xlabel('Time [yr AD]')
# # fig.savefig('plots/EWS_Salinity_detrended_d%d_rm%d_ws%d.pdf'%(dth, sm_w, ws), bbox_inches = 'tight')
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # # fig = plt.figure(figsize = (8,14))
# # # ax = fig.add_subplot(411)
# # # ax.text(-.1, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # # ax.plot(time, sal)
# # # ax.plot(time_sst, amoc_sst, color = 'b', label = 'SST AMOC index')
# # # ax.set_xlim(1870, 2020)
# # # ax.legend(loc = 2)
# # # ax.set_ylabel('SST anomaly [°C]')
# # #
# # # ax2 = ax.twinx()
# # # ax2.plot(years, sal_am, color = 'm', label = 'S$_{NN}$')
# # # ax2.plot(years, sal_s_am, color = 'r', label = 'S$_{S}$')
# # # ax2.plot(years, sal_n_am, color = 'orange', label = 'S$_{N}$')
# # # ax2.legend(loc = 1)
# # # # ax2.plot(chentung_time, chentung_amoc2)
# # # ax2.set_ylabel('NA salinity anomaly [psu]')
# # #
# # # ax = fig.add_subplot(412)
# # # ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # ax.plot(time_sst[bound : - bound], run_fit_a_ar1(amoc_sst, ws)[bound : - bound], color = 'b', label = r'$\lambda$ SST AMOC index')
# # # p0, p1 = np.polyfit(time_sst[bound : -bound], run_fit_a_ar1(amoc_sst, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# # # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.5f'%pv)
# # # ax.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# # # ax.set_xlim(1870, 2020)
# # # ax.set_ylim((-1, 0))
# # # ax.legend(loc = 2)
# # #
# # #
# # # ax2 = ax.twinx()
# # # ax2.plot(years[bound : - bound], run_fit_a_ar1(sal_am, ws)[bound : - bound], color = 'm', label = r'$\lambda$ S$_{NN}$')
# # # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(run_fit_a_ar1(sal_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], run_fit_a_ar1(sal_s_am, ws)[bound : - bound], color = 'r', label = r'$\lambda$ S$_S$')
# # # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_s_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(run_fit_a_ar1(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], run_fit_a_ar1(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'$\lambda$ S_$N$')
# # # p0, p1 = np.polyfit(years[bound : -bound], run_fit_a_ar1(sal_n_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(run_fit_a_ar1(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.set_ylim((-1, 0))
# # # ax2.set_ylabel(r'$\lambda$ (ws = %d yr)'%ws)
# # # ax2.legend(loc = 4)
# # #
# # # ax = fig.add_subplot(413)
# # # ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # ax.plot(time_sst[bound : - bound], runstd(amoc_sst, ws)[bound : - bound]**2, color = 'b', label = 'Variance SST index')
# # # p0, p1 = np.polyfit(time_sst[bound : -bound], runstd(amoc_sst, ws)[bound : -bound]**2, 1)
# # # pv = kendall_tau_test(runstd(amoc_sst, ws)[bound : -bound]**2, tt_samples, p0)
# # # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.5f'%pv)
# # # ax.set_xlim(1870, 2020)
# # # ax.legend(loc = 2)
# # # ax.set_ylabel(r'Variance (ws = %d yr)'%ws)
# # #
# # # ax2 = ax.twinx()
# # # ax2.plot(years[bound : - bound], runstd(sal_am, ws)[bound : - bound]**2, color = 'm', label = r'Variance S$_{NN}$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_am, ws)[bound : -bound]**2, 1)
# # # pv = kendall_tau_test(runstd(sal_am, ws)[bound : -bound]**2, tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], runstd(sal_s_am, ws)[bound : - bound]**2, color = 'r', label = r'Variance S$_S$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_s_am, ws)[bound : -bound]**2, 1)
# # # pv = kendall_tau_test(runstd(sal_s_am, ws)[bound : -bound]**2, tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], runstd(sal_n_am, ws)[bound : - bound]**2, color = 'orange', label = r'Variance S$_N$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runstd(sal_n_am, ws)[bound : -bound]**2, 1)
# # # pv = kendall_tau_test(runstd(sal_n_am, ws)[bound : -bound]**2, tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.legend(loc = 4)
# # # ax2.set_ylabel(r'Variance (ws = %d yr)'%ws)
# # #
# # #
# # # ax = fig.add_subplot(414)
# # # ax.text(-.1, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
# # # ax.plot(time_sst[bound : - bound], runac(amoc_sst, ws)[bound : - bound], color = 'b', label = 'AC1 SST index')
# # # p0, p1 = np.polyfit(time_sst[bound : -bound], runac(amoc_sst, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(runac(amoc_sst, ws)[bound : -bound], tt_samples, p0)
# # # ax.plot(time_sst, p0 * time_sst + p1, color = 'b', ls = '--', label = 'p = %.5f'%pv)
# # # ax.legend(loc = 2)
# # # ax.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# # # ax.set_xlabel('Time [yr AD]')
# # #
# # # ax2 = ax.twinx()
# # # ax2.plot(years[bound : - bound], runac(sal_am, ws)[bound : - bound], color = 'm', label  = 'AC1 S$_{NN}$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(runac(sal_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'm', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], runac(sal_s_am, ws)[bound : - bound], color = 'r', label = r'AC1 S$_S$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_s_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(runac(sal_s_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'r', ls = '--', label = 'p = %.5f'%pv)
# # #
# # # ax2.plot(years[bound : - bound], runac(sal_n_am, ws)[bound : - bound], color = 'orange', label = r'AC1 S$_N$')
# # # p0, p1 = np.polyfit(years[bound : -bound], runac(sal_n_am, ws)[bound : -bound], 1)
# # # pv = kendall_tau_test(runac(sal_n_am, ws)[bound : -bound], tt_samples, p0)
# # # ax2.plot(time_sst, p0 * time_sst + p1, color = 'orange', ls = '--', label = 'p = %.5f'%pv)
# # #
# # #
# # # ax2.legend(loc = 4)
# # # ax2.set_ylabel(r'AC1 (ws = %d yr)'%ws)
# # #
# # # fig.savefig('plots/EN4NA_salinities_d%d_rm%d.pdf'%(dth, sm_w), bbox_inches = 'tight')
