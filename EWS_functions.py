import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from scipy.optimize import curve_fit

def fourrier_surrogates(ts, ns):
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def kendall_tau_test(ts, ns, tau, mode1 = 'fourier', mode2 = 'linear'):
    tlen = ts.shape[0]

    if mode1 == 'fourier':
        tsf = ts - ts.mean()
        nts = fourrier_surrogates(tsf, ns)
    elif mode1 == 'shuffle':
        nts = shuffle_surrogates(ts, ns)
    stat = np.zeros(ns)
    tlen = nts.shape[1]
    if mode2 == 'linear':
        for i in range(ns):
            stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
    elif mode2 == 'kt':
        for i in range(ns):
            stat[i] = st.kendalltau(np.arange(tlen), nts[i])[0]
    p = 1 - st.percentileofscore(stat, tau) / 100.
    return p


def runmean(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xs[i] = np.nanmean(x[: i + w // 2 + 1])
   for i in range(n - w // 2, n):
      xs[i] = np.nanmean(x[i - w // 2 + 1:])

   for i in range(w // 2, n - w // 2):
      xs[i] = np.nanmean(x[i - w // 2 : i + w // 2 + 1])
   return xs


def runstd(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xw = x[: i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan
   for i in range(n - w // 2, n):
      xw = x[i - w // 2 + 1:]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1


          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   for i in range(w // 2, n - w // 2):
      xw = x[i - w // 2 : i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   return xs

def runlf(x, y, w):
   n = x.shape[0]
   xs = np.zeros_like(y)
   for i in range(w // 2):
      xw = x[: i + w // 2 + 1]
      yw = y[: i + w // 2 + 1]
      yw = yw - yw.mean()

      xs[i] = np.polyfit(xw, yw, 1)[0]

   for i in range(n - w // 2, n):
      yw = y[i - w // 2 + 1:]
      xw = x[i - w // 2 + 1:]
      yw = yw - yw.mean()
      xs[i] = st.linregress(xw, yw)[0]

   for i in range(w // 2, n - w // 2):
      yw = y[i - w // 2 : i + w // 2 + 1]
      xw = x[i - w // 2 : i + w // 2 + 1]
      yw = yw - yw.mean()
      xs[i] = st.linregress(xw, yw)[0]
   return xs


def runac(x, w):
   n = x.shape[0]
   xs = np.zeros_like(x)
   for i in range(w // 2):
      xw = x[: i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(n - w // 2, n):
      xw = x[i - w // 2 + 1:]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(w // 2, n - w // 2):
      xw = x[i - w // 2 : i + w // 2 + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1
          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   return xs



def run_fit_a(x, w):
  n = x.shape[0]
  xs = np.zeros_like(x)

  for i in range(w // 2):
     xw = x[: i + w // 2 + 1]
     xw = xw - xw.mean()
     lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
     p0 = lg[0]
     p1 = lg[1]

     xw = xw - p0 * np.arange(xw.shape[0]) - p1

     dxw = xw[1:] - xw[:-1]
     lg = st.linregress(xw[:-1], dxw)[:]
     a = lg[0]
     b = lg[1]


     xs[i] = a

  for i in range(n - w // 2, n):
     xw = x[i - w // 2 + 1:]
     xw = xw - xw.mean()
     lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
     p0 = lg[0]
     p1 = lg[1]

     xw = xw - p0 * np.arange(xw.shape[0]) - p1

     dxw = xw[1:] - xw[:-1]
     lg = st.linregress(xw[:-1], dxw)[:]
     a = lg[0]
     b = lg[1]
     xs[i] = a

  for i in range(w // 2, n - w // 2):
     xw = x[i - w // 2 : i + w // 2 + 1]
     xw = xw - xw.mean()

     lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
     p0 = lg[0]
     p1 = lg[1]

     xw = xw - p0 * np.arange(xw.shape[0]) - p1


     dxw = xw[1:] - xw[:-1]
     lg = st.linregress(xw[:-1], dxw)[:]
     a = lg[0]
     b = lg[1]

     xs[i] = a
  return xs


def run_fit_a_ar1(x, w):
  n = x.shape[0]
  xs = np.zeros_like(x)

  for i in range(w // 2):
     xs[i] = np.nan

  for i in range(n - w // 2, n):
     xs[i] = np.nan

  for i in range(w // 2, n - w // 2):
     xw = x[i - w // 2 : i + w // 2 + 1]
     xw = xw - xw.mean()

     p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
     xw = xw - p0 * np.arange(xw.shape[0]) - p1


     dxw = xw[1:] - xw[:-1]

     xw = sm.add_constant(xw)
     model = sm.GLSAR(dxw, xw[:-1], rho=1)
     results = model.iterative_fit(maxiter=10)

     a = results.params[1]

     xs[i] = a
  return xs
