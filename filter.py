from skimage import filters
import numpy as np


def symGaussFilt (filtsigma, filtbinwidth):
  filtsigmaNbin = filtsigma / filtbinwidth
  filtbins = int(int(np.ceil(filtsigmaNbin)*8)/2*2+1)
  filtmid = filtbins/2 # filtbins is odd...
  filtz = np.arange(-filtmid,filtmid+1) / filtsigmaNbin
  filtnorm = np.sqrt(2 * filtsigmaNbin ** 2 *np.pi)
  filt = np.exp ( -filtz**2 / 2) / filtnorm
  filtx = np.arange(-filtmid,filtmid+1) * filtbinwidth
  return filt, filtx, filtmid, filtbins


def wiener_filter_withpad(signal, filt, filtmid, Z):
  # Apply wiener deconvolution filter to array
  # allows use of a non-causal filter (e.g. spatial symmetric)
  # by specifying the filter mid-point/zero point
  # also sufficiently pads the signal to be deconvolved to
  # avoid periodicity issues
  filtbins = len(filt)
  padval = 0 # in deconvolution doesn't make much sense to pad non-zero
  filtpadval = 0 # separate from signal pad value
  # We're deconvolving, so the shift the filter introduces is added
  # in the padding, don't need to remove it from the start afterwards
  # as deconvolution shifts the signal back
  # The entirety of the padding is removed from the end.
  sigpad = np.pad(signal, ((filtbins-filtmid),filtmid),
    'constant', constant_values=(padval, padval))
  filtpad = np.pad(filt, (0,sigpad.shape[0]-filt.shape[0]),
    'constant', constant_values=(0, 0))

  F = np.fft.fft(sigpad)
  H = np.fft.fft(filtpad)
  H2 = np.conj(H)*H
  G = (1/H) * (H2 / (H2+Z))
  Ffilt = F * G
  sigfilt = np.real(np.fft.ifft(Ffilt))
  #plt.subplot(2,1,1)
  #sigrange=np.arange(0,(histbinwidth*(sigpad.shape[0])),histbinwidth)
  #plt.plot(sigrange,sigpad)
  #plt.subplot(2,1,2)
  #plt.plot(filtpad)
  return sigfilt[slice(signal.shape[0])]


def Eu_v (distu, valsu, filt, distv) :
  u1 = distu * valsu
  conv1 = np.convolve(u1,filt,mode="same")
  conv2 = np.convolve(distu,filt,mode="same")
  uest = conv1/conv2
  #plt.subplot(4,1,1)
  #plt.plot(u1)
  #plt.subplot(4,1,2)
  #plt.plot(conv1)
  #plt.subplot(4,1,3)
  #plt.plot(conv2)
  #plt.subplot(4,1,4)
  #plt.plot(uest)
  return uest, u1, conv1, conv2


def map_Eu_v(histval,uest, v):
  # histval are bin centers, this is because uest
  # is estimated by convolution, so need to use E(v)
  # and E(u) values for each bin.
  # Could just use np.interp here, however it will result
  # in clipping at the max and min values, which loses a
  # bin either end, so a bit more elegant to do it ourselves
  vwidths = histval[1:] - histval[:-1]
  # Clipping means we project linearly past the ends of the
  # array based on the first and last intervals
  vbins = np.clip(np.searchsorted(histval,v)-1,0,histval.shape[0]-2)
  vbindist = (v - histval[vbins]) / vwidths[vbins]
  uwidths = uest[1:] - uest[:-1]
  u = uest[vbins] + uwidths[vbins] * vbindist[vbins]
  return u


def map_quantu_v(binvals, distv, distu, v):
  # Use the idea the voxel value quantiles should remain the
  # same transferring between the distributions
  # Doesn't work properly, remapped histogram comes out very spiky
  # and CDF closer to the original than that of the filtered
  # distribution. If working correctly then the resulting CDF
  # should resembled that of the filtered distribution
  # Also, bin values are the bin centres, so there's a slight
  # offset issue to deal with, but minor so long as the bins
  # are narrow. Not entirely sure using correct binwidths for
  # the interpolation (might be shifted by one)

  # cumv are to the bin upper edge. fudge a bit and approximate
  # the middle value
  cumv = np.cumsum(distv)/float(np.sum(distv))
  cumv = np.hstack((0,cumv))
  cumv = (cumv[0:-1] + cumv[1:])/2
  cumvdiffs = cumv[1:] - cumv[:-1]

  vwidths = binvals[1:] - binvals[:-1]
  vbin = np.clip(np.searchsorted(binvals,v)-1,0,cumv.shape[0]-2)
  vbindist = (v - binvals[vbin])/ vwidths[vbin]
  
  vpercentile = cumv[vbin] + cumvdiffs[vbin] * vbindist

  # cumu, now dealing with the quantiles and project
  # back to histval
  cumu = np.cumsum(distu)/float(np.sum(distu))
  cumu = np.hstack((0,cumv))
  cumu = (cumu[0:-1] + cumu[1:])/2
  cumudiffs = cumu[1:] - cumu[:-1]

  with np.errstate(divide='ignore'):
    cumuquantscale = 1 / cumudiffs
  # This is wrong: it's saying that if the percentile is in the
  # next higher bin it gets stuck in the current one if the adjacent
  # bin is the same, should actually be a leap. Could be one cause of
  # spicky-ness
  cumuquantscale[np.isinf(cumuquantscale)] = 0
  quantbin = np.clip(np.searchsorted(cumu,vpercentile)-1,0,cumu.shape[0]-2)
  quantbindist = (vpercentile - cumu[quantbin]) * \
    cumuquantscale[quantbin]
  # vwidths are intervals for values bins represent for both
  # original and filtered histogram
  quantmappedu = binvals[quantbin] + \
    vwidths[quantbin] * quantbindist[quantbin]
  return quantmappedu
