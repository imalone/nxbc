#!/usr/bin/env python

# coding: utf-8

# In[1]:

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nibabel as nib

#from skimage import color, data, restoration, filters, exposure
from skimage import filters

import sys

if (len(sys.argv) < 5) :
  print ("wiener-histogram-check-img.py inimg.nii outplot FWHM Z [maskimg.nii] [maskmeth]")
  print ("\nOtsu mask by default, unless mask image supplied")
  print ("If mask image supplied then additional masking can be specified by otsu or pctrim (cut first 1%)")
  sys.exit(0)

# In[2]:


inname = sys.argv[1]
outname = sys.argv[2]
FWHM=float(sys.argv[3])
Z=float(sys.argv[4])
maskname = False
withotsu = True
pctrim = False
slax = 2
Nbins=256

try:
    if sys.argv[5] != "none" :
      maskname = sys.argv[5]
      withotsu = False
except IndexError:
    pass

try:
    if sys.argv[6] == "otsu" :
        withotsu = True
    elif sys.argv[6] == "pctrim" :
        pctrim = True
    else :
        print "Arg 5 not 'otsu or pctrim, ignored'"
except IndexError:
    pass

try:
    if sys.argv[7] == "0" :
      slax = 0
    elif sys.argv[7] == "1" :
      slax = 1
    elif sys.argv[7] == "2" :
      slax = 2
    else :
      print("Warning, axis option not 0, 1, 2")
except IndexError:
    pass

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


def rearrAx2(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[2]/2)
  # Earlier numpy.flip only one axis at a time
  return np.transpose(np.flip(np.flip(vol[:,:,sl],0),1))

def rearrAx1(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[1]/2)
  # Earlier numpy.flip only one axis at a time
  return np.transpose(np.flip(np.flip(vol[:,sl,:],0),1))

def rearrAx0(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[0]/2)
  # Earlier numpy.flip only one axis at a time
  return np.flip(np.flip(vol[sl,:,:],0),1)

if slax == 0 :
  rearrSel = rearrAx0
elif slax == 1 :
  rearrSel = rearrAx1
elif slax == 2 :
  rearrSel = rearrAx2


def imgplot(plotobj, data, vmin=None, vmax=None, cmap='gray'):
  if not vmin:
    vmin = np.percentile(data,2.5)
  if not vmax:
    vmax = np.percentile(data,97.5)
  plotobj.matshow(data,cmap=cmap,vmin=vmin,vmax=vmax)
  return plotobj.axis('off')



# In[9]:
inimg = nib.load(inname)
inimgdata = inimg.get_fdata()

mask = np.ones(inimgdata.shape) > 0
if maskname :
  inmask = nib.load(maskname)
  mask = np.logical_and(inmask.get_fdata() > 0, inimgdata > 0)

if withotsu :
  _thresh = filters.threshold_otsu(inimgdata[mask])
  mask = np.logical_and(inimgdata > _thresh, mask)

if pctrim :
  _hist, _edges = np.histogram(inimgdata[mask], bins=256)
  _cdf = np.cumsum(_hist) / float(np.sum(_hist))
  _pcthresh = 1e-3
  _pctrimthreshbin = np.searchsorted(_cdf, _pcthresh)
  _thresh = _edges[_pctrimthreshbin+1]
  mask = np.logical_and(inimgdata > _thresh, mask)

datamasked = inimgdata[mask]
datalog = np.copy(inimgdata)
datalog[mask] = np.log(datalog[mask])
datalog[np.logical_not(mask)] = 0
datalogmasked = datalog[mask]
(orighist, orighistedge) = np.histogram(datamasked,Nbins)
(hist,histvaledge) = np.histogram(datalogmasked,Nbins)

# For filter purposes histval (bin centre) is the better choice
histwidth = histvaledge[-1] - histvaledge[0]
histval = (histvaledge[0:-1] + histvaledge[1:])/2
histbinwidth = histwidth / (histval.shape[0]-1)
filt, filtx, filtmid, filtbins = symGaussFilt(FWHM, histbinwidth)


histfilt = wiener_filter_withpad(hist, filt, filtmid, Z)
histfiltclip = np.clip(histfilt,0,None)

uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, filt, hist)

datalogmaskednew = map_Eu_v(histval, uest, datalogmasked)
quantmappedu = map_quantu_v(histval, hist, histfiltclip, datalogmasked)

datanew = np.copy(datalog)
datanew[mask] = datalogmaskednew
datanew[mask] = np.exp(datanew[mask])

if(0) :
  datanewquant = np.copy(datalog)
  datanewquant[mask] = quantmappedu
  datanewquant[mask] = np.nan_to_num(np.exp(datanewquant[mask]))

with np.errstate(divide='ignore', invalid='ignore'):
  ratio = datanew / inimgdata
  #ratioq = datanewquant / inimgdata


fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(18, 16))
fig.suptitle("FWHM={:.3f} Z={:.4f} mask={} otsu={} pctrim={}".
               format(FWHM, Z, maskname, withotsu, pctrim),
             fontsize=18)
pos=(0,0)
imslice=rearrSel(inimgdata)
imgplot(ax[pos],imslice)
pos=(0,1)
imslice=rearrSel(datanew)
imgplot(ax[pos],imslice)
pos=(1,0)
ax[pos].plot(histval,hist)
ax[pos].plot(histval,histfilt)
ax[pos].plot(histval,histfiltclip)
pos=(1,1)
imslice=rearrSel(ratio)
imgplot(ax[pos],imslice, vmin=0.8, vmax=1.2)
if(0) :
  pos=(2,0)
  #imslice=rearrSel(datanewquant)
  imgplot(ax[pos],imslice)
  pos=(2,1)
  imslice=rearrSel(ratioq)
  imgplot(ax[pos],imslice, vmin=0.8, vmax=1.2)
plt.savefig("{}.png".format(outname))
plt.close()


if (0):
  plt.plot(histval,uest)
  plt.savefig("{}-uest-v.png".format(outname))
  plt.close()


if (0) :
  imgnew = nib.Nifti1Image(datanew, inimg.affine, inimg.header)
  nib.save(imgnew,'adni3-1022-sharphist.nii.gz')
  imgratio = nib.Nifti1Image(ratio, inimg.affine, inimg.header)
  nib.save(imgratio,'adni3-1022-sharphistratio.nii.gz')

print("Complete {}".format(outname))
