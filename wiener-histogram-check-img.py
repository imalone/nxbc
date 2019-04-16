#!/usr/bin/env python

# coding: utf-8

# In[1]:

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nibabel as nib

from filter import *
from plotsupport import *

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

rearrSel=getrarrAxSel(slax)



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
