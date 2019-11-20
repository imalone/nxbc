#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys
import argparse

import numpy as np
#%matplotlib inline  
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib

from filter import *
from plotsupport import *
from smoothing import applyMINCSmooth
from splinesmooth3d import SplineSmooth3D
from skimage import filters, restoration
#import mba

FileType=argparse.FileType
parser = argparse.ArgumentParser(description='Test multilevel bias corrector.')
parser.add_argument('--infile','-i', metavar='INIMAGE',
                    help='input file', required=True)
parser.add_argument('--mask','-m', metavar='MASKIMAGE',
                    help='optional mask')
parser.add_argument('--outfile','-o', metavar='OUTIMAGE',
                    help='output file', required=True)
parser.add_argument('--bfield','-b', metavar='FIELDIMAGE',
                    help='optional output file')
parser.add_argument('--otsu', action='store_true',
                    help='If using mask image also use Otsu filter mask')
parser.add_argument('--pctrim','-p', action='store_true',
                    help='If using mask image also use 0.1%% bottom trim')
parser.add_argument('--stepsperlevel','-s', type=int,
                    default=5,
                    help='Steps per level')
parser.add_argument('--sigmafrac','-f', type=float,
                    default=5,
                    help='Sigma fraction')
parser.add_argument('--fwhm', type=float,
                    default=0.15,
                    help='FWHM for log histogram deconvolution')
parser.add_argument('--maxlevel','-l', type=int,
                    default=1,
                    help='Maximum level')
parser.add_argument('--sub','-r', type=int,
                    default=None,
                    help='sub sampling factor')
parser.add_argument('--expansion','-e', type=float,
                    default=1.0,
                    help='expansion factor for control grid')
parser.add_argument('--thr','-t', type=float,
                    default=1e-6,
                    help='stopping threshold to be used at each level')
parser.add_argument('--dist','-d', type=float,
                    default=150,
                    help='spline spacing (mm)')
parser.add_argument('--savehists', default=None, type=str,
                    help="directory name to save histogram files")
parser.add_argument('--saveplots', default=None, type=str,
                    help="directory name to save histogram plots")
parser.add_argument('--savefields', default=None, type=str,
                    help="directory name to save intermediate field estimates")
parser.add_argument('--accumulate', action='store_true',
                    help="use accumulated bias field fitting (N4 style)")


if False:
  args="-i fad-1015-1-143136_gw.nii.gz -m fad-1015-1-143136_gw_mask.nii.gz "+\
    "-p -f 10 -s20 -l4 "+\
    "-o testmultilevel/xfad-1015-itml-optrembmv-s20-f10-l4-bc.nii.gz "+\
    "-b testmultilevel/xfad-1015-itml-optrembmv-s20-f10-l4-bf.nii.gz " +\
    "-r 2"
  args=args.split(" ")
  args = parser.parse_args(args)
else:
  args = parser.parse_args()

infile = args.infile
outfile = args.outfile
outfieldfile = args.bfield
#FWHM=0.15
Z=0.01
maskfile = args.mask
withotsu = args.otsu
pctrim = args.pctrim
Nbins=256
steps=args.stepsperlevel
fwhmfrac = args.sigmafrac
subsamp = args.sub
expand = args.expansion
stopthr = args.thr

if expand < 1:
  print("Expansion factor must be >=1")
  sys.exit(0)

if (maskfile is None):
  withotsu = True

print("Running, input {}, output {}, mask {}".format(
  infile,
  outfile,
  maskfile))
print("Options Otsu {}, pc-trim {}, bfield {}".format(
  withotsu,
  pctrim,
  outfieldfile))
print("FWHM {} Z {:0.04f} nbins".format(None,Z,Nbins))

inimg = nib.load(infile)
inimgdata = inimg.get_fdata()

mask = np.ones(inimgdata.shape) > 0
if maskfile :
  inmask = nib.load(maskfile)
  mask = np.logical_and(inmask.get_fdata() > 0, inimgdata > 0)

if subsamp :
  for ax in range(mask.ndim):
    mask0 = np.zeros(mask.shape)
    subidx = [slice(None)]*mask.ndim
    els = range(0,mask0.shape[ax],subsamp)
    subidx[ax] = els
    mask0[tuple(subidx)] = 1
    mask = np.logical_and(mask, mask0)

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
datafill = np.zeros_like(datalog)


datalogmaskedcur = np.copy(datalogmasked)
eps=0.01
min_fill=0.5

# Descending FWHM scheme
levels=[ lvl for lvl in range(args.maxlevel) for _ in range(steps) ]
levelfwhm = args.fwhm / (np.arange(args.maxlevel) + 1)

splsm3d = SplineSmooth3D(datalog, nib.affines.voxel_sizes(inimg.affine),
                         args.dist, domainMethod="minc", mask=mask, Lambda=1.0/subsamp**3)

lastinterpbc = np.zeros(datalogmasked.shape[0])
datalogcur = np.copy(datalog)
nextlevel = 0
savehists = args.savehists
saveplots= args.saveplots
savefields=args.savefields
accumulate=args.accumulate

controlField=None

for N in range(len(levels)):
    if N%1 == 0 :
        print("{}/{}".format(N,len(levels)))
    if levels[N] < nextlevel:
      continue
    nextlevel = levels[N]
    #hist,histvaledge = np.histogram(datalogmaskedcur,Nbins)
    #histwidth = histvaledge[-1] - histvaledge[0]
    #histval = (histvaledge[0:-1] + histvaledge[1:])/2
    #histbinwidth = histwidth / (histval.shape[0]-1)
    #hist,histvaledge,histval,histbinwidth = distrib_histo(datalogmaskedcur, Nbins)
    hist,histvaledge,histval,histbinwidth = \
      distrib_kde(datalogmaskedcur, Nbins)
    #thisFWHM = optFWHM(hist,histbinwidth)
    #thisFWHM = optEntropyFWHM(hist, histbinwidth, histval, datalogmaskedcur, distrib="kde")
    thisFWHM = levelfwhm[levels[N]] # * math.sqrt(8*math.log(2))
    #thisSD = picksdremmeanvar(datalogcur, mask)
    #thisFWHM = thisSD * math.sqrt(8*math.log(2))
    thisSD = thisFWHM /  math.sqrt(8*math.log(2))
    #thisFWHM = thisFWHM / fwhmfrac
    print ("reduced sigma {} fwhm {}".format(thisSD, thisFWHM))
    mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)

    histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
    histfiltclip = np.clip(histfilt,0,None)

    if savehists:
      os.mkdir(savehists)
      np.save("{}/kdetracksteps-{:02d}".format(savehists,N),
              np.vstack((histval,hist)))
      #np.save("{}/kdetrackhist-{:02d}".format(savehists,N),datalogmaskedcur)

    uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
    datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
    if accumulate:
      logbc = datalogmaskedcur - datalogmaskedupd
    else:
      logbc = datalogmasked - datalogmaskedupd
    meanadj=True
    if meanadj:
      logbc = logbc - np.mean(logbc)
    usegausspde=True
    if saveplots:
      os.mkdir(saveplots)
      if usegausspde:
        updhist = kdepdf(histval, datalogmaskedupd, histbinwidth)
      else:
        updhist = kdepdf(histval, datalogmaskedupd, histbinwidth, kernelfntri)
        histmax = hist.max()
      plt.title("Step {}, level {}, FWHM {:0.3f}".format(N,levels[N],thisFWHM))
      plt.plot(histval,updhist/updhist.max()*histmax,color="OrangeRed")
      plt.plot(histval,histfiltclip/histfiltclip.max()*histmax,color="DarkOrange")
      plt.plot(histval,histfilt/histfilt.max()*histmax,color="LimeGreen")
      plt.plot(histval,hist,color="RoyalBlue")
      plt.savefig("{}/kdetracksteps-{:04d}.png".format(saveplots,N))
      plt.close()

    # Need masking!
    datafill[mask] = logbc
    splsm3d.fit(datafill, reportingLevel=1)
    logbcsmfull = splsm3d.predict()
    if savefields:
      os.mkdir(savefields)
      tmpnii = nib.Nifti1Image(datafill, inimg.affine, inimg.header)
      nib.save(tmpnii,"{}/in-{:02d}.nii.gz".format(savefields,N))
      tmpnii = nib.Nifti1Image(logbcsmfull, inimg.affine, inimg.header)
      nib.save(tmpnii,"{}/out-{:02d}.nii.gz".format(savefields,N))
    logbcsm = logbcsmfull[mask]

    if accumulate:
      logbcratio = logbcsm
    else:
      logbcratio = logbcsm - lastinterpbc
      lastinterpbc = logbcsm
    bcratio = np.exp(logbcratio)
    ratiomean = bcratio.mean()
    ratiosd = bcratio.std()
    conv = ratiosd / ratiomean
    print(conv,ratiosd,ratiomean)
    if accumulate:
      datalogmaskedcur = datalogmaskedcur - logbcsm
      if controlField is None:
        controlField  = splsm3d.P.copy()
      else:
        controlField += splsm3d.P
    else:
      datalogmaskedcur = datalogmasked - logbcsm
    datalogcur[mask] = datalogmaskedcur
    if (conv < stopthr):
      nextlevel = levels[N] + 1
print("\nComplete")

if accumulate:
  splsm3d.P = controlField

bfieldlog = splsm3d.predict()

bfield = np.exp(bfieldlog)
imgcorr = inimgdata / bfield

imgcorrnii = nib.Nifti1Image(imgcorr, inimg.affine, inimg.header)
nib.save(imgcorrnii,outfile)
imgbfnii = nib.Nifti1Image(bfield, inimg.affine, inimg.header)
nib.save(imgbfnii,outfieldfile)
