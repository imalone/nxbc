#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys
import argparse
import os
import errno

import numpy as np
#%matplotlib inline  
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib

from filter import *
from plotsupport import *
from smoothing import applyMINCSmooth
from splinesmooth3d.splinesmooth3d import SplineSmooth3D, \
  SplineSmooth3DUnregularized
from skimage import filters, restoration
#import mba

def lambdaCheck(lambdaStr):
    try:
        Lambda=float(lambdaStr)
    except ValueError:
        try:
            lambdaParts = lambdaStr.split(',')
            lambdaParts = [float(x) for x in lambdaParts]
            lenParts = len(lambdaParts)
            if (lenParts>3):
                raise ValueError("Must be less than 4 parts to lambda")
            Lambda = {deriv:lam for (deriv,lam) in
                zip(range(lenParts),lambdaParts)}
        except AttributeError:
            raise ValueError("lambda must be floating point or comma-separated string of floating points")
    return Lambda


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
                    help='Maximum level. Fitting is either repeated for each level at FWHM=(starting FWHM)/level or with a subdivided mesh (see --subdivide)')
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
parser.add_argument('--Lambda', '-L', default=1.0, type=lambdaCheck,
                    help="spline smoothing lambda (image level)")
parser.add_argument('--subdivide', action='store_true',
                    help="subdivide mesh at each level")
parser.add_argument('--unregularized', action='store_true',
                    help="subdivide mesh at each level")
parser.add_argument('--costDerivative', type=int, default=2,
                    help="derivative order for cost function")


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

savehists = args.savehists
saveplots= args.saveplots
savefields=args.savefields
accumulate=args.accumulate
subdivide=args.subdivide

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

dataSub = inimgdata
dataVoxSize = nib.affines.voxel_sizes(inimg.affine)
affineSub = np.copy(inimg.affine)
dataSubVoxSize = dataVoxSize


if subsamp :
  # Can't use offset != 0 yet, as the spline smoother takes voxel positions
  # to start from 0, meaning some small interface changes to:
  # 1. control initial voxel offsets
  # 2. keep domain consistent allowing same parameters to be used to
  #    supersample from the spline model.
  offset = 0 # subsamp // 2
  dataSub = dataSub[offset::subsamp,offset::subsamp,offset::subsamp]
  mask = mask[offset::subsamp,offset::subsamp,offset::subsamp]
  affineSub[0:3,3] = affineSub[0:3,0:3].sum(1) * offset + affineSub[0:3,3]
  affineSub[0:3,0:3] *= subsamp
  dataSubVoxSize = nib.affines.voxel_sizes(affineSub)


if withotsu :
  _thresh = filters.threshold_otsu(dataSub[mask])
  mask = np.logical_and(dataSub > _thresh, mask)

if pctrim :
  _hist, _edges = np.histogram(dataSub[mask], bins=256)
  _cdf = np.cumsum(_hist) / float(np.sum(_hist))
  _pcthresh = 1e-3
  _pctrimthreshbin = np.searchsorted(_cdf, _pcthresh)
  _thresh = _edges[_pctrimthreshbin+1]
  mask = np.logical_and(dataSub > _thresh, mask)


datamasked = dataSub[mask]
datalog = np.copy(dataSub)
datalog[mask] = np.log(datalog[mask])
datalog[np.logical_not(mask)] = 0
datalogmasked = datalog[mask]
datafill = np.zeros_like(datalog)

if savefields:
  try:
    os.mkdir(savefields)
  except OSError as e:
    if e.errno == errno.EEXIST:
      pass
    else:
      raise
  tmpnii = nib.Nifti1Image(mask*1, affineSub)
  nib.save(tmpnii,"{}/mask.nii.gz".format(savefields))
  if accumulate:
    accumulateSaveField = np.zeros(datalog.shape)

datalogmaskedcur = np.copy(datalogmasked)
eps=0.01
min_fill=0.5

# Descending FWHM scheme
levels=[ lvl for lvl in range(args.maxlevel) for _ in range(steps) ]
# At some point will have to generalise into fwhm and subdivision
# level scheme, at the moment it's either or:
if not subdivide:
  levelfwhm = args.fwhm / (np.arange(args.maxlevel) + 1)
else:
  levelfwhm = args.fwhm * np.ones(args.maxlevel)

if args.unregularized:
  splsm3d = SplineSmooth3DUnregularized(datalog, dataSubVoxSize,
                                        args.dist, domainMethod="minc",
                                        mask=mask)
else:
  try:
    effLambda=args.Lambda / subsamp**3
  except TypeError:
    effLambda = {d:l/subsamp**3 for d,l in args.Lambda.items()}
  splsm3d = SplineSmooth3D(datalog, dataSubVoxSize,
                           args.dist, domainMethod="minc", mask=mask,
                           Lambda=effLambda,
                           costDerivative=args.costDerivative)
predictor = SplineSmooth3D(inimgdata, dataVoxSize,
                           args.dist, domainMethod="minc", dofit=False)

lastinterpbc = np.zeros(datalogmasked.shape[0])
datalogcur = np.copy(datalog)
nextlevel = 0

controlField=None

for N in range(len(levels)):
    if N%1 == 0 :
        print("{}/{}".format(N,len(levels)))
    if levels[N] < nextlevel:
      continue
    nextlevel = levels[N]
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
      try:
        os.mkdir(savehists)
      except OSError as e:
        if e.errno == errno.EEXIST:
          pass
        else:
          raise
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
      try:
        os.mkdir(saveplots)
      except OSError as e:
        if e.errno == errno.EEXIST:
          pass
        else:
          raise
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
      try:
        os.mkdir(savefields)
      except OSError as e:
        if e.errno == errno.EEXIST:
          pass
        else:
          raise
      tmpnii = nib.Nifti1Image(datafill, affineSub)
      nib.save(tmpnii,"{}/in-{:02d}.nii.gz".format(savefields,N))
      if accumulate:
        accumulateSaveField += logbcsmfull
      else:
        accumulateSaveField = logbcsmfull
      tmpnii = nib.Nifti1Image(accumulateSaveField, affineSub)
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
    if subdivide and (N+1)<len(levels) and (N+1)%steps == 0:
      # Applies to both cumulative and normal iterative
      # mode, in normal iterative mode we're just upgrading
      # to a finer mesh for the following updates.
      # In cumulative mode we first get the current cumulative
      # estimate before refining.
      if accumulate:
        splsm3d.P = controlField
      splsm3d = splsm3d.promote()
      predictor = predictor.promote()
      controlField = splsm3d.P

if accumulate:
  splsm3d.P = controlField
# Back from subsampled space to full size:
predictor.P = splsm3d.P
bfieldlog = predictor.predict()

bfield = np.exp(bfieldlog)
imgcorr = inimgdata / bfield
#imgcorr = dataSub / bfield

#imgcorrnii = nib.Nifti1Image(imgcorr, affineSub, inimg.header)
#nib.save(imgcorrnii,outfile)
#imgbfnii = nib.Nifti1Image(bfield, affineSub, inimg.header)
#nib.save(imgbfnii,outfieldfile)
imgcorrnii = nib.Nifti1Image(imgcorr, inimg.affine) #, inimg.header)
nib.save(imgcorrnii,outfile)
imgbfnii = nib.Nifti1Image(bfield, inimg.affine) #, inimg.header)
nib.save(imgbfnii,outfieldfile)
