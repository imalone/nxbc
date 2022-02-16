#!/usr/bin/env python

from builtins import *

import sys
import argparse
import os
import errno
import textwrap

import numpy as np
#%matplotlib inline  
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib

from filter import *
from plotsupport import *
from smoothing import applyMINCSmooth
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, \
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


class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        paragraphs = text.split('\n')
        multiline_text = ""
        lead = ""
        for paragraph in paragraphs:
            paragraph = self._whitespace_matcher.sub(' ', paragraph).strip()
            if paragraph == "":
              multiline_text+="\n"
              lead=""
            else:
              formatted_paragraph = textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent)
              multiline_text = multiline_text + lead + formatted_paragraph
              lead=" "
        return multiline_text

FileType=argparse.FileType
parser = argparse.ArgumentParser(formatter_class=MultilineFormatter,
                                 description="""
NX bias correction


Inhomogeneity bias correction for MRI images in NIfTI (.nii/.nii.gz) format.
A reimplementation of N3 (Sled et al. 1998 https://doi.org/10.1109/42.668698) and N4 (Tustison et al. 2010 https://doi.org/10.1109/tmi.2010.2046908) supporting features of both.


Usage examples


In N3 mode with default Otsu mask, factor 2 subsampling and 75mm control point
spacing:

%(prog)s -r 2 -d 75 -i image.nii.gz -o imagenorm.nii.gz


In N3 mode as above with a user-supplied mask:

%(prog)s -r 2 -d 75 -i image.nii.gz -m mask.nii.gz -o imagenorm.nii.gz


In N4 mode with default Otsu mask, factor 2 subsampling, 4 fitting levels:

%(prog)s -r 2 --N4 -l4 -i image.nii.gz -o imagenorm.nii.gz


In N4 mode with default Otsu mask, factor 2 subsampling, 4 fitting levels:

%(prog)s -r 2 --N4 -l4 -i image.nii.gz -m mask.nii.gz -o imagenorm.nii.gz
""")
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
                    default=1000,
                    help='Steps per level')
parser.add_argument('--fwhm', type=float,
                    default=0.05,
                    help='FWHM for log histogram deconvolution')
parser.add_argument('-Z', type=float,
                    default=0.01,
                    help='Noise estimate for Wiener deconvolution')
parser.add_argument('--binEndLimits', action='store_true',
                    help="Place end bin edges on data limits, rather"
                    " than bin centres on limits.")
parser.add_argument('--maxlevel','-l', type=int,
                    default=1,
                    help='Maximum level. Fitting is either repeated for each level at FWHM=(starting FWHM)/level or with a subdivided mesh (see --subdivide)')
parser.add_argument('--sub','-r', type=int,
                    default=None,
                    help='sub sampling factor')
parser.add_argument('--thr','-t', type=float,
                    default=1e-4,
                    help='stopping threshold to be used at each level')
parser.add_argument('--dist','-d', type=float,
                    default=150,
                    help='spline spacing (mm)')
parser.add_argument('--ITKspacing','-I', action='store_true',
                    help='Use ITK spacing, adjust so single interval over '
                    +'each image dimension.')
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
parser.add_argument('--N4', action='store_true',
                    help='Use N4 features: ITKspacing, subdivide, unregularized, accumulate')
parser.add_argument('--costDerivative', type=int, default=2,
                    help="derivative order for cost function")
parser.add_argument('--reduceFOV', action='store_true',
                    help="Reduce regularization FOV to bounding"+\
                    "box of mask/above threshold region")
parser.add_argument('--kern', choices=['tri','gauss'],
                    help="Kernel function to use (tri => Parzen,"+\
                    "gauss => gauss KDE)", default='tri')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile
outfieldfile = args.bfield
Z=args.Z
bcl=not args.binEndLimits
maskfile = args.mask
withotsu = args.otsu
pctrim = args.pctrim
reduceFOV = args.reduceFOV
Nbins=256
steps=args.stepsperlevel
subsamp = args.sub
stopthr = args.thr

savehists = args.savehists
saveplots= args.saveplots
savefields=args.savefields
accumulate=args.accumulate
subdivide=args.subdivide
ITKspacing=args.ITKspacing
unregularized=args.unregularized
maxlevel=args.maxlevel

if args.N4:
  ITKspacing=True
  subdivide=True
  unregularized=True
  accumulate=True
  if maxlevel == 1:
    print("--N4 was specified, but maxlevel (-l) is 1, this may be a mistake")

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
print("FWHM {} Z {:0.04f} nbins {}".format(args.fwhm,Z,Nbins))

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
spacing=args.dist


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

if ITKspacing:
    spacing = 1
    dataSubVoxSize = 1 / (np.array(dataSub.shape) -1)
    dataVoxSize = dataSubVoxSize / subsamp

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

bboxFOV=None
if reduceFOV:
  allAx = range(3)
  bboxFOV=[]
  for ax in allAx:
    otherAx = np.delete(allAx,ax)
    alongAx=np.any(mask, axis=tuple(otherAx))
    ends=alongAx.nonzero()[0][[0,-1]]
    bboxFOV.append(ends*dataSubVoxSize[ax])
    inds=np.arange(ends[0],ends[1]+1)
    dataSub=dataSub.take(inds,ax)
    mask=mask.take(inds,ax)

datamasked = dataSub[mask]
# Since assigning into it we need to make sure float
# beforehand, otherwise assigning into int array will
# cause a cast
datalog = dataSub.astype(np.float32)
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
levels=[ lvl for lvl in range(maxlevel) for _ in range(steps) ]
# At some point will have to generalise into fwhm and subdivision
# level scheme, at the moment it's either or:
if not subdivide:
  levelfwhm = args.fwhm / (np.arange(maxlevel) + 1)
else:
  levelfwhm = args.fwhm * np.ones(maxlevel)

if unregularized:
  splsm3d = SplineSmooth3DUnregularized(datalog, dataSubVoxSize,
                                        spacing, domainMethod="minc",
                                        mask=mask)
else:
  if subsamp:
    try:
      effLambda=args.Lambda / float(subsamp)**3
    except TypeError:
      effLambda = {d:l/subsamp**3 for d,l in args.Lambda.items()}
  else:
    effLambda = args.Lambda
  splsm3d = SplineSmooth3D(datalog, dataSubVoxSize,
                           spacing, domainMethod="minc", mask=mask,
                           Lambda=effLambda,
                           costDerivative=args.costDerivative)


# Prediction interpolator, shift knot locations to match the
# reduced FOV region.
if bboxFOV is None:
  predictor = SplineSmooth3D(inimgdata, dataVoxSize,
                           spacing, knts=splsm3d.kntsArr, dofit=False)
else:
  predKnts = [ (knts[0],knts[1]+limits[0]) for
               knts, limits in zip(splsm3d.kntsArr, bboxFOV) ]
  predictor = SplineSmooth3D(inimgdata, dataVoxSize,
                             spacing, knts=predKnts, dofit=False)
lastinterpbc = np.zeros(datalogmasked.shape[0])
datalogcur = np.copy(datalog)
nextlevel = 0

controlField=None

chosenkernelfn = kernelfntri
if args.kern == "gauss":
  chosenkernelfn = kernelfngauss

for N in range(len(levels)):
    if N%1 == 0 :
        print("{}/{}".format(N,len(levels)))
    if levels[N] < nextlevel:
      continue
    nextlevel = levels[N]
    hist,histvaledge,histval,histbinwidth = \
      distrib_kde(datalogmaskedcur, Nbins, kernfn=chosenkernelfn,
                  binCentreLimits=bcl)
    #thisFWHM = optFWHM(hist,histbinwidth)
    #thisFWHM = optEntropyFWHM(hist, histbinwidth, histval, datalogmaskedcur, distrib="kde")
    thisFWHM = levelfwhm[levels[N]] # * math.sqrt(8*math.log(2))
    tuneSD=False
    if tuneSD:
      retry=True
      sampsize=300
      while retry:
        try:
          thisSD = picksdremmeanvar(datalogcur, mask>0, sampsize=300) / 10
          retry=False
        except ValueError:
          sampsize=1000
          print("Retrying picksdremmeanvar")
      thisFWHM = thisSD * math.sqrt(8*math.log(2))
    thisSD = thisFWHM /  math.sqrt(8*math.log(2))
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
    if subdivide and (N+1)<len(levels) and N%steps == 0:
      print ("subdividing")
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
#nib.save(imgbfnii,outfieldfile) #.astype(np.float32)
imgcorrnii = nib.Nifti1Image(imgcorr.astype(np.float32), inimg.affine) #, inimg.header)
nib.save(imgcorrnii,outfile)
if outfieldfile is not None:
  imgbfnii = nib.Nifti1Image(bfield.astype(np.float32), inimg.affine) #, inimg.header)
  nib.save(imgbfnii,outfieldfile)
