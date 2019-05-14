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
from skimage import filters
import mba

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
                    help='Steps per level')
parser.add_argument('--maxlevel','-l', type=int,
                    default=4,
                    help='Maximum level')


#args="-i fad-1015-1-143136_gw.nii.gz -m fad-1015-1-143136_gw_mask.nii.gz "+\
#  "-p -f 10 -s20 -l4 "+\
#  "-o testmultilevel/fad-1015-itml-optrembmv-s20-f10-l4-bc.nii.gz "+\
#  "-b testmultilevel/fad-1015-itml-optrembmv-s20-f10-l4-bf.nii.gz"
#args=args.split(" ")
#args = parser.parse_args(args)
args = parser.parse_args()

infile = args.infile
outfile = args.outfile
outfieldfile = args.bfield
#FWHM=0.15
Z=0.01
maskfile = args.mask
withotsu = args.otsu
pctrim = args.pctrim
slax = 2
Nbins=256
steps=args.stepsperlevel
fwhmfrac = args.sigmafrac


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

voxgrid3 = np.argwhere(mask)
voxfullgrid3 = np.argwhere(mask==mask)

datalogmaskedcur = np.copy(datalogmasked)
grid = 2
maxlvl = 4
eps=0.01

levels = [2] * steps + [3] * steps + [4] * steps + [5] * steps
levels = []
[ levels.extend([x] * steps) for x in range(2,args.maxlevel+1) ]
#levels = [2] * 20 + [3] * 20 + [4] * 20
#filtw = [0.15] * 20 + [0.15] * 20 + [0.1] * 20 + [0.05] * 20



lastinterpbc = np.zeros(datalogmasked.shape[0])
datalogcur = np.copy(datalog)
for N in range(len(levels)):
    if N%1 == 0 :
        print("{}/{}".format(N,len(levels)))
    #hist,histvaledge = np.histogram(datalogmaskedcur,Nbins)
    #histwidth = histvaledge[-1] - histvaledge[0]
    #histval = (histvaledge[0:-1] + histvaledge[1:])/2
    #histbinwidth = histwidth / (histval.shape[0]-1)
    #hist,histvaledge,histval,histbinwidth = distrib_histo(datalogmaskedcur, Nbins)
    hist,histvaledge,histval,histbinwidth = distrib_kde(datalogmaskedcur, Nbins)
    #thisFWHM = optFWHM(hist,histbinwidth)
    thisFWHM = optEntropyFWHM(hist, histbinwidth, histval, datalogmaskedcur, distrib="kde")
    datalogcur[mask] = datalogmaskedcur
    #thisSD = picksdremmeanvar(datalogcur, mask)
    #thisFWHM = thisSD * math.sqrt(8*math.log(2))
    thisSD = thisFWHM /  math.sqrt(8*math.log(2))
    thisFWHM = thisFWHM / fwhmfrac
    print ("reduced sigma {} fwhm {}".format(thisSD, thisFWHM))
    mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisFWHM, histbinwidth)
    histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
    histfiltclip = np.clip(histfilt,0,None)
    plt.title("Step {}, level {}, FWHM {:0.3f}".format(N,levels[N],thisFWHM))
    np.save("outnpyent/kdetracksteps-{:02d}".format(N),np.vstack((histval,hist)))
    #np.save("outnpyent/kdetrackhist-{:02d}".format(N),datalogmaskedcur)
    uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
    datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
    logbc = datalogmasked - datalogmaskedupd
    logbc = logbc - np.mean(logbc)
    updhist = kdepdf(histval, datalogmaskedupd, histbinwidth)
    plt.plot(histval,updhist,color="OrangeRed")
    plt.plot(histval,histfiltclip,color="DarkOrange")
    plt.plot(histval,histfilt,color="LimeGreen")
    plt.plot(histval,hist,color="RoyalBlue")
    plt.savefig("outpngent/kdetracksteps-{:02d}.png".format(N))
    plt.close()
    interpbc = mba.mba3([-eps]*3, [x + eps for x in inimgdata.shape], [grid]*3,
              voxgrid3.tolist(),
              logbc, max_levels=levels[N])
    logbcsm=interpbc(voxgrid3.tolist())
    logbcratio = logbcsm - lastinterpbc
    lastinterpbc = logbcsm
    bcratio = np.exp(logbcratio)
    ratiomean = bcratio.mean()
    ratiosd = bcratio.std()
    conv = ratiosd / ratiomean
    print(conv,ratiosd,ratiomean)
    datalogmaskedcur = datalogmasked - logbcsm
print("\nComplete")

bfieldlog = np.reshape(interpbc(voxfullgrid3.tolist()),inimgdata.shape)
bfield = np.exp(bfieldlog)
imgcorr = inimgdata / bfield

imgcorrnii = nib.Nifti1Image(imgcorr, inimg.affine, inimg.header)
nib.save(imgcorrnii,outfile)
imgbfnii = nib.Nifti1Image(bfield, inimg.affine, inimg.header)
nib.save(imgbfnii,outfieldfile)
