#!/usr/bin/env python

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

args = parser.parse_args()

infile = args.infile
outfile = args.outfile
outfieldfile = args.bfield
FWHM=0.15
Z=0.01
maskfile = args.mask
withotsu = args.otsu
pctrim = args.pctrim
slax = 2
Nbins=256

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
print("FWHM {:0.04f} Z {:0.04f} nbins".format(FWHM,Z,Nbins))

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
levels = [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20
filtw = [0.15] * 20 + [0.15] * 20 + [0.1] * 20 + [0.05] * 20
for N in range(len(levels)):
    if N%1 == 0 :
        print("{}/{}".format(N,len(levels)),end='\r')
    hist,histvaledge = np.histogram(datalogmaskedcur,Nbins)
    histwidth = histvaledge[-1] - histvaledge[0]
    histval = (histvaledge[0:-1] + histvaledge[1:])/2
    histbinwidth = histwidth / (histval.shape[0]-1)
    mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(filtw[N], histbinwidth)
    histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
    histfiltclip = np.clip(histfilt,0,None)
    uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
    datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
    logbc = datalogmasked - datalogmaskedupd
    logbc = logbc - np.mean(logbc)
    interpbc = mba.mba3([-eps]*3, [x + eps for x in inimgdata.shape], [grid]*3,
              voxgrid3.tolist(),
              logbc, max_levels=levels[N])
    logbcsm=interpbc(voxgrid3.tolist())
    datalogmaskedcur = datalogmasked - logbcsm
print("\nComplete")

bfieldlog = np.reshape(interpbc(voxfullgrid3.tolist()),inimgdata.shape)
bfield = np.exp(bfieldlog)
imgcorr = inimgdata / bfield

imgcorrnii = nib.Nifti1Image(imgcorr, inimg.affine, inimg.header)
nib.save(imgcorrnii,outfile)
imgbfnii = nib.Nifti1Image(bfield, inimg.affine, inimg.header)
nib.save(imgbfnii,outfieldfile)
