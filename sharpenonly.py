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


def quick_kde(data, Nbins, kernfn=kernelfntri):
  dmin=0.493135 -0.0173985
  dmax=7.417714 +0.0173985
  histvaledge = np.linspace(dmin,dmax,
  #histvaledge = np.linspace(data.min(),data.max(),
                            num=Nbins+1)
  histwidth = histvaledge[-1] - histvaledge[0]
  histval = (histvaledge[0:-1] + histvaledge[1:])/2
  histbinwidth = histwidth / (histval.shape[0]-1)
  hist=[]
  for ii in range(Nbins):
    cdist = np.abs(histval[ii] - data)/histbinwidth
    cweight = np.clip(1-cdist,0,None)
    hist.append(cweight.sum())
  return hist,histvaledge,histval,histbinwidth


def quick_kde2(data, Nbins, kernfn=kernelfntri):
  dmin=0.493135 -0.0173985
  dmax=7.417714 +0.0173985
  histvaledge = np.linspace(dmin,dmax,
  #histvaledge = np.linspace(data.min(),data.max(),
      num=Nbins+1)
  histwidth = histvaledge[-1] - histvaledge[0]
  histval = (histvaledge[0:-1] + histvaledge[1:])/2
  histbinwidth = histwidth / (histval.shape[0]-1)
  hist=[0]*Nbins
  data = data[data>=histval[0]]
  data = data[data<=histval[Nbins-1]]
  locs = ((data - histval[0])/histbinwidth).astype(np.float32)
  histval = (histval) + histbinwidth
  inds = np.floor(locs).astype(np.int)
  offs = locs - inds
  for ii in range(offs.size):
    off=offs[ii]
    ind=inds[ii]
    if(off==0):
      hist[ind]+=1
    elif (off>0 and ind <= Nbins-2):
      hist[ind] += 1-off
      hist[ind+1] += off
    elif (ind>=1):
      hist[ind] += 1+off
      hist[ind+1] -= off
  return hist,histvaledge,histval,histbinwidth


FileType=argparse.FileType
parser = argparse.ArgumentParser(description='Test multilevel bias corrector.')
parser.add_argument('--infile','-i', metavar='INIMAGE',
                    help='input file', required=True)
parser.add_argument('--mask','-m', metavar='MASKIMAGE',
                    help='optional mask')
parser.add_argument('--outbase','-o', metavar='OUTBASE',
                    help='output file base name', required=True)
parser.add_argument('--fwhm', type=float,
                    default=0.15,
                    help='FWHM for log histogram deconvolution')
parser.add_argument('--nbins', type=int,
                    default=256,
                    help='FWHM for log histogram deconvolution')
parser.add_argument('--no-log',
                    action='store_true',
                    help='Use log values')
parser.add_argument('--binCentreLimits', action='store_true',
                    help="Place end bin centres on data limits, rather"
                    " than bin edges on limits.")
parser.add_argument('--binPastLimits', action='store_true',
                    help="Place bin edges exactly one bin width past"
                    " data limits.")

if False:
  argstr="-i fad-1015-1-143136_gw.nii.gz -m fad-1015-1-143136_gw_mask.nii.gz "+\
    "-p -f 10 -s20 -l4 "+\
    "-o testmultilevel/xfad-1015-itml-optrembmv-s20-f10-l4-bc.nii.gz "+\
    "-b testmultilevel/xfad-1015-itml-optrembmv-s20-f10-l4-bf.nii.gz " +\
    "-r 2"
  args=argstr.split(" ")
  args = parser.parse_args(args)
else:
  args = parser.parse_args()

infile = args.infile
outbase = args.outbase
#FWHM=0.15
Z=0.01
maskfile = args.mask
Nbins=args.nbins

print("Running, input {}, output {}, mask {}".format(
  infile,
  outbase,
  maskfile))
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



datamasked = dataSub[mask]

datalog = np.copy(dataSub)
if not args.no_log:
  datalog[mask] = np.log(datalog[mask])
datalog[np.logical_not(mask)] = 0
datalogmasked = datalog[mask]
datafill = np.zeros_like(datalog)


datalogmaskedcur = np.copy(datalogmasked)
eps=0.01
min_fill=0.5
levelfwhm = args.fwhm

# Would like to re-use the input arguments easily, but may eventually
# also wish to compare alternatives.
def set_distrib_kde(datalogmaskedcur, Nbins, kernfn=kernelfntri,
      binCentreLimits=args.binCentreLimits,
      binPastLimits=args.binPastLimits):
  return distrib_kde(datalogmaskedcur, Nbins, kernfn=kernfn,
                     binCentreLimits=binCentreLimits,
                     binPastLimits=binPastLimits)

datalogcur = np.copy(datalog)
hist,histvaledge,histval,histbinwidth = \
    set_distrib_kde(datalogmaskedcur, Nbins)
    #quick_kde(datalogmaskedcur, Nbins, kernfn=kernelfntri)
    #quick_kde2(datalogmaskedcur, Nbins, kernfn=kernelfntri)
#thisFWHM = optFWHM(hist,histbinwidth)
#thisFWHM = optEntropyFWHM(hist, histbinwidth, histval, datalogmaskedcur, distrib="kde")
histsave = np.vstack((histval,hist))
np.savetxt(outbase+"-kde.csv",histsave.T)
thisFWHM = levelfwhm # * math.sqrt(8*math.log(2))
#thisSD = picksdremmeanvar(datalogcur, mask)
#thisFWHM = thisSD * math.sqrt(8*math.log(2))
thisSD = thisFWHM /  math.sqrt(8*math.log(2))
#thisFWHM = thisFWHM / fwhmfrac
print ("reduced sigma {} fwhm {}".format(thisSD, thisFWHM))
mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)

histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
histfiltclip = np.clip(histfilt,0,None)
uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
datalogcur[mask] = datalogmaskedupd

uestsave = np.vstack((histval,uest))
np.savetxt(outbase+"-uest.csv",uestsave.T)
filtsave = np.vstack((histval,histfiltclip))
np.savetxt(outbase+"-kdefilt.csv",filtsave.T)

histnew,histvaledgenew,histvalnew,histbinwidthnew = \
    set_distrib_kde(datalogmaskedupd, Nbins)
histsharpsave = np.vstack((histvalnew,histnew))
np.savetxt(outbase+"-kdesharp.csv",histsharpsave.T)

imgcorrnii = nib.Nifti1Image(datalogcur.astype(np.float32), inimg.affine) #, inimg.header)
nib.save(imgcorrnii,outbase+".nii.gz")
