# NX Bias Correction

This module provides 3D bias correction for NIfTI (.nii/.nii.gz) images based on the N3 and N4 algorithms. It provides a simple command to apply N3 and N4, as well as a platform to further develop the algorithms.

Requires https://github.com/imalone/SplineSmooth3D

Straightforward N3 with a default Otsu mask:

nxbc -i image.nii.gz -o imagenorm.nii.gz

Recommended N3 using brain masking:

nxbc -a niftytools -i image.nii.gz -o imagenorm.nii.gz

This mode requires FSL to be installed, as well as [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) and [NiftySeg](https://github.com/KCL-BMEIS/NiftySeg).

Alternative modes for automated brain masking are -a flirt (requires FSL only) and -a ants (requires python packages antspyx and templateflow). These are less well tested and it is worth checking the mask generated (also possible for -a niftytools):

nxbc -a ants -i image.nii.gz -o imagenorm.nii.gz --outmask imagemask.nii.gz

## Options

nxbc is intended to allow experimentation with the N3/N4 algorithm class, which means there are a lot of options that may not be useful to the average user. Some of the more useful ones:

- -h/--help Show help
- -i/--infile Input image
- -a/--automask Brain masking method, options are flirt, niftytools, ants and
  niftytoolsreplicate (for replicating original results)
- -m/--mask Input mask image
- -o/--outfile Output image
- --outmask Output mask image
- --N4 use N4 mode
- -l/--maxlevel maximum level, in N4 mode the number of levels to use

Options that will be familiar to N3 and N4 users:
- --Lambda/-L spline smoothing lambda, default 1 as in MINC N3 version<=1.11
- -b/--bfield Output bias field image
- --fwhm Full Width Half Maximum of the log intensity histogram deconvolution
- --nbins Number of bins for histogram deconvolution
- -Z noise estimate for histogram deconvolution
- -r subsampling factor, default 2 (MINC -shrink option)
- -d/--dist Control point spacing, default 75mm

The default 75mm control point distance is intended to be more appropriate for 3T MP-RAGE images in line with [Boyes et al. 2008](https://pubmed.ncbi.nlm.nih.gov/18063391/), which suggests a distance in the range 50-100mm. Template mask registration prior to bias correction is also due to Boyes et al.

From version 1.11 to 1.12 MINC nu_correct changed from an interpretation of Lambda measured over the whole volume to one scaled by number of voxels. The default Lambda thus changed from 1.0 (as in the original N3 paper by Sled et al.) to 1.0e-7. For a common MPRAGE volume size of 240x256x208 voxels this would result in an effective total lambda of ~1.28, however this varies as the number of voxels included in bias correction is always less (often much less) than the total in the image. nxbc currently uses the MINC<=1.11 global lambda with a default of 1.0.
