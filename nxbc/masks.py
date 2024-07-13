import os
import os.path
import subprocess 
import shutil 
import tempfile 

def maskGenFlirt(inImg, outImg):
    """Derive mask from registration of MNI152 using FSL tools

    Requires FSL installed and available on system path with FSLDIR
    environment variable set.

    Args:
        inImg -- input image path (nifti format)
        outImg -- output image path (nifti .nii.gz)

    Returns:
        
    """
    # Exists to allow use without niftyreg+niftyseg tools
    # Example: 
    # flirt -omat MNI152_T1_1mm_aff.txt -in image.nii.gz
    #    -ref MNI152_T1_1mm.nii.gz -usesqform
    # convert_xfm -inverse MNI152_T1_1mm_aff.txt -omat MNI152_T1_1mm_inv.txt
    # flirt -init MNI152_T1_1mm_inv.txt -applyxfm
    #    -in MNI152_T1_1mm_brain_mask_dil.nii.gz -interp nearestneighbour
    #    -ref image.nii.gz
    #    -out MNI152_T1_1mm_brain_mask_dil_res.nii.gz
    # fslmaths MNI152_T1_1mm_brain_mask_dil_res.nii.gz -ero -ero
    #    MNI152_T1_1mm_brain_mask_dil_res_maths.nii.gz
    try:
        FSLDIR=os.environ['FSLDIR']
    except KeyError:
        raise RuntimeError("Need FSL installed and environment variable"
                           "FSLDIR set to locate templates")
    stdDir = os.path.join(FSLDIR,"data","standard")
    refImg = os.path.join(stdDir,"MNI152_T1_1mm.nii.gz")
    refMask = os.path.join(stdDir,"MNI152_T1_1mm_brain_mask_dil.nii.gz")
    cmdFL = shutil.which('flirt')
    cmdMA = shutil.which('fslmaths')
    cmdCX = shutil.which('convert_xfm')
    if cmdFL is None:
        raise RuntimeError("flirt command not found")
    tmpAff=tempfile.NamedTemporaryFile(suffix=".mat", prefix="masks-")
    tmpRes=tempfile.NamedTemporaryFile(suffix=".nii.gz", prefix="masks-")
    args = [cmdFL,"-ref",refImg,"-in",inImg,"-omat",tmpAff.name,"-usesqform",
            "-usesqform", "-dof", "12", "-cost", "normmi"]
    subpRes = subprocess.run(args, check=True)
    args = [cmdCX,"-inverse",tmpAff.name,"-omat",tmpAff.name]
    subpRes = subprocess.run(args, check=True)
    args = [cmdFL,"-in",refMask,"-ref",inImg,"-init",tmpAff.name,
            "-applyxfm", "-interp", "nearestneighbour",
            "-out",tmpRes.name]
    subpRes = subprocess.run(args, check=True)
    args = [cmdMA,tmpRes.name] + 2*["-ero"] + [outImg]
    subpRes = subprocess.run(args, check=True)


def maskGenRegAlad(inImg,outImg, replication=False):
    """Derive mask from registration of MNI152 using NiftyReg tools

    Requires NiftyReg (https://github.com/KCL-BMEIS/niftyreg) and
    NiftySeg (https://github.com/KCL-BMEIS/NiftySeg) installed,
    Alongside FSL for MNI152 templates (FSLDIR must be set). This is
    the recommended mask method for NXBC as it is very robust.
    Replication mode checks the version of reg_aladin used is the same
    as that used for NXBC evaluation, however more recent versions should
    perform comparably.

    Args:
        inImg -- input image path (nifti format)
        outImg -- output image path (nifti .nii.gz)
        replication -- replication mode, check reg_aladin version matches
                       that of nxbc publication
    Returns:
        
    """
    # Example: 
    # reg_aladin -aff MNI152_T1_2mm_aff.txt -flo MNI152_T1_2mm.nii.gz
    #    -ref image.nii.gz -res MNI152_T1_2mm_res.nii.gz
    # reg_resample -aff MNI152_T1_2mm_aff.txt
    #    -flo MNI152_T1_2mm_brain_mask_dil.nii.gz -inter 0 -ref image.nii.gz
    #    -res MNI152_T1_2mm_brain_mask_dil_res.nii.gz
    # seg_maths MNI152_T1_2mm_brain_mask_dil_res.nii.gz -ero 3.00000000
    #    MNI152_T1_2mm_brain_mask_dil_res_maths.nii.gz

    # replication mode (masks as in paper), requires
    # reg_aladin version (-v) 7ac1f44ab7ebda2d535a227de0d047e6462d33a7
    # was run with '-omp 4', but doesn't seem nec. for replication
    replicationVer = b'7ac1f44ab7ebda2d535a227de0d047e6462d33a7'

    try:
        FSLDIR=os.environ['FSLDIR']
    except KeyError:
        raise RuntimeError("Need FSL installed and environment variable"
                           "FSLDIR set to locate templates")
    cmdRA = shutil.which('reg_aladin')
    cmdRR = shutil.which('reg_resample')
    cmdSM = shutil.which('seg_maths')
    if cmdRA is None:
        raise RuntimeError("reg_aladin command not found")
    stdDir = os.path.join(FSLDIR,"data","standard")
    refImg = os.path.join(stdDir,"MNI152_T1_2mm.nii.gz")
    refMask = os.path.join(stdDir,"MNI152_T1_2mm_brain_mask_dil.nii.gz")
    # We don't really want tmpRes, but reg_aladin insists on writing res.nii.gz
    # if not supplied
    tmpAff=tempfile.NamedTemporaryFile(suffix=".aff", prefix="masks-")
    tmpRes=tempfile.NamedTemporaryFile(suffix=".nii.gz", prefix="masks-")

    if replication:
        args = [cmdRA,"-v"]
        subpRes = subprocess.run(args,check=True,capture_output=True)
        if subpRes.stdout != (replicationVer + b'\n'):
             raise RuntimeError("Replication masks mode requires "
                                "specific reg_aladin version "
                                +str(replicationVer))
    args = [cmdRA,"-flo",refImg,"-ref",inImg,"-aff",tmpAff.name,"-res",
            tmpRes.name]
    subpRes = subprocess.run(args, check=True)
    args = [cmdRR,"-aff",tmpAff.name,"-flo",refMask,"-inter","0",
            "-ref",inImg,"-res",tmpRes.name]
    subpRes = subprocess.run(args,check=True)
    args = [cmdSM,tmpRes.name,"-ero","3.00000000",outImg]
    subpRes = subprocess.run(args,check=True)


def maskGenRegAladRepl(inImg,outImg, replication=False):
    maskGenRegAlad(inImg,outImg, replication=True)


def maskGenAnts(inImg,outImg):
    """Derive mask from registration of MNI152 using ANTs tools

    Requires templateflow and antspyx python packages (available in pip).
    Provided to allow operation without installing non-python tools.

    Args:
        inImg -- input image path (nifti format)
        outImg -- output image path (nifti .nii.gz)
    Returns:
        
    """
    from templateflow import api as tflow
    import ants
    from ants.registration import affine_initializer
    refImg = tflow.get('MNI152NLin6Asym', desc=None, resolution=1,
              suffix='T1w', extension='nii.gz')
    startMask = tflow.get('MNI152NLin6Asym', desc="brain", resolution=1,
              suffix='mask', extension='nii.gz')
    startMaskImage = ants.image_read(str(startMask))
    # relatively close to FSL's MNI152_T1_1mm_brain_mask_dil.nii.gz
    refMaskImage  = ants.morphology( startMaskImage,
                                     operation='dilate', radius=5,
                                     mtype='binary', shape='box')
    inImage = ants.image_read(inImg)
    refImage= ants.image_read(str(refImg))
    # Using the initilizer appears to be the most robust approach,
    # on a small random sample of 4 ADNI images just running
    # ants.registration with type "Affine" failed on one. Type "TRSSA"
    # also succeeded, but as it re-runs the registration maybe it's also
    # prone to local minima.
    initreg = affine_initializer(inImage,refImage)
    oreg = ants.registration(inImage,refImage,type_of_transform="Affine",
                             initial_transform=initreg)
    omask = ants.apply_transforms(inImage,refMaskImage,
                                  oreg['fwdtransforms'],"nearestNeighbor")
    omask = ants.morphology(omask,operation='erode',radius=2,
                            mtype='binary',shape='box')
    ants.image_write(omask,outImg)


maskGenList = {"flirt":maskGenFlirt,
               "niftytools":maskGenRegAlad,
               "niftytoolsreplicate":maskGenRegAladRepl,
               "ants":maskGenAnts}


def maskGen(inimg, outimg=None, method="niftytools"):
    try:
        maskGenFn = maskGenList[method]
    except:
        raise ValueError("Method for maskGen not recognised")
    if (outimg is None):
        outimg=tempfile.NamedTemporaryFile(suffix=".nii.gz", prefix="masks-")
        outimg=outimg.name
    maskGenFn(inimg,outimg)
    return outimg

