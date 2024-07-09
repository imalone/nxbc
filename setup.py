from setuptools import setup

setup(
  name='nxbc',
  version='0.1.2',
  author='Ian Malone',
  author_email='i.malone@ucl.ac.uk',
  packages=['nxbc'],
  scripts=['bin/nxbc', 'bin/nxbc-sharpenonly'],
  url='https://github.com/imalone/nxbc.git',
  license='LICENSE.txt',
  classifiers=['License :: OSI Approved :: BSD License'],
  description='NX Bias Correction. NIfTI image bias correction, implementation of N3 (Sled) and N4 (Tustison) algorithms',
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  install_requires=[
    "SplineSmooth3D >= 0.1.0",
    "nibabel >= 2.5.1",
    "scikit_image >= 0.15.0",
    "matplotlib >= 3.1.1",
    "numpy >= 1.17.2",
    "scipy >= 1.3.1",
  ],
)
