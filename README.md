# Introduction

*Voxelize* converts a list of simulation particles which have a field associated with them
into a cubic lattice of voxels.
This can be useful, for example, to measure power spectra and other summary statistics,
as well as to generate data to be used in machine learning applications.

*Voxelize* uses a spherical top-hat kernel for the particle assignments.
Each particle can have an individual value for the radius of this top-hat.
In principle, extension to other kernels should be possible with relatively little modification
of the code; do get in touch if that is would you need.

Mathematically, *Voxelize* computes the field value associated with voxel *i* as

f<sub>i</sub> = &Sigma<sub>&alpha;</sub> f<sub>&alpha;</sub> V<sub>i,&alpha;</sub> V<sub>voxel</sub><sup>-1</sup>,

where &alpha; indexes the simulation particles, f<sub>&alpha;</sub> is the field associated with them,
V<sub>i,&alpha;</sub> is the overlap volume between voxel *i* and a sphere of radius r<sub>&alpha;</sub>
centered at particle &alpha;'s position, and V<sub>voxel</sub> is the volume of a voxel.
Note that with this definition, intensive fields f<sub>&alpha;</sub> are mapped to intensive fields f<sub>i</sub>,
for example if the input field associated with the particles is the local density,
the output field will be a local density too (and not a mass).


# Performance


# Before you build the GPU version



# Build
