# KomaMRI Phantom Conversion Files

This directory contains the files used to generate a KomaMRI-compatible version of the flow phantom described in [Fortin et al. (2021)](https://doi.org/10.1002/mrm.27114).

> **Note**: These files are **not required** to reproduce the experiments by Fortin et al., as the KomaMRI-compatible phantom is already available for direct download from [Zenodo](https://zenodo.org/records/15591102).

## Purpose

The scripts and files in this directory are provided for transparency and reproducibility regarding the conversion process. They allow users to see how the original phantom data (in JEMRIS format) was transformed into a KomaMRI-compatible format.

## Overview of Files

- **`.dat` files**: These are spin trajectory files provided by Alexandre Fortin and downloaded automatically from [Zenodo](https://zenodo.org/records/15591102). They are formatted for use with JEMRIS.
- **`generate_jemris_phantom.jl`**: A Julia script that generates the `.h5` file representing the JEMRIS phantom.
- **`jemris_flow_phantom_to_koma.jl`**: A Julia script that converts the spin trajectories and JEMRIS phantom into a format compatible with KomaMRI.

## Summary

While this directory is not essential for using the KomaMRI phantom directly, it serves as documentation of the conversion process from the original data provided by Fortin et al. to the KomaMRI format.
