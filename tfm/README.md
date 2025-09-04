# *Contributions to Advanced Magnetic Resonance Imaging Simulation: A Model for Dynamic Simulation and a Web Interface for Pulse Sequence Development and Visualization*

This folder contains the code neccesary to recreate the experiments conducted in Section 4.1 of the Master's Thesis. \
[https://uvadoc.uva.es/handle/10324/77414](https://uvadoc.uva.es/handle/10324/77414)

## Demonstrative Experiments

- **4.1.2a: Simulation of patient motion and motion-corrected reconstruction**

    Simple example that shows the effect of patient motion during an MRI scan.
    The code is not in this folder but hosted at:
    https://juliahealth.org/KomaMRI.jl/stable/tutorial/05-SimpleMotion/


- **4.1.2b: Cardiac cine over an XCAT phantom**

    2D cine acquisition of a XCAT heart phantom model.

- **4.1.2c: Time of Flight (TOF) acquisition over a user-defined flow phantom**

    2D cine acquisition of a flow cylinder phantom, in which Time of Flight (TOF) effect is demonstrated.

- **4.1.2d: Phase Contrast (PC) imaging of a user-made phantom and a realistic aorta**

    This section contains two Phase Constrast (PC) experiments over two different phantoms:
    - User-defined flow phantom, which consists of two parallel flow cylinders, with blood flowing along opposite directions.
    - Realistic aorta phantom, whose velocities were extracted from the [Vascular Model Repository](https://www.vascularmodel.com/).

## Comparative Experiments

- **4.1.3a: Myocardial tagging over a user-defined phantom**

    2D cine tagging acquisition, which recreates the tagging experiment conducted in:

    Xanthis CG, Venetis IE, Aletras AH. *High performance MRI simulations of motion on multi-GPU systems*. J Cardiovasc Magn Reson. 2014 Jul 4;16(1):48
    https://doi.org/10.1186/1532-429X-16-48


- **4.1.3b: Validation of the Bloch solver under flow motion conditions**

    This experiment recreates the one oconducted in Section 4.1 from:

    Puiseux T, Sewonu A, Moreno R, Mendez S, Nicoud F. *Numerical simulation of time-resolved 3D phase-contrast magnetic resonance imaging*. PLOS ONE 2021 16(3)
    https://doi.org/10.1371/journal.pone.0248816 

- **4.1.3c: Turbulent flow with velocity encoded spoiled GRE**

    Spoiled GRE PC acquisition over a stenotic U-bend flow phantom. 
    This recreates the experiment carried out in Sections 2.8.1 and 3.2 from:

    Weine J, McGrath C, Dirix P, Buoso S, Kozerke S. *CMRsim–A python package for cardiovascular MR simulations incorporating complex motion and flow*. Magn Reson Med. 2024; 91: 2621-2637.
    https://doi.org/10.1002/mrm.30010

## Other Directories

- `/phantoms`: 

    This directory is empty, as phantom files are too large to be stored on GitHub. Instead, they have been included in [Zenodo](https://zenodo.org/records/14984766).

    Once downloaded from Zenodo, you need to place the phantom files in this `/phantoms` directory for everything to work correctly.

- `/sequences`:

    This directory contains Julia scripts that generate pulse sequences used for the experiments.

- `/utils`: 

    Some utility scripts for dividing phantoms into parts with a maximum number of spins and for visualizing cine acquisitions.

