# MEmilio - a high performance Modular EpideMIcs simuLatIOn software #

[![CI](https://github.com/DLR-SC/memilio/actions/workflows/main.yml/badge.svg)](https://github.com/DLR-SC/memilio/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/DLR-SC/memilio/branch/main/graph/badge.svg?token=DVQXIQJHBM)](https://codecov.io/gh/DLR-SC/memilio)

MEmilio is a common project between the Institute for Software Technology of the German Aerospace Center (DLR) and the department of Systems Immunology (SIMM) of the Helmholtz Center for Infection Research (HZI). This project will bring cutting edge and compute intensive epidemiological models to a large scale, which enables a precise and high-resolution spatiotemporal pandemic simulation for entire countries. MEmilio is still under developement but it is available as Open Source and we encourage everyone to make use of it. If you use it, please cite:

M. J. Kühn, D. Abele, T. Mitra, W. Koslow, M. Abedi, K. Rack, M. Siggel, S. Khailaie, M. Klitz, S. Binder, Luca Spataro, J. Gilg, J. Kleinert, M. Häberle, L. Plötzke, C. D. Spinner, M. Stecher, X. X. Zhu, A. Basermann, M. Meyer-Hermann, "Assessment of effective mitigation and prediction of the spread of SARS-CoV-2 in Germany using demographic information and spatial resolution". Mathematical Biosciences 339, 108648 (2021). https://www.sciencedirect.com/science/article/pii/S0025556421000845

**Getting started**

This project is divided into multiple building blocks. The C++ implementation of the epidemiological models can be found in the cpp directory (see the [README](cpp/README.md) there). Currently, there is an ODE-SECIR and an agent-based model. 

Contact and inter-county mobility data for Germany are to be found in [data](data/README.md). Data download tools are found in the pycode folder.

In pycode, different MEmilio python packages are defined. Via our [python bindings package](pycode/memilio-simulation), you can run our simulations from python; this package actually calls the C++ code from python. The [epidata package](pycode/memilio-epidata) provides tools to download and structure important data such as infection or mobility data. More about the python packages can be found in [Python README](pycode/README.rst).

**Documentation**

Each important part of the project described above is described in detail in the README in the corresponding directory. The README contains e.g. configuration and usage instructions for users and developers.

Also, the code is documented with doxygen and instructions on how to obtain it can be found in the docs folder.
The documentation of the code of the master branch can be found at the following URL:

https://dlr-sc.github.io/memilio/documentation/index.html

**Installation, Usage and Requirements**

Each part has different requirements and usage. Detailed instruction can be found in the corresponding READMEs.

**Development**
* [Git workflow and change process](https://github.com/DLR-SC/memilio/wiki/git-workflow)
* [Coding Guidelines](https://github.com/DLR-SC/memilio/wiki/coding-guidelines)
