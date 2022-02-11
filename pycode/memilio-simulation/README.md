# MEmilio Python Bindings

This package contains Python bindings for the MEmilio C++ library. It enables setting up and running simulations from Python code.

## Installation

Use the provided `setup.py` script to build the bindings and install the package. The script requires CMake and the Scikit-Build packages. Both are installed by the script if not available on the system. The package uses the [Pybind11 C++ library](https://pybind11.readthedocs.io) to create the bindings.

To install the package, use the command (from the directory containing `setup.py`)

```bash
pip install .
```

This builds the C++ library and C++ Python extension module and copies everything required to your site-packages. 

For developement of code use

```bash
pip install -e .[dev]
```

This command allows you to work on the code without having to reinstall the package after a change. Note that this only works for changes to Python code. If C++ code is modified, the install command has to be repeated every time. The command also installs all additional dependencies required for development and maintenance. 

For development, it may be easier to use the alternative command `python setup.py <build|install|develop>` which provides better configuration and observation of the C++ build process.

All the requirements of the [C++ library](../../cpp/README.md) must be met in order to build and use the python bindings. A virtual environment is recommended. 

CMake is executed internally by the `setup.py` script. All the options provided by the CMake configuration of the C++ library are available when building the Python extension as well. Additionally, the CMake configuration for the bindings provide the following CMake options:

- MEMILIO_USE_BUNDLED_PYBIND11: ON or OFF, default ON. If ON, downloads Pybind11 automatically from a repository during CMake configuration. If OFF, Pybind11 needs to be installed on the system.

When building the bindings, CMake options can be set by appending them to the install command, e.g.

```bash
python setup.py install -- -DCMAKE_BUILD_TYPE=Debug -DMEMILIO_USE_BUNDLED_PYBIND11=OFF
```

Alternatively, the `CMakeCache.txt` in the directory created by Scikit-Build can be edited to set the options.

## Usage

The package provides the following modules:

- `memilio.simulation`: core simulation framework and utilities, corresponds to the framework in `cpp/memilio`.
- `memilio.simulation.secir`: SECIR model and simulation with demographic and geographic resolution, corresponds to the model in `cpp/models/secir`.

Detailed documentation under construction. See the scripts in the examples directory for more information.

## Testing

The package provides a test suite in `memilio/simulation_test`. To run the tests, simply run the following command

```bash
python -m unittest
```

Note that these tests do not cover every case of the C++ library, they are only intended to test the binding code. To verify correctness of the C++ library itself, build and run the [C++ unit tests](../../cpp/README.md).
