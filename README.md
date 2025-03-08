# SGWB3D

SGWB3D is a Python package designed for computing angular power spectra and correlation functions for gravitational waves in 3D space with different polarization modes. The package provides a comprehensive set of tools for researchers and enthusiasts in the field of gravitational wave astronomy.

## Features

- Compute angular power spectra for various polarization modes: tensor, vector, scalar-transverse (ST), and scalar-longitudinal (SL).
- Calculate correlation functions based on the computed power spectra.
- Visualization tools for power spectra and correlation functions.
- Example usage and unit tests included for easy understanding and verification.

## Installation

To install the package, clone the repository and run the following command in the terminal:

```bash
python setup.py install
```

Alternatively, you can install the package in editable mode:

```bash
pip install -e .
```

## Usage

After installation, you can use the `GW3D` class from the `sgwb3d` package. Here is a basic example of how to use it:

```python
from sgwb3d.gw3d import GW3D

# Create an instance of the GW3D class
gw = GW3D(l_max=20)

# Compute power spectrum for tensor mode with group velocity v=0.5
C_zz = gw.C_zz(ell=2, alpha='tensor', v=0.5)

# Print the result
print(C_zz)
```

For more detailed examples, please refer to the `examples/basic_usage.py` file.

## Testing

To run the unit tests for the package, navigate to the project directory and execute:

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.