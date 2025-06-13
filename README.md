# oticscream

**oticscream** is a Python library that integrates the ICSCREAM methodology as published in [Nuclear Science and Engineering](https://www.tandfonline.com/doi/full/10.1080/00295639.2021.1980362) with [OpenTURNS](https://openturns.org/), enabling advanced sensitivity analysis, uncertainty quantification, and robust model exploration. 

## 📌 Overview

This library implements the ICSCREAM (Identification of Penalizing Configurations in Computer Experiments Using Screening and Metamodel) methodology using OpenTURNS components. It is designed to help practitioners build surrogate models efficiently and assess model sensitivity and uncertainty in a structured, repeatable way, in order to identify penalizing input configurations.

## ✨ Features

- Full implementation of the ICSCREAM methodology
- Integration with OpenTURNS for given-data robust identification of penalizing input values
- Modular and extensible architecture
- Automated unit tests and documentation
- Suitable for industrial and academic use cases

## 📦 Installation

### Requirements

- Python 3.8+
- [OpenTURNS](https://openturns.org/)
- NumPy, SciPy, Matplotlib
- otkerneldesign

### Install with pip (local)

Clone this repository and install it using pip:

```bash
git clone https://github.com/vchabri/oticscream.git
cd oticscream
pip install .
```

### Optional: Development Setup

For a development install with editable mode:

```bash
pip install -e .[dev]
```

## 🛠 Usage

Here's a minimal example of how to use `oticscream`:

```python
from oticscream.core import ICSCREAM

# Define your model and input distribution
# model = ...
# input_distribution = ...

ics = ICSCREAM(model=model, input_distribution=input_distribution)
ics.run()
results = ics.get_results()
```

See the [documentation](#documentation) for full examples and API details.

## 🧚‍♂️ Testing

To run the test suite:

```bash
pytest test/
```

Make sure you have the development dependencies installed:

```bash
pip install -r requirements-dev.txt
```

## 📚 Documentation

Documentation is built using Sphinx. To build the HTML docs locally:

```bash
cd doc
make html
```

Output will be available in `doc/_build/html/index.html`.

## 🧠 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request

Please ensure code passes tests and follows PEP8 conventions.

## 📄 License

This project is licensed under the GNU General Public License v3.0 – see the [COPYING](COPYING) file for details.

## 🙏 Acknowledgments

- [The OpenTURNS consortium](https://openturns.org/)
- ICSCREAM methodology authors and contributors (A. Marrel, B. Iooss, V. Larget)

---

© 2025 - vchabri and contributors.
