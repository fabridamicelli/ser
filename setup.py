from setuptools import setup, find_packages


DESCRIPTION = 'SER Model'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

base_packages = [
    "numpy>=1.16.0",
    "numba==0.49.1",
]

test_packages = [
    "pytest>=4.0.2",
    "mypy>=0.770",
]


setup(
    name='ser',
    version='0.0.1',
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    author='Fabrizio Damicelli',
    author_email='f.damicelli@uke.de',
    url="https://github.com/fabridamicelli/ser",
    packages=find_packages(exclude=['notebooks', 'docs']),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.7',
    install_requires=base_packages,
    include_package_data=True
)
