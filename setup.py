from setuptools import find_packages, setup
setup(
    name='grbalpha-tools',
    packages=find_packages(),
    version='0.1.12',
    description='tools to work with data from GRBAlpha and VZLUSAT-2 CubeSats',
    author='Marianna Dafcikova',
    install_requires=[         
        'astropy>=5.1.1',
        'pyorbital>=1.7.3',
        'pandas>=1.5.1',
        'dataclasses>=0.6',
        'numpy>=1.23.4',
        'matplotlib>=3.7.1',
        'scipy>=1.9.3'
    ],
)