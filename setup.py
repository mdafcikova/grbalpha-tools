from setuptools import find_packages, setup
setup(
    name='grbalpha-tools',
    packages=find_packages(),
    version='0.1.0',
    description='tools to work with data from GRBAlpha and VZLUSAT-2 CubeSats',
    author='Marianna Dafcikova',
    install_requires=[         
        'dataclasses>=0.6',         
        'pandas>=1.5.1',
        'numpy>=1.23.4',
        'matplotlib>=3.6.2',
        'astropy>=5.1.1',
        'scipy>=1.9.3',
        'os',
        'warnings',
    ],
)