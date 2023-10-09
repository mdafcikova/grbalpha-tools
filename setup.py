from setuptools import find_packages, setup
setup(
    name='grbalpha-tools',
    packages=find_packages(),
    version='0.1.8',
    description='tools to work with data from GRBAlpha and VZLUSAT-2 CubeSats',
    author='Marianna Dafcikova',
    install_requires=[         
        'astropy>=5.1.1',
        'pyorbital>=1.7.3',
        'pandas',
        'dataclasses',
        'numpy',
        'matplotlib',
        'scipy',
        'os',
        'warnings'
    ],
)