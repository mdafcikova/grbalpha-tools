from setuptools import find_packages, setup
setup(
    name='grbalpha-tools',
    packages=find_packages(),
    version='0.1.0',
    description='tools to work with data from GRBAlpha and VZLUSAT-2 CubeSats',
    author='Marianna Dafcikova',
    install_requires=[         
        'astropy>=5.1.1',
    ],
)