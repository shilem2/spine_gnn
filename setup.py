import setuptools
from setuptools import setup

#version_file = open(os.path.join(os.getcwd(), 'VERSION'))


setup(
    name='gnn',
    version='0.0.1',
    #version=version_file.read().strip(),
    # url='https://code.medtronic.com/magic_sw_and_algorithm_team/algorithms/spine3d_wrapper.git',
    author='Moshe Shilemay',
    author_email='moshe.shilemay@medtronic.com',
    description='Playground for GNN',
    packages=setuptools.find_namespace_packages(),
)
