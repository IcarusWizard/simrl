import os
import sys
from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simrl'))
from version import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='simrl',
    description="Simple Implementations of RL Algorithm in PyTorch",
    url="https://github.com/IcarusWizard/simrl",
    version=__version__,
    packages=find_packages(),
    author="Icarus",
    author_email="wizardicarus@gmail.com",
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "tensorboard",
        "ray",
        "gym",
        "numpy",
        "matplotlib",
        "tqdm",
        "tianshou",
        "pyro-ppl",
        "opencv-python",
        "moviepy",
    ],
    extras_require={
        'env' : ['gym[box2d]', 'pybullet']
    }
)