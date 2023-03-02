from setuptools import setup

setup(
    name='fastldm',
    version='0.1',
    description='',
    packages=['fastldm'],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'flash-attn',
    ],
)
