from setuptools import setup

setup(
    name='fastldm',
    version='0.2',
    description='',
    packages=['fastldm'],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'flash-attn',
    ],
)
