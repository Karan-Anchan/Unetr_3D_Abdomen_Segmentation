from setuptools import setup, find_packages

setup(
    name='UnetrAbdomen3DSegmentation',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'monai-weekly',
        'torch',
        'matplotlib',
    ],
)