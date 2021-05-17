from setuptools import setup, find_packages

setup(
    name='shape_inversion',
    version='0.0.1',
    description='shape-inversion',
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},
    install_requires=['torch==1.8.1',
                      'torchvision==0.9.1',
                      'plyfile==0.7.4',
                      'h5py==3.2.1',
                      'ninja==1.10.0.post2',
                      'matplotlib==3.4.2',
                      'scipy==1.6.3',
                      'tqdm==4.60.0',],
)
