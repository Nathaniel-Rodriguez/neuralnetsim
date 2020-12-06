from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='neuralnetsim',
      version='0.1.0',
      description='Library for the rodent neural slice project.',
      author='Nathaniel Rodriguez',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      url='https://github.com/Nathaniel-Rodriguez/neuralnetsim.git',
      install_requires=[
          'numpy>=1.19.2',
          'infomap>=1.1.4',
          'scipy>=1.5.2',
          'networkx>=2.5',
          'matplotlib>=3.3.2',
          'statsmodels>=0.12.0',
          'seaborn>=0.11.0',
          'PyNEST',
          'dask>=2.30.0',
          'distributed>=2.30.0',
          'pandas>=1.1.3'
      ],
      include_package_data=False,
      python_requires='>=3.7',
      zip_safe=False)
