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
          'nest-simulator>=2.20.0'
      ],
      include_package_data=False,
      zip_safe=False)
