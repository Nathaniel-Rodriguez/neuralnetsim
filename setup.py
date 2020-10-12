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
          'numpy',
          'infomap>=1.1.4',
          'scipy',
          'networkx'
      ],
      include_package_data=False,
      zip_safe=False)
