from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='neuralnetsim',
    version='0.1',
    description='Uses nest for simulating neural networks: Must have nest v2.10.0 or greater',
    author='Nathaniel Rodriguez',
    packages=['neuralnetsim'],
    url='https://github.com/Nathaniel-Rodriguez/neuralnetsim.git',
    install_requires=[
          'networkx',
          'numpy',
          'matplotlib',
          'utilities-0.1'
      ],
    dependency_links=['https://github.com/Nathaniel-Rodriguez/utilities.git#egg=utilities-0.1'],
    include_package_data=True,
    zip_safe=False)