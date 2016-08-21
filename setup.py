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
          'scikit-learn',
      ],
    dependency_links=['https://github.com/scikit-learn/scikit-learn/archive/0.17.1-1.tar.gz#egg=scikit-learn'],
    include_package_data=True,
    zip_safe=False)