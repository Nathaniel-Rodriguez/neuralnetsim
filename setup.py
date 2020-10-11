from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='neuralnetsim',
      version='0.1.0',
      description='Library for the rodent neural slice project.',
      author='Nathaniel Rodriguez',
      packages=['neuralnetsim'],
      url='https://github.com/Nathaniel-Rodriguez/neuralnetsim.git',
      install_requires=[
          'numpy'
      ],
      include_package_data=True,
      zip_safe=False)
