from setuptools import find_packages, setup


setup(name='mldocs',
      version='0.0.1',
      description='',
      author='Simon Stiebellehner',
      author_email='simon.stiebellehner@craftworks.at',
      url='https://github.com/stiebels/mldocs',
      packages=find_packages(exclude=['*test']),
      license='MIT',
      install_requires=[
            'numpy',
            'pandas',
            'sklearn',
            'scipy'
      ],
      )
