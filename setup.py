from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='mldocs',
      version='0.0.1',
      description="Documents the Machine Learning process from data to model and performance.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Simon Stiebellehner',
      author_email='simon.stiebellehner@craftworks.at',
      url='https://github.com/craftworksgmbh/mldocs',
      packages=find_packages(exclude=['*test']),
      license='MIT',
      install_requires=[
            'numpy',
            'pandas',
            'sklearn',
            'scipy'
      ],
      classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent'
      ]
      )
