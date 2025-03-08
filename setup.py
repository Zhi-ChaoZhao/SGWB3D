from setuptools import setup, find_packages

setup(
    name='sgwb3d',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for computing angular power spectra and correlation functions for gravitational waves in 3D space.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/sgwb3d',  # Replace with your actual repository URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your actual license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)