
from setuptools import setup

setup(
    name='tools',
    version=0.1,
    author='Kristian Monsen Haug, Mathias Lohne',
    author_email='mathialo@ifi.uio.no',
    license='MIT',
    description='Tools for working with compressive sensing',
    url='https://github.com/uio-cs/tools',
    install_requires=['tensorflow', 'numpy'],
    packages=['tools'],
    zip_safe=False)
