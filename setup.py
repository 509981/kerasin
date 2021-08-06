from setuptools import setup
#from distutils.core import setup

DISTUTILS_DEBUG

setup(
   name='kerasin',
   version='1.0',
   description='A useful module',
   author='Dmitry Utenkov',
   author_email='509981@gmail.com',
   #packages=['kerasin'],  #same as name
   py_modules=['kerasin']
   #install_requires=['bar', 'greek'], #external packages as dependencies
)
