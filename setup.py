from setuptools import setup

setup(name='dlgo',
      version='0.2',
      description='Deep Learning and the Game of Go',
      url='http://github.com/hmobley/dlgo',
      install_requires=[
            'numpy>=1.14.5', 
            'tensorflow>=1.10.1', 
            'keras>=2.2.2', 
            'future'],
      author='Rich Devine',
      author_email='hankdman@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)