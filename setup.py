from setuptools import setup, find_packages

setup(name='st_rl',
      version='1.0.0',
      author='Tao Sun',
      email='suntao.hn@gmail.com',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='This is a variation of orgial st_rl developed by Nikita Rudin. The rls_rl is a fast and simple RL algorithms implemented in pytorch',
      python_requires='>=3.10',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4",
            "tabulate",
      ],
      )
