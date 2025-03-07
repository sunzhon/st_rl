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
            "tensorboardX",
            "pandas==2.2.3",
            "termcolor==2.5.0",
            "pybullet==3.2.7",
            "neptune>=1.13.0"
            "neptune-optuna>=1.4.1"
            "optuna==3.1.0"
            "optuna-dashboard>=0.17.0"
      ],
      )
