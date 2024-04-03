from setuptools import setup, find_packages
import os

version = os.getenv('PACKAGE_VERSION', '0.1.0')

here = os.path.abspath(os.path.dirname(__file__))

readme_path = os.path.join(here, 'README.md')
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

requirements_path = os.path.join(here, 'requirements.txt')
with open(requirements_path) as f:
    requirements = f.read().splitlines()

setup(
    name='ml_backtest',
    version=version,
    packages=find_packages(),
    description='Backtesting of trading strategies with machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Maxwell Mendenhall',
    author_email='mendenhallmaxwell@gmail.com',
    url='https://github.com/MaxwellMendenhall/ml-backtest',
    install_requires=requirements,
    python_requires='>=3.11',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',  # Update as your project progresses
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
    ],
    include_package_data=True
)
