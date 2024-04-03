from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ml_backtest',
    version='0.1.0',
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
