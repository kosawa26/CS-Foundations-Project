from setuptools import setup, find_packages

setup(
    name='MissingValueImputerCSF',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'lightgbm',
        'missforest',
    ],
    entry_points={
        'console_scripts': [
            'MissingValueImputerCSF=MissingValueImputerCSF.MissingValueImputerCSF:main',
        ],
    },
)