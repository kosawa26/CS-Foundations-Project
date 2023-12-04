from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MissingValImputerDats6450",
    version="1.0",
    author="Bharat Premnath, Kentaro Osawa, Kunal Inglunkar, Navneeth Vittal H R",
    description="A package for handling missing values using MissForest.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kosawa26/CS-Foundations-Project.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "missforest",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
