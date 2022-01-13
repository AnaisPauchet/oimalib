from setuptools import find_packages, setup

project_name = "oimalib"

setup(
    name=project_name,
    version=0.1,
    packages=find_packages(),
    author="Anthony Soulain",
    author_email="anthony.soulain@protonmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Professional Astronomers",
        "Topic :: High Angular Resolution Astronomy :: Interferometry",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "matplotlib",
        "munch",
        "numpy",
        "emcee",
        "astropy",
        "scipy",
        "termcolor",
        "tqdm",
        "uncertainties",
        "astroquery",
        "corner",
        "pytest",
    ],
)
