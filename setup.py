import setuptools

setuptools.setup(
    name="dynawind",
    version="0.2.0",
    url="https://bitbucket.org/24sea/dynawind",
    author="Wout Weijtjens",
    author_email="wout.weijtjens@24sea.eu",
    description="DYNAwind package for python",
    long_description=open("README.rst").read(),
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pytz",
        "scipy",
        "nptdms",
        "sklearn",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
    zip_safe=False,
)
