from setuptools import setup, find_packages

setup(
    name="TensorFlow",
    version="0.1",
    packages=find_packages(where="."),
    install_requires=[
    	"tensorflow>=2.10,<3",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pyyaml"
    ],
    author="Tyler Kelly",
    package_dir={"": "."},
)
