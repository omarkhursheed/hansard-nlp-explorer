from setuptools import setup, find_packages

setup(
    name="hansard",
    version="0.1.0",
    description="UK Parliamentary Hansard NLP Analysis Tools",
    author="Omar Khursheed",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
)
