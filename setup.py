"""
Setup configuration for Optimal Demonstration Selection for ICL
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="optimal-demo-selection-icl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Benchmarking demonstration selection methods for in-context learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Optimal-Demo-Selection-ICL",
    packages=find_packages(exclude=["tests", "notebooks", "experiments"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "icl-benchmark=experiments.run_benchmark:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
