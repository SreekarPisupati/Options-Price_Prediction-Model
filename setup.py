"""
Setup configuration for Options Price Prediction Model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="options-price-prediction-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML system for predicting option prices with 12%+ performance boost over Black-Scholes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Options-Price-Prediction-Model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "advanced": [
            "statsmodels>=0.13.0",
            "ta>=0.10.0",
            "fredapi>=0.4.3",
            "alpha_vantage>=2.3.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "options-predict=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="options trading machine learning finance quantitative black-scholes tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Options-Price-Prediction-Model/issues",
        "Source": "https://github.com/yourusername/Options-Price-Prediction-Model",
        "Documentation": "https://github.com/yourusername/Options-Price-Prediction-Model/tree/main/docs",
    },
)
