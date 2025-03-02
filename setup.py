from setuptools import setup, find_packages
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.version import __version__

setup(
    name="AI-Evolution",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "scipy",
        "pytest",
    ],
    entry_points={
        'console_scripts': [
            'ai-evolution=cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="An experimental project exploring AI self-evolution",
    keywords="AI, evolution, concept lattice",
    url="https://github.com/yourusername/AI-Evolution",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
) 