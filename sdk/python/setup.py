from setuptools import setup, find_packages

setup(
    name="memchip",
    version="0.2.0",
    description="MemChip — Memory-as-a-service for AI agents",
    author="MemChip",
    url="https://github.com/memchip/memchip-python",
    packages=find_packages(),
    install_requires=["httpx>=0.24.0"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
