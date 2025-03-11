"""Setup script for persona_agent package."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="persona_agent",
    version="0.1.0",
    author="memenowLLC",
    author_email="billduke@memenow.xyz",
    description="AI persona simulation system based on AutoGen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/memenow/persona_agent",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=[
        "autogen-agentchat>=0.2.0",
        "openai>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.10.0", "mypy>=1.0.0"],
    },
)
