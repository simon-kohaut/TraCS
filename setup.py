"""Sets up the TraCS package for installation."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import re

import setuptools

# Find TraCS version and author strings
with open("tracs/__init__.py", encoding="utf8") as fd:
    content = fd.read()
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)
    author = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

# Import readme
with open("README.md", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="tracs",
    version=version,
    author=author,
    author_email="simon.kohaut@cs.tu-darmstadt.de",
    description="A Python package to apply probabilistic logic programming to motion prediction tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        "tracs": ["py.typed"],  # https://www.python.org/dev/peps/pep-0561/
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Inference backend
        "promis>=4.0.0.dev1",
        # Language tools
        "langchain>=0.3",
        "langchain-core>=0.3",
        "langchain-ollama>=0.2",
        # CLI
        "rich>=13.0",
    ],
    entry_points={
        "console_scripts": [
            "tracs_gen=tracs.cli:main",
        ],
    },
)