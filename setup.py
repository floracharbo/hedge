"""Set up repository."""


import pathlib
from pip._internal.req import parse_requirements
from setuptools import find_packages, setup

REPO_ROOT = pathlib.Path(__file__).parent

with open(REPO_ROOT / "README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="data_preparation",
    description="Prepare local electricity usage data for RL simulations",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Flora Charbonnier",
    author_email="flora.charbonnier@eng.ox.ac.uk",
    url="https://github.com/floracharbo/hedge",
    license="GNU Affero General Public License v3.0",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt', session='hack'),
    python_requires=">=3.7",
)
