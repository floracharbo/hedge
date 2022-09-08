"""Set up repository."""
import pathlib

from setuptools import find_packages, setup

REPO_ROOT = pathlib.Path(__file__).parent

with open(REPO_ROOT / "README.md", encoding="utf-8") as f:
    README = f.read()

REQUIREMENTS = ["pandas"]

setup(
    name="data_preparation",
    description="Prepare local electricity usage data for RL simulations",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Flora Charbonnier",
    author_email="flora.charbonnier@eng.ox.ac.uk",
    url="https://github.com/floracharbo/data_preparation",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.7",
)
