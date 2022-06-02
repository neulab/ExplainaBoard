import codecs
import os

from setuptools import find_packages, setup
from version import __version__

setup(
    name="explainaboard",
    version=__version__,
    description="Explainable Leaderboards for Natural Language Processing",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neulab/ExplainaBoard",
    author="Pengfei Liu",
    license="MIT License",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "explainaboard=explainaboard.explainaboard_main:main",
        ],
    },
    install_requires=[
        "datalabs>=0.4.3",
        "eaas>=0.3.9",
        "lexicalrichness",
        "matplotlib",
        "nltk>=3.2",
        "numpy",
        "sacrebleu",
        "scikit-learn",
        "scipy",
        "seqeval",
        "spacy",
        "tqdm",
        "wheel",
    ],
    extras_require={
        "dev": [
            "pre-commit",
        ],
    },
    include_package_data=True,
)

# TODO(odashi): Consider avoiding external invocation to install required models.
os.system("python -m spacy download en_core_web_sm")
