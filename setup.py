from setuptools import setup, find_packages
import unittest
import codecs
import explainaboard
import explainaboard.tasks
# def test_suite():
#   test_loader = unittest.TestLoader()
#
#   return test_suite



setup(
  name="explainaboard",
  version=explainaboard.__version__,
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
  "Programming Language :: Python :: 3",
  ],
  packages=find_packages(),
  # packages=find_packages("explainaboard"),
  # package_data={
  #     "":["*.txt"],
  #     "tasks":["ner/*.aspects","ner/*.json"],
  # },
  entry_points={
    "console_scripts": [
      "explainaboard=explainaboard.explainaboard_main:main",
    ],
  },
  install_requires=[
      "nltk>=3.2",
      "numpy",
      "scipy",
      "matplotlib",
      "scikit-learn",
      "seqeval",
      "pandas"
  ],
  include_package_data=True,
)
