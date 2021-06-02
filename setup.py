from setuptools import setup, find_packages
import unittest
import codecs
import interpret_eval
import interpret_eval.tasks
# def test_suite():
#   test_loader = unittest.TestLoader()
#
#   return test_suite



setup(
  name="interpret_eval",
  version=interpret_eval.__version__,
  description="Interpretable Evaluation for Natural Language Processing",
  long_description=codecs.open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/neulab/ExplainaBoard",
  author="Pengfei Liu",
  license="MIT License",
  classifiers=[
  "Intended Audience :: Developers",
  "Topic :: Text Processing",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "License :: OSI Approved :: BSD License",
  "Programming Language :: Python :: 3",
  ],
  packages=find_packages(),
  # packages=find_packages("interpret_eval"),
  # package_data={
  #     "":["*.txt"],
  #     "tasks":["ner/*.aspects","ner/*.json"],
  # },
  entry_points={
    "console_scripts": [
      "interpret-eval=interpret_eval.interpret_eval_main:main",
    ],
  },
  install_requires=[
      "nltk >= 3.2",
      "numpy",
      "scipy",
      "matplotlib",
      "scikit-learn",
      "seqeval==0.0.12",
  ],
  include_package_data=True,
)