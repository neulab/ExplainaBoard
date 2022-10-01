import os

from setuptools import setup

setup()

# TODO(odashi): Consider avoiding external invocation to install required models.
os.system("python -m spacy download en_core_web_sm")
