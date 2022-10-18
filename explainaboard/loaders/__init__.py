"""Classes related to loading resources from disk."""

from __future__ import annotations

from explainaboard.loaders import file_loader, loader_factory

get_loader_class = loader_factory.get_loader_class
DatalabLoaderOption = file_loader.DatalabLoaderOption
