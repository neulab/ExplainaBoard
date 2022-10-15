"""Classes related to loading resources from disk."""


from explainaboard.loaders import file_loader, loader_factory

get_loader_class = loader_factory.get_loader_class
DatalabLoaderOption = file_loader.DatalabLoaderOption
