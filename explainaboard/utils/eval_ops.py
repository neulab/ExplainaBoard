from typing import Callable, Mapping, Iterator
from explainaboard import get_processor
from explainaboard import get_loader
from explainaboard import get_task_categories, TaskType
from explainaboard import FileType, Source
from datalabs.operations.aggregate.auto_eval import auto_eval


@auto_eval(name="explainaboard")
def explainaboard(samples: Iterator, dataset_info=None):
    """
    Package: python
    Input:
        texts: Iterator
    Output:
        int
    """

    # Setup metadata
    metadata = {
        "dataset_name": dataset_info.builder_name,
        "sub_dataset_name": dataset_info.config_name,
        "task_name": dataset_info.task_templates[0].task_category,
        "reload_stat": True,
    }

    # if metric_names is not None:
    #     metadata["metric_names"] = metric_names

    loader = get_loader(
        dataset_info.task_templates[0].task_category,
        Source.in_memory,
        FileType.datalab,
        samples,
    )

    data = loader.load()

    # Run analysis
    report = get_processor(
        dataset_info.task_templates[0].task_category, metadata=metadata, data=data
    ).process()

    # res_info = {}
    # n_acc = 0
    # for sample in samples:
    #     true_label, predict_label = sample["label"], sample["prediction"]
    #     if true_label == predict_label:
    #         n_acc += 1

    # return {"accuracy":n_acc*1.0/len(samples)}

    return {"report": report}
