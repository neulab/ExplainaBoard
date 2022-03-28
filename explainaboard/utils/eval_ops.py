from __future__ import annotations

from datalabs.operations.aggregate.auto_eval import auto_eval

from explainaboard import FileType, get_loader, get_processor, Source


@auto_eval(name="explainaboard")
def explainaboard(samples: str, dataset_info=None):
    # TODO(odashi): I didn't understand what this docstring mean.
    # This function seems to be not used anywhere, so I guessed the file can be removed.
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
        samples,
        Source.in_memory,
        FileType.datalab,
    )

    data = loader.load()

    # TODO(odashi): Mismatching the function signature, resulting in runtime panic.
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
