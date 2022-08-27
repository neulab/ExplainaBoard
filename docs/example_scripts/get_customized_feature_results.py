from explainaboard import get_loader_class, get_processor, TaskType


# This code details (1) how to evaluate your systems using ExplainaBoard
# programmatically (2)how to get results of your customized features
def get_customized_results(dataset, customized_features):

    customized_features_performance = {}

    task = TaskType.kg_link_tail_prediction
    loader = get_loader_class(task)(dataset, dataset)
    data = loader.load()
    # Initialize the processor and perform the processing
    processor = get_processor(TaskType.kg_link_tail_prediction.value)
    metadata = {
        "task_name": TaskType.kg_link_tail_prediction.value,
        "custom_features": data.metadata.custom_features,
    }
    print(metadata)
    sys_info = processor.process(metadata=metadata, sys_output=data.samples)

    # print bucket information
    processor.print_analyses(sys_info.results.analyses)  # type: ignore

    # get overall results of different metrics
    for metric_name, metric_info in sys_info.results.overall.items():  # type: ignore

        metric_name = metric_info.metric_name
        value = metric_info.value
        confidence_score_low = metric_info.confidence_score_low
        confidence_score_high = metric_info.confidence_score_high

        print(
            f"metric_name:{metric_name}\n"
            f"value:{value }\n"
            f"confidence_score_low:{confidence_score_low}\n"
            f"confidence_score_high:{confidence_score_high}\n"
        )

    # get fine-grained results
    for (
        feature_name,
        feature_info,
    ) in sys_info.results.analyses.items():  # type: ignore
        if feature_name in customized_features:
            customized_features_performance[feature_name] = feature_info
    return customized_features_performance


customized_features = ["rel_type"]
dataset = (
    "../../integration_tests/artifacts/"
    "kg_link_tail_prediction/with_custom_feature.json"
)
customized_features_performance = get_customized_results(dataset, customized_features)
# print(customized_features_performance)

for (
    feature_name,
    feature_info,
) in customized_features_performance.items():  # type: ignore
    for bucket_name, bucket_info in feature_info.items():
        bucket_n_samples = bucket_info.n_samples
        for bucket_performance in bucket_info.performances:
            metric_name = bucket_performance.metric_name
            value = bucket_performance.value
            confidence_score_low = bucket_performance.confidence_score_low
            confidence_score_high = bucket_performance.confidence_score_high

            print("------------------------------------------------------")
            print(f"feature_name:{feature_name}")
            print(
                f"metric_name:{metric_name}\n"
                f"value:{value}\n"
                f"confidence_score_low:{confidence_score_low}\n"
                f"confidence_score_high:{confidence_score_high}\n"
            )
