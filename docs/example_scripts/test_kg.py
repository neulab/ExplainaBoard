from explainaboard import get_custom_dataset_loader, get_processor, TaskType

# This code details (1) how to evaluate your systems using ExplainaBoard
# programmatically (2)how to collect different results
# Load the data
dataset = (
    "../../explainaboard/tests/artifacts/kg_link_tail_prediction/no_custom_feature.json"
)
task = TaskType.kg_link_tail_prediction
loader = get_custom_dataset_loader(task, dataset, dataset)
data = loader.load()
# Initialize the processor and perform the processing
processor = get_processor(TaskType.kg_link_tail_prediction.value)
sys_info = processor.process(metadata={}, sys_output=data.samples)


# print bucket information
processor.print_bucket_info(sys_info.results.fine_grained)  # type: ignore

# save analysis report locally
sys_info.print_as_json(file=open("./report.json", 'w'))


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
for feature_name, feature_info in sys_info.results.fine_grained.items():  # type: ignore
    for bucket_name, bucket_info in feature_info.items():
        bucket_n_samples = bucket_info.n_samples
        for bucket_performance in bucket_info.performances:
            metric_name = bucket_performance.metric_name
            value = bucket_performance.value
            confidence_score_low = bucket_performance.confidence_score_low
            confidence_score_high = bucket_performance.confidence_score_high

            print("------------------------------------------------------")
            print(f"feature_name:{feature_name} bucket_name:{bucket_name}")
            print(
                f"metric_name:{metric_name}\n"
                f"value:{value}\n"
                f"confidence_score_low:{confidence_score_low}\n"
                f"confidence_score_high:{confidence_score_high}\n"
            )
