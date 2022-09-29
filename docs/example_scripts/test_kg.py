from typing import cast

from explainaboard import get_loader_class, get_processor_class, TaskType

# This code details (1) how to evaluate your systems using ExplainaBoard
# programmatically (2)how to collect different results
# Load the data
from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.utils.typing_utils import unwrap

dataset = (
    "../../integration_tests/artifacts/kg_link_tail_prediction/no_custom_feature.json"
)
task = TaskType.kg_link_tail_prediction
loader = get_loader_class(task)(dataset, dataset)
data = loader.load()
# Initialize the processor and perform the processing
processor = get_processor_class(TaskType.kg_link_tail_prediction)()
sys_info = processor.process(metadata={}, sys_output=data.samples)

fine_grained_res = unwrap(sys_info.results.analyses)
overall_res = unwrap(sys_info.results.overall)

# print bucket information
for analysis in fine_grained_res:
    if analysis is not None:
        print(analysis.generate_report())

# save analysis report locally
sys_info.print_as_json(file=open("./report.json", 'w'))


# get overall results of different metrics
for name, metric_info in sys_info.results.overall[0].items():
    value = metric_info.value
    confidence_score_low = metric_info.confidence_score_low
    confidence_score_high = metric_info.confidence_score_high

    print(
        f"metric_name:{name}\n"
        f"value:{value}\n"
        f"confidence_score_low:{confidence_score_low}\n"
        f"confidence_score_high:{confidence_score_high}\n"
    )


# get fine-grained results
for analyses in fine_grained_res:
    buckets = cast(BucketAnalysisResult, analyses)
    for bucket_info in buckets.bucket_performances:
        for metric_name, bucket_performance in bucket_info.performances.items():
            value = bucket_performance.value
            confidence_score_low = bucket_performance.confidence_score_low
            confidence_score_high = bucket_performance.confidence_score_high

            print("------------------------------------------------------")

            bucket_name = unwrap(bucket_info.bucket_name)
            print(f"feature_name:{buckets.name} bucket_name:{bucket_name}")

            print(
                "\n"
                f"metric_name:{metric_name}\n"
                f"value:{value}\n"
                f"confidence_score_low:{confidence_score_low}\n"
                f"confidence_score_high:{confidence_score_high}\n"
            )
