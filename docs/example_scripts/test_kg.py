from __future__ import annotations

from typing import cast

from explainaboard import get_loader_class, get_processor_class, TaskType

# This code details (1) how to evaluate your systems using ExplainaBoard
# programmatically (2)how to collect different results
# Load the data
from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.metrics.metric import ConfidenceInterval, Score
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

fine_grained_res = sys_info.results.analyses
overall_res = sys_info.results.overall

# print bucket information
for analysis in fine_grained_res:
    if analysis is not None:
        print(analysis.generate_report())

# save analysis report locally
sys_info.print_as_json(file=open("./report.json", 'w'))


# get overall results of different metrics
for name, metric_result in sys_info.results.overall["example"].items():
    value = metric_result.get_value(Score, "score").value
    ci = metric_result.get_value_or_none(ConfidenceInterval, "score_ci")

    print(
        f"metric_name:{name}\n"
        f"value:{value}\n"
        f"confidence_score_low:{ci.low if ci is not None else None}\n"
        f"confidence_score_high:{ci.high if ci is not None else None}\n"
    )


# get fine-grained results
for analyses in fine_grained_res:
    buckets = cast(BucketAnalysisResult, analyses)
    for bucket_performance in buckets.bucket_performances:
        for metric_name, metric_result in bucket_performance.results.items():
            value = metric_result.get_value(Score, "score").value
            ci = metric_result.get_value_or_none(ConfidenceInterval, "score_ci")

            print("------------------------------------------------------")

            bucket_name = unwrap(bucket_performance.bucket_name)
            print(f"feature_name:{buckets.name} bucket_name:{bucket_name}")

            print(
                "\n"
                f"metric_name:{metric_name}\n"
                f"value:{value}\n"
                f"confidence_score_low:{ci.low if ci is not None else None}\n"
                f"confidence_score_high:{ci.high if ci is not None else None}\n"
            )
