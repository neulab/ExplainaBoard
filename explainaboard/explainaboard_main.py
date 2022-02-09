import argparse
import json
from explainaboard import FileType, Source, get_loader, get_processor
from explainaboard import TaskType

def run_explainaboard(task:TaskType, path_system_output:str, metadata:dict={}):
    '''
        task:TaskType:  the task name supported by ExplainaBoard
        path_system_output:str:  the file path of system output
        metadata:dict: other metadata information for analysis report, such as dataset names.
    '''


    if task not in TaskType.list():
        raise ValueError(f'{task} can not been recognized. ExplainaBoard currently supports: {TaskType.list()}')

    loader = get_loader(task, data = path_system_output)
    data = loader.load()
    processor = get_processor(task, metadata = metadata, data = data)
    analysis = processor.process()

    return analysis
    #analysis = processor.process().to_memory()

def get_performance_gap(sys1, sys2):

    for metric_name, performance_unit in sys1["results"]["overall"].items():
        sys1["results"]["overall"][metric_name]["value"] = float(sys1["results"]["overall"][metric_name]["value"]) -  float(sys2["results"]["overall"][metric_name]["value"])
        sys1["results"]["overall"][metric_name]["confidence_score_low"] = 0
        sys1["results"]["overall"][metric_name]["confidence_score_up"] = 0


    for attr, performance_list in sys1["results"]["fine_grained"].items():
        for idx, performances in enumerate(performance_list):
            for idy, performance_unit in enumerate(performances): # multiple metrics' results
                sys1["results"]["fine_grained"][attr][idx][idy]["value"] = float(sys1["results"]["fine_grained"][attr][idx][idy]["value"]) - float(sys2["results"]["fine_grained"][attr][idx][idy]["value"])
                sys1["results"]["fine_grained"][attr][idx][idy]["confidence_score_low"] = 0
                sys1["results"]["fine_grained"][attr][idx][idy]["confidence_score_up"] = 0


    return sys1


def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=True,
                        help="the task name")

    parser.add_argument('--system_outputs', type=str, required=True, nargs="+",
                        help="the directories of system outputs. Multiple one should be separated by space, for example: system1 system2")

    parser.add_argument('--type', type=str, required=False, default="single",
                        help="analysis type: single|pair|combine")

    parser.add_argument('--dataset', type=str, required=False, default="dataset_name",
                        help="the name of dataset")

    parser.add_argument('--metrics', type=str, required=False, nargs="*",
                        help="multiple metrics should be separated by space")


    args = parser.parse_args()


    dataset = args.dataset
    task = args.task
    system_outputs = args.system_outputs
    metric_names = args.metrics

    metadata = {
        "dataset_name": dataset,
        "task_name": task
    }

    if metric_names != None:
        metadata["metric_names"] = metric_names



    if len(system_outputs) == 1: # individual system analysis
        run_explainaboard(task, system_outputs[0], metadata=metadata).to_memory()
    else:
        #raise NotImplementedError
        report_sys1 = run_explainaboard(task, system_outputs[0], metadata=metadata).to_dict()
        report_sys2 = run_explainaboard(task, system_outputs[1], metadata=metadata).to_dict()

        # To be implemented: we should validate the format consistency of these two system reports

        compare_analysis = get_performance_gap(report_sys1, report_sys2)

        print(json.dumps(compare_analysis, indent=4))





    
if __name__ == '__main__':
    main()
