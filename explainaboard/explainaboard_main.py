import argparse
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
    # analysis = processor.process()
    analysis = processor.process().to_memory()




def main():

    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=True,
                        help="the task name")

    parser.add_argument('--system_outputs', type=str, required=True,
                        help="the directories of system outputs. Multiple one should be separated by comma, for example, system1,system2 (no space)")

    parser.add_argument('--type', type=str, required=False, default="single",
                        help="analysis type: single|pair|combine")

    parser.add_argument('--dataset', type=str, required=False, default="dataset_name",
                        help="the name of dataset")


    args = parser.parse_args()


    dataset = args.dataset
    task = args.task
    system_outputs = args.system_outputs.split(",")

    metadata = {
        "dataset_name":dataset,
        "task_name":task
    }

    if len(system_outputs) == 1: # individual system analysis
        run_explainaboard(task, system_outputs[0], metadata=metadata)
    else:
        raise NotImplementedError



    
if __name__ == '__main__':
    main()
