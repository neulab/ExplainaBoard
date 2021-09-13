import sys
import string
import argparse
import explainaboard.tasks


def run_explainaboard(task, systems, output, analysis_type='single', is_print_ci=False, is_print_case=False,
                      is_print_ece=False):
    '''
    Run ExplainaBoard analysis suite

    Args:
      task: The ID of the task
      systems: A path to the system files
      output: The output path where the files should be written out
      analysis_type: analysis type: single|pair|combine
      is_print_ci: TODO
      is_print_case: TODO
      is_print_ece: TODO
    '''

    # TODO: This could probably be modified to directly find the directories in "tasks"
    valid_tasks = ['absa', 'ner', 'pos', 'chunk', 'cws', 'tc', 'nli', 're']
    if task not in valid_tasks:
        raise ValueError(f'{task} is not a known ExplainaBoard task')

    eval_func = getattr(sys.modules[f'explainaboard.tasks.{task}.eval_spec'], 'evaluate')
    eval_func(task_type=task,
              systems=systems,
              output_filename=output,
              analysis_type=analysis_type,
              is_print_ci=is_print_ci,
              is_print_case=is_print_case,
              is_print_ece=is_print_ece)


    
#if __name__ == '__main__':
def main():
    # python explainaboard_main.py --task absa  --systems ./test-laptop.tsv --output ./output/a.json
    # python explainaboard_main.py --task ner --systems ./test-conll03.tsv --output ./a.json
    # python explainaboard_main.py --task re --systems ./test_re.tsv --output ./a.json
    # python eval_spec.py  --task re --case True --systems ./test_re.tsv --output a.json --ci True
    parser = argparse.ArgumentParser(description='Explainable Leaderboards for NLP')

    parser.add_argument('--task', type=str, required=True,
                        help="absa")

    parser.add_argument('--ci', type=str, required=False, default=False,
                        help="True|False")

    parser.add_argument('--case', type=str, required=False, default=False,
                        help="True|False")

    parser.add_argument('--ece', type=str, required=False, default=False,
                        help="True|False")

    parser.add_argument('--type', type=str, required=False, default="single",
                        help="analysis type: single|pair|combine")
    parser.add_argument('--systems', type=str, required=True,
                        help="the directories of system outputs. Multiple one should be separated by comma, for example, system1,system2 (no space)")

    parser.add_argument('--output', type=str, required=True,
                        help="analysis output file")
    args = parser.parse_args()

    is_print_ci = args.ci
    is_print_case = args.case
    is_print_ece = args.ece

    task = args.task
    analysis_type = args.type
    systems = args.systems.split(",")
    output = args.output

    print("task", task)
    print("type", analysis_type)
    print("systems", systems)

    run_explainaboard(task, systems, output, analysis_type, is_print_ci, is_print_case, is_print_ece)
    
    
if __name__ == '__main__':
    main()
