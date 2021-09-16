import fire
import json


def main(prev_file, new_file, out_file):
    with open(prev_file, "r") as f:
        prev_data = json.loads(f.read())
    with open(new_file, "r") as f:
        new_data = json.loads(f.read())
    for k in prev_data:
        hypos = prev_data[k]["hypos"]
        for name in hypos:
            hypos[name]["scores"].update(new_data[k]["hypos"][name]["scores"])

    with open(out_file, "w") as f:
        f.write(json.dumps(prev_data))


if __name__ == "__main__":
    fire.Fire(main)

"""
python merge.py --prev_file pre.json --new_file new.json --out_file out.json
"""
