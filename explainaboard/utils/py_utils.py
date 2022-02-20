import itertools
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def zip_dict(*dicts):
    """Iterate over items of dictionaries grouped by their keys."""
    for key in set(itertools.chain(*dicts)):  # set merge all keys
        # Will raise KeyError if the dict don't have the same keys
        yield key, tuple(d[key] for d in dicts)


def sort_dict(dict_obj, flag="key"):
    sorted_dict_obj = []
    if flag == "key":
        sorted_dict_obj = sorted(dict_obj.items(), key=lambda item: item[0])
    elif flag == "value":
        # dict_bucket2span_
        sorted_dict_obj = sorted(
            dict_obj.items(), key=lambda item: len(item[1]), reverse=True
        )
    return dict(sorted_dict_obj)


def print_dict(dict_obj, print_infomation="dict"):
    # print("-----------------------------------------------")
    eprint("the information of #" + print_infomation + "#")
    eprint("Bucket_interval\tF1\tEntity-Number")
    for k, v in dict_obj.items():
        if len(k) == 1:
            eprint(
                "["
                + str(k[0])
                + ",]"
                + "\t"
                + str(v[0].value)
                + "\t"
                + str(v[0].n_samples)
            )
        else:
            eprint(
                "["
                + str(k[0])
                + ", "
                + str(k[1])
                + "]"
                + "\t"
                + str(v[0].value)
                + "\t"
                + str(v[0].n_samples)
            )

    eprint("")
