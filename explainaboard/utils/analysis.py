import json


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return "low_caps"
    elif s.upper() == s:
        return "full_caps"
    elif s[0].upper() == s[0]:
        return "first_caps"
    else:
        return "not_first_caps"


def save_json(obj_json, path):
    with open(path, "w") as f:
        json.dump(obj_json, f, indent=4, ensure_ascii=False)


def beautify_interval(interval):

    if isinstance(interval[0], str):
        return interval[0]
    else:
        if len(interval) == 1:
            bk_name = '(' + format(float(interval[0]), '.3g') + ',)'
            return bk_name
        else:
            range1_r = '(' + format(float(interval[0]), '.3g') + ','
            range1_l = format(float(interval[1]), '.3g') + ')'
            bk_name = range1_r + range1_l
            return bk_name


def tuple2str(triplet):
    res = ""
    for v in triplet:
        res += str(v) + "|||"
    return res.rstrip("|||")


def interval_transformer(inter_list):
    dict_old2new = {}
    last = 0
    for ind, interval in enumerate(inter_list):
        if ind == 0:
            last = interval[0]
        if len(interval) == 1:
            # new_inter_list.append(interval)
            dict_old2new[interval] = interval
            last = interval[0]
        else:
            # new_inter_list.append((last, interval[1]))
            dict_old2new[interval] = (last, interval[1])
            last = interval[1]
    return dict_old2new


def find_key(dict_obj, x):
    for k, v in dict_obj.items():
        if len(k) == 1:
            if x == k[0]:
                return k
        elif len(k) == 2 and x >= k[0] and x <= k[1]:  # Attention !!!
            return k


def reverse_dict(dict_a2b):
    dict_b2a = {}
    for k, v in dict_a2b.items():
        v = float(v)
        if v not in dict_b2a.keys():
            dict_b2a[float(v)] = [k]
        else:
            dict_b2a[float(v)].append(k)

    return dict_b2a


def reverse_dict_discrete(dict_a2b):
    dict_b2a = {}
    # print(dict_a2b)
    for k, v in dict_a2b.items():
        if v not in dict_b2a.keys():
            dict_b2a[v] = [k]
        else:
            dict_b2a[v].append(k)

    return dict_b2a
