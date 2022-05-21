from __future__ import annotations


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


# def save_json(obj_json, path):
#     with open(path, "w") as f:
#         json.dump(obj_json, f, indent=4, ensure_ascii=False)


# def beautify_interval(interval):
#
#     if isinstance(interval[0], str):
#         return interval[0]
#     else:
#         if len(interval) == 1:
#             bk_name = '(' + format(float(interval[0]), '.3g') + ',)'
#             return bk_name
#         else:
#             range1_r = '(' + format(float(interval[0]), '.3g') + ','
#             range1_l = format(float(interval[1]), '.3g') + ')'
#             bk_name = range1_r + range1_l
#             return bk_name


# def tuple2str(triplet):
#     res = ""
#     for v in triplet:
#         res += str(v) + "|||"
#     return res.rstrip("|||")


# def interval_transformer(inter_list):
#     dict_old2new = {}
#     last = 0
#     for ind, interval in enumerate(inter_list):
#         if ind == 0:
#             last = interval[0]
#         if len(interval) == 1:
#             # new_inter_list.append(interval)
#             dict_old2new[interval] = interval
#             last = interval[0]
#         else:
#             # new_inter_list.append((last, interval[1]))
#             dict_old2new[interval] = (last, interval[1])
#             last = interval[1]
#     return dict_old2new
