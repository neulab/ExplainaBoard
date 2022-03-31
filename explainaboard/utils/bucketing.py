from explainaboard.utils.analysis import find_key, reverse_dict, reverse_dict_discrete
from explainaboard.utils.py_utils import sort_dict


def bucket_attribute_specified_bucket_value(
    dict_obj=None, bucket_number=4, bucket_setting=None
):
    if not dict_obj or len(dict_obj) == 0:
        return None
    # Bucketing different Attributes
    dict_span2att_val = dict_obj
    n_buckets = bucket_number
    hardcoded_bucket_values = bucket_setting

    # hardcoded_bucket_values = [set([float(0), float(1)])]
    # print("!!!debug-7--")
    p_infinity = 1000000
    n_infinity = -1000000
    n_spans = len(dict_span2att_val)
    dict_att_val2span = reverse_dict(dict_span2att_val)
    dict_att_val2span = sort_dict(dict_att_val2span)
    dict_bucket2span = {}

    for bucket_value in hardcoded_bucket_values:
        if bucket_value in dict_att_val2span.keys():
            # print("------------work!!!!---------")
            # print(bucket_value)
            dict_bucket2span[(bucket_value,)] = dict_att_val2span[bucket_value]
            n_spans -= len(dict_att_val2span[bucket_value])
            n_buckets -= 1

    avg_entity = n_spans * 1.0 / n_buckets
    n_tmp = 0
    entity_list = []
    val_list = []

    #
    # print("-----avg_entity----------")
    # print(avg_entity)

    for att_val, entity in dict_att_val2span.items():
        if att_val in hardcoded_bucket_values:
            continue

        # print("debug-att_val:\t",att_val)
        val_list.append(att_val)
        entity_list += entity
        n_tmp += len(entity)

        # print(att_val)
        # print(n_tmp, avg_entity)

        if n_tmp > avg_entity:

            if len(val_list) >= 2:
                key_bucket = (val_list[0], val_list[-1])
                dict_bucket2span[key_bucket] = entity_list
            # print("debug key bucket:\t", key_bucket)
            else:
                dict_bucket2span[(val_list[0],)] = entity_list
            entity_list = []
            n_tmp = 0
            val_list = []
    if n_tmp != 0:
        if n_buckets == 1:
            dict_bucket2span[(n_infinity, p_infinity)] = entity_list
        else:
            if val_list[0] <= 1:
                p_infinity = 1.0
            # print("!!!!!-debug-2")
            if len(val_list) >= 2:
                key_bucket = (val_list[0], p_infinity)
                dict_bucket2span[key_bucket] = entity_list
            else:
                dict_bucket2span[(val_list[0], p_infinity)] = entity_list  # fix bugs

    return dict_bucket2span


def bucket_attribute_discrete_value(
    dict_obj=None, bucket_number=100000000, bucket_setting=1
):
    # Bucketing different Attributes

    # print("!!!!!debug---------")
    # 	hardcoded_bucket_values = [set([float(0), float(1)])]
    dict_span2att_val = dict_obj
    # print(f"dict_span2att_val:\n{dict_span2att_val}")
    n_buckets = bucket_number
    n_entities = bucket_setting

    dict_bucket2span = {}

    dict_att_val2span = reverse_dict_discrete(dict_span2att_val)
    dict_att_val2span = sort_dict(dict_att_val2span, flag="value")
    # dict["q_id"] = 2

    n_total = 1
    for att_val, entity in dict_att_val2span.items():

        if len(entity) < n_entities or n_total > n_buckets:
            break
        dict_bucket2span[(att_val,)] = entity

        n_total += 1

    return dict_bucket2span


def bucket_attribute_specified_bucket_interval(
    dict_obj=None, bucket_number=None, bucket_setting=None
):
    # Bucketing different Attributes

    # hardcoded_bucket_values = [set([float(0), float(1)])]

    # intervals = [0, (0,0.5], (0.5,0.9], (0.99,1]]
    dict_span2att_val = dict_obj
    intervals = bucket_setting

    dict_bucket2span = {}

    # print("!!!!!!!enter into bucket_attribute_SpecifiedBucketInterval")

    if isinstance(list(intervals)[0][0], str):  # discrete value, such as entity tags
        dict_att_val2span = reverse_dict_discrete(dict_span2att_val)
        dict_att_val2span = sort_dict(dict_att_val2span, flag="value")
        for att_val, entity in dict_att_val2span.items():
            att_val_tuple = (att_val,)
            if att_val_tuple in intervals:
                if att_val_tuple not in dict_bucket2span.keys():
                    dict_bucket2span[att_val_tuple] = entity
                else:
                    dict_bucket2span[att_val_tuple] += entity

        for val in intervals:
            if val not in dict_bucket2span.keys():
                dict_bucket2span[val] = []
    # print("dict_bucket2span: ",dict_bucket2span)
    else:
        # print("---debug----5")
        # print(intervals)
        dict_att_val2span = reverse_dict(dict_span2att_val)
        dict_att_val2span = sort_dict(dict_att_val2span)
        for v in intervals:
            if len(v) == 1:
                dict_bucket2span[v] = []
            else:
                dict_bucket2span[v] = []

        # print("debug-interval:\t", intervals)

        for att_val, entity in dict_att_val2span.items():
            res_key = find_key(dict_bucket2span, att_val)
            # print("res-key:\t"+ str(res_key))
            if res_key is None:
                continue
            dict_bucket2span[res_key] += entity

    return dict_bucket2span
