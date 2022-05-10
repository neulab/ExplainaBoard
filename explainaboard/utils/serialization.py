import copy


def general_to_dict(data):
    if hasattr(data, 'to_dict'):
        return getattr(data, 'to_dict')()
    else:
        return copy.deepcopy(data)


def explainaboard_dict_factory(data):
    """
    This can be used to serialize data through the following command:
    serialized_data = dataclasses.asdict(data, dict_factory=explainaboard_dict_factory)
    """
    return {field: general_to_dict(value) for field, value in data}
