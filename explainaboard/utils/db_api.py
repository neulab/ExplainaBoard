import requests
import json


def read_statistics_from_db(
    dataset_name,
    subset_name=None,
    version='Hugging Face',
    transformation={'type': 'origin'},
):

    end_point_upload_dataset = "https://datalab.nlpedia.ai/api/normal_dataset/read_stat"
    data_info = {
        'dataset_name': dataset_name,
        'subset_name': subset_name,
        'version': version,
        'transformation': transformation,
    }
    response = requests.post(end_point_upload_dataset, json=data_info)
    # message = json.loads(response.text.replace("null", ""))["message"]
    return response


def write_statistics_from_db(
    dataset_name,
    subset_name=None,
    version='Hugging Face',
    transformation={'type': 'origin'},
    content={},
):

    end_point_upload_dataset = (
        "https://datalab.nlpedia.ai/api/normal_dataset/update_stat"
    )
    data_info = {
        'dataset_name': dataset_name,
        'subset_name': subset_name,
        'version': version,
        'transformation': transformation,
        'content': content,
    }
    response = requests.post(end_point_upload_dataset, json=data_info)
    # message = json.loads(response.text.replace("null", ""))["message"]
    return response
