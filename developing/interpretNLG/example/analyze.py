import json
import sys
import numpy as np

# pre-processing
with open(sys.argv[1], encoding='utf-8') as f:
    lines = f.readlines()
metrics = sys.argv[2].strip().split(',')
output = open(sys.argv[3], 'w')

def bucket_strs(bucket_cutoffs):
    bucket_strs = []
    for i, x in enumerate(bucket_cutoffs):
        if i == 0:
            bucket_strs.append(f'<{x}')
        else:
            bucket_strs.append(f'[{bucket_cutoffs[i-1]},{x})')
    bucket_strs.append(f'>={x}')
    return bucket_strs

len_cutoffs = [10, 20, 30, 40, 50, 60]
len_bucket_strs = bucket_strs(len_cutoffs)
len_buckets = {metric: [[] for _ in range(len(len_cutoffs)+1)] for metric in metrics}
score_cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
score_bucket_strs = bucket_strs(score_cutoffs)
score_buckets = {metric: [[] for _ in range(len(score_cutoffs)+1)] for metric in metrics}
def cutoff_into_bucket(value, cutoffs):
    for i, v in enumerate(cutoffs):
        if value < v:
            return i
    return len(cutoffs)


# bucket
for line_id, l in enumerate(lines):
    l = json.loads(l)
    refs = l['refs']
    ref_len = np.mean([len(r.strip().split()) for r in refs])
    len_bucket = cutoff_into_bucket(ref_len, len_cutoffs)
    for metric in metrics:
        score = l[metric]
        len_buckets[metric][len_bucket].append((line_id, score))
        score_bucket = cutoff_into_bucket(score, score_cutoffs)
        score_buckets[metric][score_bucket].append((line_id, score))
    
# output
def sign_test(lid_scores, num_samples=1000):
    num = len(lid_scores)
    if num == 0:
        return 0, 0, 0, 0
    sample_size = int((num+1)*0.5)
    ids = list(range(num))
    scores = [x[1] for x in lid_scores]
    sign_scores = []
    for _ in range(num_samples):
        reduced_ids = np.random.choice(ids, size=sample_size, replace=True)
        reduced_scores = [scores[i] for i in reduced_ids]
        reduced_score = np.mean(reduced_scores)
        sign_scores.append(reduced_score)

    sign_scores.sort()
        
    return np.mean(scores), num, sign_scores[int(num_samples*0.025)], sign_scores[int(num_samples*0.975)]

output_json = {}
for metric in metrics:
    metric_len_bucket = []
    for bucket_id, bucket_str in enumerate(len_bucket_strs):
        bucket = {}
        bucket['bucket_name'] = bucket_str
        value, num, lower, upper = sign_test(len_buckets[metric][bucket_id])
        bucket['value'] = value
        bucket['num'] = num
        bucket['confidence'] = [lower, upper]
        bucket['bucket_error_case'] = [f'{score} ||| {line_id}' for line_id, score in len_buckets[metric][bucket_id]]
        metric_len_bucket.append(bucket)
    output_json[f'{metric}_by_length_bucket'] = metric_len_bucket

    count_metric_bucket = []
    for bucket_id, bucket_str in enumerate(score_bucket_strs):
        bucket = {}
        bucket['bucket_name'] = bucket_str
        bucket['value'] = bucket['num'] = len(score_buckets[metric][bucket_id])
        bucket['confidence'] = 0
        bucket['bucket_error_case'] = [f'{score} ||| {line_id}' for line_id, score in score_buckets[metric][bucket_id]]
        count_metric_bucket.append(bucket)
    output_json[f'count_of_{metric}'] = count_metric_bucket

        
with open(sys.argv[3], 'w', encoding='utf-8') as out:
    json.dump(output_json, out, ensure_ascii=False, indent=4)
