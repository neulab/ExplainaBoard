class RankFlippingMetaAnalysis:  # (can inherit from an abstract MetaAnalysis class)
    def __init__(self, model1_report, model2_report):
        self.model1_report = model1_report
        self.model2_report = model2_report

    def run_meta_analysis(self):
        '''
        This method is what the user will call.
        '''

        # construct the new "metadata", treating each metric as a "feature"
        metadata = self._metrics_to_features()

        # construct the new "system outputs", treating each bucket from each
        # report as an "example/observation" to be bucketed further
        model1_meta_sysout = report_to_sysout(self.model1_report)
        model2_meta_sysout = report_to_sysout(self.model2_report)

        # get reference info
        reference_dict = self._get_reference_info(metadata)
        self.reference_dict = reference_dict

        # calculate paired metrics. This is what will be bucketed further to
        # obtain the results of meta-analysis.
        paired_sysout = self._get_paired_sysout(
            model1_meta_sysout, model2_meta_sysout, reference_dict, metadata
        )
        self.paired_sysout = (
            paired_sysout  # save before bucketing for more fine-grained analysis
        )

        # bucket the paired system outputs
        return self._bucket_paired_sysout(paired_sysout, metadata)

    def _metrics_to_features(self):
        '''
        Turns a `metric_configs` object into a metadata object suitable
        for use as metadata in a system output file.

        Uses the metric configs of `self.model1_report` as metadata.
        '''
        metadata = {
            'custom_features': {
                metric_config.name: {
                    "dtype": "string",  # for rank flipping, True or False
                    "description": metric_config.name,
                    "num_buckets": 2,  # for rank flipping, True or False
                }
                for metric_config in self.model1_report.metric_configs
            }
        }
        return metadata

    def _get_reference_info(self, metadata):
        '''
        Returns a dictionary indicating, for each metric, whether model1's value
        for that metric is higher than model2's value.

        Interpretation: this tells us which model (model1 or model2) we should
        expect to outperform the other, for each metric.

        The rank-flipping meta-analysis will then reveal which buckets, and how
        many, subvert this expectation.
        '''
        m1_overall = self.model1_report.results.overall
        m2_overall = self.model2_report.results.overall
        model1_metric_is_greater = {
            metric_name: m1_overall[metric_name].value > m2_overall[metric_name].value
            for metric_name in metadata['custom_features'].keys()
        }
        return model1_metric_is_greater

    def _get_paired_sysout(
        self, model1_meta_sysout, model2_meta_sysout, reference_dict, metadata
    ):
        '''
        Many meta-analyses require a quantity which is based on the relationship
        between the metrics of the two models we want to compare.

        For example, in rank-flipping, we are interested in whether, for each bucket,
        the relationship between model1's bucket-level metrics and model2's bucket-level
        metrics is the opposite from what is expected.

        The expectation between the relationship between model1 and model2 is passed in
        through `reference_dict`.
        '''

        def is_flipped(val1, val2, reference):
            '''
            `reference` contains our expectation/baseline of whether `val1` should be
            greater than `val2`.

            This function checks whether `val1` and `val2` stands in the *opposite*
            relationship as what is expected in `reference`. This is a sort of signal
            for "surprise".
            '''
            return (val1 > val2) != reference

        paired_score_examples = []
        for m1_bucket, m2_bucket in zip(model1_meta_sysout, model2_meta_sysout):

            # bucket info (feature name, bucket interval, bucket size) should match
            # exactly
            if m1_bucket['feature_name'] != m2_bucket['feature_name']:
                raise ValueError(
                    f'feature name does not match:\n{m1_bucket} vs {m2_bucket}'
                )
            if m1_bucket['bucket_interval'] != m2_bucket['bucket_interval']:
                raise ValueError(
                    f'bucket interval does not match:\n{m1_bucket} vs {m2_bucket}'
                )
            if m1_bucket['bucket_size'] != m2_bucket['bucket_size']:
                raise ValueError(
                    f'bucket size does not match:\n{m1_bucket} vs {m2_bucket}'
                )

            # calculate difference metrics
            example = {
                'feature_name': m1_bucket['feature_name'],
                'bucket_interval': m1_bucket['bucket_interval'],
                'bucket_size': m1_bucket['bucket_size'],
            }
            for feature in metadata['custom_features'].keys():
                reference = reference_dict[feature]
                example[feature] = is_flipped(
                    m1_bucket[feature], m2_bucket[feature], reference
                )
            paired_score_examples.append(example)

        return paired_score_examples

    def _bucket_paired_sysout(self, paired_sysout, metadata):
        '''
        Taking paired_sysout as a system output, do bucketing on it based on
        the metadata provided.

        Here we write the code which does the bucketing directly in this method,
        but in the future we may consider using the base ExplainaBoard package
        to do it for us, since `paired_sysout` is already in the format of a
        legitimate system output.
        '''

        # for rank-flipping, it's very easy to bucket, since there are only 2
        # buckets (True or False). Let's just immediately calculate it here.
        rank_flipping_buckets = {
            metric_name: {'ranking_same': 0, 'ranking_flipped': 0}
            for metric_name in metadata['custom_features'].keys()
        }

        for example in paired_sysout:
            for metric_name in metadata['custom_features'].keys():
                if example[metric_name] is True:
                    rank_flipping_buckets[metric_name]['ranking_flipped'] += 1
                else:
                    rank_flipping_buckets[metric_name]['ranking_same'] += 1
        return rank_flipping_buckets


def report_to_sysout(report):
    '''
    Loops through all the buckets in a report, converts them to "examples"
    as if they were a system output file.

    The metrics that describe each bucket become the "features" of this new
    system output.
    '''
    meta_examples = []
    for feature_name, feature_buckets in report.results.fine_grained.items():

        # feature_perfs has `n_buckets` elements, each corresponding to a single bucket
        for bucket in feature_buckets:

            # loop through and record all the metrics that describe this bucket
            example_features = {}
            for perf in bucket.performances:

                example_features['feature_name'] = feature_name
                example_features['bucket_interval'] = bucket.bucket_interval
                example_features['bucket_size'] = bucket.n_samples
                example_features[perf.metric_name] = perf.value
                # example_features[f'{perf.metric_name}_CI'] = \
                # [perf.confidence_score_low, perf.confidence_score_high]

            meta_examples.append(example_features)
    return meta_examples
