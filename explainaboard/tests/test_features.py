from dataclasses import asdict
import unittest

from explainaboard import feature
from explainaboard.feature import (
    BucketInfo,
    Dict,
    FeatureType,
    Position,
    Sequence,
    Value,
)


class TestFeature(unittest.TestCase):
    def test_get_spans(self):

        value1 = Value("double")
        self.assertEqual(value1.dtype, "float64")

        value2 = Value(dtype="double", is_bucket=True)
        self.assertEqual(value2.bucket_info.number, 4)

        value3 = Dict(feature={"a": 1})
        self.assertEqual(value3.feature, {'a': 1})

        value4 = Position(positions=[1, 2])
        self.assertEqual(value4.positions, [1, 2])

        value2_dict = asdict(value2)
        self.assertEqual(asdict(value2), asdict(Value.from_dict(value2_dict)))

        a = Sequence(
            feature=Dict(
                feature={
                    "span_text": Value("string"),
                    "span_tokens": Value(
                        dtype="float",
                        description="entity length",
                        is_bucket=True,
                        bucket_info=BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=4,
                            setting=(),
                        ),
                    ),
                    "span_pos": Position(positions=[0, 0]),
                    "span_tag": Value(
                        dtype="string",
                        description="entity tag",
                        is_bucket=True,
                        bucket_info=BucketInfo(
                            method="bucket_attribute_discrete_value",
                            number=4,
                            setting=1,
                        ),
                    ),
                }
            )
        )

        a_dict = asdict(a)
        self.assertEqual(
            FeatureType.from_dict(a_dict).feature.feature, a.feature.feature
        )

    def test_get_bucket_features(self):
        ner_task_features = feature.Features(
            {
                "tokens": feature.Sequence(feature=feature.Value("string")),
                "true_tags": feature.Sequence(feature=feature.Value("string")),
                "pred_tags": feature.Sequence(feature=feature.Value("string")),
                # --- the following are features of the sentences ---
                "sentence_length": feature.Value(
                    dtype="float",
                    description="sentence length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "entity_density": feature.Value(
                    dtype="float",
                    description="the ration between all entity "
                    "tokens and sentence tokens ",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "num_oov": feature.Value(
                    dtype="float",
                    description="the number of out-of-vocabulary words",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                "fre_rank": feature.Value(
                    dtype="float",
                    description=(
                        "the average rank of each word based on its frequency in "
                        "training set"
                    ),
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                    require_training_set=True,
                ),
                # --- the following are features of each entity ---
                "true_entity_info": feature.Sequence(
                    feature=feature.Dict(
                        feature={
                            "span_text": feature.Value("string"),
                            "span_tokens": feature.Value(
                                dtype="float",
                                description="entity length",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_pos": feature.Position(positions=[0, 0]),
                            "span_tag": feature.Value(
                                dtype="string",
                                description="entity tag",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "span_capitalness": feature.Value(
                                dtype="string",
                                description=(
                                    "The capitalness of an entity. For example, "
                                    "first_caps represents only the first character of "
                                    "the entity is capital. full_caps denotes all "
                                    "characters of the entity are capital"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_discrete_value",
                                    number=4,
                                    setting=1,
                                ),
                            ),
                            "span_rel_pos": feature.Value(
                                dtype="float",
                                description=(
                                    "The relative position of an entity in a sentence"
                                ),
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_chars": feature.Value(
                                dtype="float",
                                description="The number of characters of an entity",
                                is_bucket=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_econ": feature.Value(
                                dtype="float",
                                description="entity label consistency",
                                is_bucket=True,
                                require_training_set=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                            "span_efre": feature.Value(
                                dtype="float",
                                description="entity frequency",
                                is_bucket=True,
                                require_training_set=True,
                                bucket_info=feature.BucketInfo(
                                    method="bucket_attribute_specified_bucket_value",
                                    number=4,
                                    setting=(),
                                ),
                            ),
                        }
                    )
                ),
            }
        )
        bucket_features = ner_task_features.get_bucket_features()
        self.assertEqual(
            set(bucket_features),
            set(
                [
                    'sentence_length',
                    'entity_density',
                    'num_oov',
                    'fre_rank',
                    'span_tokens',
                    'span_tag',
                    'span_capitalness',
                    'span_rel_pos',
                    'span_chars',
                    'span_econ',
                    'span_efre',
                ]
            ),
        )
