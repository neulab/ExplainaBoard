from dataclasses import asdict
import unittest

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
