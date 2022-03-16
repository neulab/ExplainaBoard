def get_pairwise_performance_gap(sys1, sys2):

    for metric_name, performance_unit in sys1["results"]["overall"].items():
        sys1["results"]["overall"][metric_name]["value"] = float(
            sys1["results"]["overall"][metric_name]["value"]
        ) - float(sys2["results"]["overall"][metric_name]["value"])
        sys1["results"]["overall"][metric_name]["confidence_score_low"] = None
        sys1["results"]["overall"][metric_name]["confidence_score_high"] = None

    for attr, performance_list in sys1["results"]["fine_grained"].items():
        for idx, performances in enumerate(performance_list):
            for idy, performance_unit in enumerate(
                performances
            ):  # multiple metrics' results
                sys1["results"]["fine_grained"][attr][idx][idy]["value"] = float(
                    sys1["results"]["fine_grained"][attr][idx][idy]["value"]
                ) - float(sys2["results"]["fine_grained"][attr][idx][idy]["value"])
                sys1["results"]["fine_grained"][attr][idx][idy][
                    "confidence_score_low"
                ] = None
                sys1["results"]["fine_grained"][attr][idx][idy][
                    "confidence_score_high"
                ] = None

    return sys1
