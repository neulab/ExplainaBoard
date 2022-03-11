from sklearn.metrics import accuracy_score
from random import choices
import numpy as np
import scipy
import sklearn.metrics


class Metric:
    def __init__(self):
        self._name = None
        self._n_samples = None
        self._eval_function = None
        self._n_times = 1000
        self._sampling_rate = 0.8
        self._results = None
        self._is_print_confidence_interval = False

    def get_confidence_interval(self, *args, **kwargs):
        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
            return m - h, m + h

        n_sampling = int(self._n_samples * self._sampling_rate)
        if n_sampling == 0:
            n_sampling = 1

        # print(f"n_sampling: {n_sampling}\n"
        #       f"self._n_samples {self._n_samples}\n")

        performance_list = []
        confidence_low, confidence_up = 0, 0
        for i in range(self._n_times):
            sample_index_list = choices(range(self._n_samples), k=n_sampling)
            performance = self._eval_function(
                np.array(args[0])[sample_index_list],
                np.array(args[1])[sample_index_list],
                **kwargs
            )
            performance_list.append(performance)

        if self._n_times != 1000:
            confidence_low, confidence_up = mean_confidence_interval(performance_list)
        else:
            performance_list.sort()
            confidence_low = performance_list[24]
            confidence_up = performance_list[974]
        return confidence_low, confidence_up

    def _evaluate(self, *args, **kwargs):

        self._results = {}
        self._results["value"] = self._eval_function(*args, **kwargs)

        if not self._is_print_confidence_interval:
            self._results["confidence_score_low"] = 0
            self._results["confidence_score_up"] = 0
        else:
            (
                confidence_interval_low,
                confidence_interval_up,
            ) = self.get_confidence_interval(*args, **kwargs)
            self._results["confidence_score_low"] = confidence_interval_low
            self._results["confidence_score_up"] = confidence_interval_up

        return self._results


class Accuracy(Metric):
    def __init__(
        self, true_labels, predicted_labels, is_print_confidence_interval=False
    ):
        super(Accuracy, self).__init__()
        # Metric.__init__(self)
        self._name = self.__class__.__name__
        self._true_labels = true_labels
        self._predicted_labels = predicted_labels
        self._eval_function = accuracy_score
        self._is_print_confidence_interval = is_print_confidence_interval
        self._n_samples = len(self._true_labels)

    def evaluate(self):

        return self._evaluate(self._true_labels, self._predicted_labels)


class F1score(Metric):
    def __init__(
        self, true_labels, predicted_labels, is_print_confidence_interval=False
    ):
        super(F1score, self).__init__()
        # Metric.__init__(self)
        self._name = self.__class__.__name__
        self._true_labels = true_labels
        self._predicted_labels = predicted_labels
        self._eval_function = sklearn.metrics.f1_score
        self._is_print_confidence_interval = is_print_confidence_interval
        self._n_samples = len(self._true_labels)

    def evaluate(self):
        # print(self._true_labels[0:10])
        # print(self._predicted_labels[0:10])
        # print(sklearn.metrics.f1_score(self._true_labels[0:10], self._predicted_labels[0:10], average='micro'))
        # exit()
        return self._evaluate(
            self._true_labels, self._predicted_labels, average='micro'
        )


def hits(true_labels, predicted_labels):
    num_hits = 0
    for i in range(len(true_labels)):
        i_true = true_labels[i]
        i_preds = predicted_labels[i]
        if i_true in i_preds:
            num_hits += 1
    return num_hits / len(true_labels)


class Hits(Metric):
    def __init__(
        self, true_labels, predicted_labels, is_print_confidence_interval=False
    ):
        super(Hits, self).__init__()
        # Metric.__init__(self)
        self._name = self.__class__.__name__
        self._true_labels = true_labels
        self._predicted_labels = predicted_labels
        self._eval_function = self.hits
        self._is_print_confidence_interval = is_print_confidence_interval
        self._n_samples = len(self._true_labels)

    @staticmethod
    def hits(true_labels, predicted_labels):
        num_hits = 0
        for i in range(len(true_labels)):
            i_true = true_labels[i]
            i_preds = predicted_labels[i]
            if i_true in i_preds:
                num_hits += 1
        return num_hits / len(true_labels)

    def evaluate(self):

        return self._evaluate(self._true_labels, self._predicted_labels)


class MeanReciprocalRank(Metric):
    def __init__(
        self, true_labels, predicted_labels, is_print_confidence_interval=False
    ):
        super(MeanReciprocalRank, self).__init__()
        # Metric.__init__(self)
        self._name = self.__class__.__name__
        self._true_labels = true_labels
        self._predicted_labels = predicted_labels
        self._eval_function = self.mean_reciprocal_rank
        self._is_print_confidence_interval = is_print_confidence_interval
        self._n_samples = len(self._true_labels)

    @staticmethod
    def mean_reciprocal_rank(true_labels, predicted_labels):
        total_reciprocal_rank = 0
        for i in range(len(true_labels)):
            i_true = true_labels[i]
            i_preds = predicted_labels[i]
            if i_true in i_preds:
                true_rank = list(i_preds).index(i_true) + 1  # 1-indexed
                total_reciprocal_rank += 1 / true_rank
        return total_reciprocal_rank / len(true_labels)

    def evaluate(self):

        return self._evaluate(self._true_labels, self._predicted_labels)
