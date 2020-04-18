from sklearn import metrics
import numpy as np


def binary_clf_thresh_by_tpr(thresholds, tpr, min_tpr=1.0):
    """Determine threshold by TPR.

    Example:
        fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
        thresh = bin_clf_thresh_by_tpr(thresholds, tpr, min_tpr=your_min_tpr)
    """

    for th, rate in zip(thresholds, tpr):
        if min_tpr <= rate:
            return th
    assert tpr[-1] == 1.0, f'TPR should have 1.0 at the end, why?'
    return None


def binary_clf_thresh_by_fpr(thresholds, fpr, max_fpr=0.1):
    """Determine threshold by FPR.

    Example:
        fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
        thresh = det.thresh_by_fpr(thresholds, fpr, max_fpr=your_max_fpr)
    """

    for th, next_rate in zip(thresholds, fpr[1:]):
        if max_fpr <= next_rate:
            return th
    # Unfortunately or fortunately all threshold can be under the max_fpr.
    return thresholds[-1]


class BaseAnoDet(object):
    """Anomaly Detector Abstract Base Class."""
    
    def __init__(self, params=None):
        self.params = params
        self.experiment_no = None

    def prepare_experiment(self, experiment_no=None, test_target=None):
        if experiment_no is None:
            self.experiment_no = 0 if self.experiment_no is None else self.experiment_no + 1
        else:
            self.experiment_no = experiment_no
        self.test_target = test_target

    def setup_train(self, train_samples, **kwargs):
        """(optional) Prepare for training, expected actions:
            Data conversion (data type, data format)
            Data caching
            Random seed setting
            Working folder setting
            Parameter updating
        """
        pass

    def setup_runtime(self, model_weights, ref_samples):
        """(optional) Prepare dictionaries of reference samples for example."""
        pass

    def train_model(self, train_samples, **kwargs):
        raise Exception('Implement me if you call this.')

    def load_model(self, **kwargs):
        pass

    def predict_test(self, test_samples, return_raw=False, **kwargs):
        """Predict scores for test samples.
        Score can be either distance to closest reference samples,
        or normality score, or any metric value.
        Smaller value shows test sample is closer to reference sample.

        Args:
            test_samples: Test samples to predict score.
            return_raw: Return raw score matrix (test x ref) in addition to scores.

        Returns:
            Scores. **Not normalized.**
        """
        pass

    def normalize_score(self, scores):
        if 'distance_norm_factor' not in self.params:
            raise ValueError(f'Set params.distance_norm_factor to normalize score.')
        return scores / self.params.distance_norm_factor

    def evaluate_test(self, test_samples, test_y_trues, show_thresh=True):
        scores, raw_scores = self.predict_test(test_samples, return_raw=True)
        fpr, tpr, thresholds = metrics.roc_curve(test_y_trues, scores)
        auc = metrics.auc(fpr, tpr)
        pauc = metrics.roc_auc_score(test_y_trues, scores,
             max_fpr=self.params.max_fpr) if 'max_fpr' in self.params else None

        norm_threshs, norm_factor = self.calc_thresholds(fpr, tpr, thresholds, scores,
            show_thresh=show_thresh)

        return auc, pauc, norm_threshs, norm_factor, scores, raw_scores

    def calc_thresholds(self, fpr, tpr, thresholds, scores, show_thresh):
        """Calculates normalizaion factor and normalized thresholds for 3 policies.

        Returns (([], float)): Normalized thresholds explained below, and norm_factor. 
            [threshold that covers k-sigma (usually 2 sigma to cover 95%),
             threshold FPR based policy,
             threshold TPR based policy]
        """
        assert 'max_fpr' in self.params, 'Set params.max_fpr.'
        assert 'min_tpr' in self.params, 'Set params.min_tpr.'
        assert 'sigma_k' in self.params, 'Set params.sigma_k.'
        mean_score, sigma_score = np.mean(scores), np.std(scores)
        norm_factor = np.max([mean_score, np.finfo(float).eps])
        threshs = np.array([mean_score + self.params.sigma_k*sigma_score, # coverage: 2 sigma = 95%, 3 sigma = 99.7%
                 binary_clf_thresh_by_fpr(thresholds, fpr, max_fpr=self.params.max_fpr), # coverage: max_fpt=0.1 = ?%
                 binary_clf_thresh_by_tpr(thresholds, tpr, min_tpr=self.params.min_tpr)]) # coverage: min_tpr=1.0 = 100%
        norm_threshs = threshs / norm_factor

        if show_thresh:
            print(f'distance_threshold: {norm_threshs[0]} # Threshold k-sigma [{self.params.sigma_k}]')
            print(f'distance_threshold: {norm_threshs[1]} # Threshold max FPR [{self.params.max_fpr}]')
            print(f'distance_threshold: {norm_threshs[2]} # Threshold min TPR [{self.params.min_tpr}]')
            print(f'distance_norm_factor: {norm_factor}')

        return norm_threshs, norm_factor
