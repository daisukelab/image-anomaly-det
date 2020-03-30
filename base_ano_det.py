from sklearn import metrics


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

    def setup_train(self, train_samples):
        """(optional)

        Expected:
            Data conversion (data type, data format)
            Data caching
            Random seed setting
            Working folder setting
            Parameter updating
        """
        pass

    def train_model(self, train_samples, *args, **kwargs):
        raise Exception('Implement me if you call this.')

    def load_model(self, *args, **kwargs):
        pass

    def set_example(self, *args, **kwargs):
        raise Exception('Implement me if you call this.')

    def predict_test(self, test_samples, *args, **kwargs):
        pass

    def evaluate_test(self, test_samples, test_targets):
        scores = self.predict_test(test_samples)
        auc = metrics.roc_auc_score(test_targets, scores)
        pauc = metrics.roc_auc_score(test_targets, scores,
             max_fpr=self.params.max_fpr) if 'max_fpr' in self.params else None

        # TODO: optim threshold  calculation

        return auc, pauc, scores

    def test(samples, *args, **kwargs):
        """Test samples. (optional)

        Returns:
            result (list(bool)): True if positive or False.
            score (list or None): Score or distance if available, or None otherwise.
            heatmap (list or None): Heatmap array (np.uint8) if available, or None otherwise.
        """
        return [], None, None
