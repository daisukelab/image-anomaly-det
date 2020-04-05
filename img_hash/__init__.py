from dlcliche.utils import (load_pkl, save_as_pkl_binary, ensure_folder)
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
from base_ano_det import BaseAnoDet


class ImgHashAnoDet(BaseAnoDet):
    """Anomaly Detector by using Image Hashing."""

    def __init__(self, params):
        super().__init__(params)
        self.good_hashes = None
        self.hasher = (imagehash.phash if params.hash_algorithm == 'phash' else
                       imagehash.whash if params.hash_algorithm == 'whash' else
                       'unknown hash')
        self.hash_size = params.hash_size
        # work folder
        self.work = Path(params.work_folder)/params.project

    def build_good_hash(self, good_samples, cache=True):
        # build hash dictionary
        if cache and self._model_file().exists():
            self.load_model()
            a, b = np.array(list(self.good_hashes.values())), np.array(good_samples)
            hash_value_size = len(list(self.good_hashes.keys())[0].hash)
            if np.all(a == b) and hash_value_size == self.hash_size:
                print(' using cached', self._model_file())
                return # files are the same, and hash size is the same
        self.good_hashes = {}
        for f in good_samples:
            hash_value = self.hasher(Image.open(f), hash_size=self.hash_size)
            self.good_hashes[hash_value] = f
        # cache if needed
        if cache:
            self.save_model()
            print(' saved cache as', self._model_file())

    def train_model(self, train_samples, cache=True, *args, **kwargs):
        self.build_good_hash(train_samples, cache=cache)

    def _model_file(self, file_name=None):
        ensure_folder(self.work)
        if file_name == None:
            file_name = self.work/f'hashtable-{self.test_target}.pkl'
        return file_name

    def save_model(self, file_name=None):
        assert self.good_hashes is not None
        save_as_pkl_binary(self.good_hashes, self._model_file(file_name))

    def load_model(self, file_name=None):
        self.good_hashes = load_pkl(self._model_file(file_name))

    def predict(self, file_name):
        """Predict distance from prototype (good) samples.
        Returns:
            distance in range [0, 1]. Smaller if file is closer to any of prototype sample.
        """
        hash_value = self.hasher(Image.open(file_name), hash_size=self.hash_size)
        ref_hashes = list(self.good_hashes.keys())
        possible_max = hash_value.hash.shape[0] * hash_value.hash.shape[1]
        # distance := mean Manhattan distance from dictionary hashes.
        # `-` operation is implemented in imagehash module.
        distances = [(hash_value - ref) / possible_max for ref in ref_hashes]
        closest_idx = np.argmin(distances)
        #print(hash_value, key_list[0], distances[closest_idx], distances[closest_idx: closest_idx+5])
        return distances[closest_idx]

    def predict_test(self, test_samples, *args, **kwargs):
        return [self.predict(f) for f in test_samples]
