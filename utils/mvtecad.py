from dlcliche.utils import *
sys.path.append('..')
from base_ano_det import BaseAnoDet
from utils import preprocess_det_images


def evaluate_MVTecAD(data_root, ano_det_cls: BaseAnoDet, params, iteration=1,
                     test_targets=None, skip_preprocess=False, **kwargs):
    """Evaluate anomaly detectors for entire MVTecAD dataset.
    
    Args:
        data_root: Folder to your MVTecAD dataset copy.
        ano_det_cls: Anomaly detector class following BaseAnoDet.
        params: parameters for evaluation.
        test_targets: List of test target names or None to test all.
    """

    data_root = Path(data_root)

    # determine test targets
    if test_targets is None:
        test_targets = sorted([d.name for d in data_root.glob('*') if d.is_dir()])

    # result data frame
    result_df = pd.DataFrame()

    print('Evaluating:', test_targets)
    for test_target in test_targets:
        print(f'\n--- Start evaluating [{test_target}] ----')
        target_data = data_root/test_target
        raw_train_files = sorted(target_data.glob(f'train/good/*.png'))
        raw_test_files = sorted(target_data.glob(f'test/*/*.png'))
        test_labels = [f.parent.name for f in raw_test_files]
        test_y_trues = [label != 'good' for label in test_labels]

        # create detector
        det = ano_det_cls(params=params, **kwargs)

        # preprocess data
        print((' preprocessing...'))
        train_files = preprocess_det_images(params, det, raw_train_files,
            'train', skip_preprocess)
        test_files = preprocess_det_images(params, det, raw_test_files,
            'test', skip_preprocess)

        for exp in range(iteration):
            # prepare
            det.prepare_experiment(experiment_no=exp, test_target=test_target)

            # train
            print((' training...'))
            det.create_model()
            det.setup_train(train_files)
            det.train_model(train_files)

            # evaluate 
            print((' evaluating...'))
            det.setup_runtime(ref_samples=train_files)
            values = det.evaluate_test(test_files, test_y_trues, test_labels=test_labels)
            auc, pauc, norm_threshs, norm_factor, scores, raw_scores = values

            # store results
            contents = {'target': [test_target],
                'auc': [auc],
                'th_k_sigma': [norm_threshs[0]],
                'th_fpr': [norm_threshs[1]],
                'th_tpr': [norm_threshs[2]],
                'norm_factor': [norm_factor],
            }
            if pauc is not None:
                contents['pauc']  = [pauc]
            this_result = pd.DataFrame(contents)
            result_df = pd.concat([result_df, this_result], ignore_index=True)

            print(f'Results: {contents}')

    # write results
    result_df.to_csv('results.csv', index=False)
    print(('done.'))
    return result_df
