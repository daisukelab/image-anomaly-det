from dlcliche.utils import *
sys.path.append('..')
from base_ano_det import BaseAnoDet


def evaluate_MVTecAD(data_root, ano_det_cls: BaseAnoDet, params, iteration=1,
                     test_targets=None, **kwargs):
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
    results = pd.DataFrame()

    print('Evaluating:', test_targets)
    for test_target in test_targets:
        print(f'\n--- Start evaluating [{test_target}] ----')
        target_data = data_root/test_target
        train_files = sorted(target_data.glob(f'train/good/*.png'))
        test_files = sorted(target_data.glob(f'test/*/*.png'))
        test_y_trues = [f.parent.name != 'good' for f in test_files]

        for exp in range(iteration):

            # create detector
            det = ano_det_cls(params=params, **kwargs)

            # prepare
            det.prepare_experiment(experiment_no=exp, test_target=test_target)

            # preprocess data
            print((' preprocessing...'))
            det.setup_train(train_files)

            # train
            print((' training...'))
            det.train_model(train_files)

            # evaluate 
            print((' evaluating...'))
            auc, pauc, scores = det.evaluate_test(test_files, test_y_trues)

            # store results
            contents = {'target': [test_target], 'auc': [auc]}
            if pauc is not None:
                contents['pauc']  = [pauc]
            this_result = pd.DataFrame(contents)
            results = pd.concat([results, this_result], ignore_index=True)
            
            print(f'Results: {contents}')

    # write results
    results.to_csv('results.csv', index=False)
    print(('done.'))
    return results
