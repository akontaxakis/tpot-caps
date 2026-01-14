from AutoML_data_manager.data_manager import DataManager
from tpot.config import classifier_config_dict

if __name__ == '__main__':
    from tpot import TPOTClassifier

    # CAPS parameters
    data_id = "jannis"
    mode = "CAPS"  # CAPS specific parameter
    sel_algo = "caps_greedy"  # CAPS specific parameter
    lamda = 0.5  # CAPS specific parameter
    selection = 50  # CAPS specific parameter

    # DataManager setup
    datasets = [data_id]
    iris = DataManager(datasets[0], r'datasets',
                           replace_missing=True,
                           verbose=3)
    X = iris.data['X_train']
    y = iris.data['Y_train']

    seed = 7777
    # TPOT-CAPS setup
    tpot = TPOTClassifier(mode=mode,sel_algo=sel_algo,lamda=lamda, selection = selection, data_id=data_id, population_size=100, generations=10, cv=2, verbosity=3,
                              mutation_rate=0.9,
                              crossover_rate=0.1, n_jobs=1, template='Transformer-Transformer-Classifier',
                              random_state=seed, config_dict=classifier_config_dict)

    tpot.fit(X, y)