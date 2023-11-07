import numpy as np
import torch as th
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def shuffle(X, y):
    num_samples = len(X)
    random_indexes = np.random.permutation(num_samples)
    X = X[random_indexes]
    y = y[random_indexes]

    return X, y


def generate_synthetic_data(n_train_data, generators, ks, params):
    profiles = {}
    for k in ks:
        n_batch = int(n_train_data[k] / 50) + 1
        profiles_nonzero = generators[k](
            th.randn(n_batch, 1)
        ).view(-1, params[k]['n_profile']).detach().numpy()[:n_train_data[k]]
        profiles[k] = np.zeros((n_train_data[k], params[k]['n']))
        profiles[k][:, ~params[k]['zero_values']] = profiles_nonzero

    return profiles


def predict_clusters(generators, vals_k, n_train_data, n_test_data, n_repeats, params, ks):
    setups = {
        'baseline': {
            'train_data': 'real_train',
            'test_data': 'real_test',
        },
        'TSTR': {
            'train_data': 'synthetic',
            'test_data': 'real_test'
        },
        'TRTS': {
            'train_data': 'real_test',
            'test_data': 'synthetic',
        },
    }
    for setup_label, setup in setups.items():
        print(f"Setup: {setup_label}")
        scores = []
        for _ in range(n_repeats):
            profiles_train = {}
            profiles_test = {}
            if setup['train_data'] == 'real_train':
                for k in ks:
                    profiles_train[k] = vals_k[k][: n_train_data[k]]
            elif setup['train_data'] == 'real_test':
                for k in ks:
                    profiles_train[k] = vals_k[k][n_train_data[k]:]
            elif setup['train_data'] == 'synthetic':
                profiles_train = generate_synthetic_data(
                    n_train_data, generators, list(vals_k.keys()), params
                )

            X_train = np.concatenate([profiles_train[k] for k in ks])
            y_train = np.concatenate([np.full(len(profiles_train[k]), k) for k in ks])

            if setup['test_data'] == 'real_test':
                for k in ks:
                    profiles_test[k] = vals_k[k][n_train_data[k]:]
            elif setup['test_data'] == 'synthetic':
                profiles_test = generate_synthetic_data(
                    n_test_data, generators, list(vals_k.keys()), params
                )

            X_test = np.concatenate([profiles_test[k] for k in ks])
            y_test = np.concatenate([np.full(len(profiles_test[k]), k) for k in ks])

            # Shuffle the data
            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test = shuffle(X_test, y_test)

            # Train the classifier
            classifier = RandomForestClassifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)

        print(f"Average accuracy {setup_label} over {n_repeats}: {np.mean(scores)}")


def discriminate_real_synthetic(
        vals_k, generators, n_train_data, n_test_data, n_repeats, params, ks
):
    for k in ks:
        scores = []
        for _ in range(n_repeats):
            synthetic_data_train = generate_synthetic_data(n_train_data, generators, [k], params)[k]
            X_train = np.concatenate([vals_k[k][:n_train_data[k]], synthetic_data_train])
            y_train = np.concatenate([np.full(n_train_data[k], 1), np.full(n_train_data[k], 0)])
            X_train, y_train = shuffle(X_train, y_train)

            synthetic_data_test = generate_synthetic_data(n_test_data, generators, [k], params)[k]
            X_test = np.concatenate([vals_k[k][n_train_data[k]:], synthetic_data_test])
            y_test = np.concatenate([np.full(n_test_data[k], 1), np.full(n_test_data[k], 0)])
            X_test, y_test = shuffle(X_test, y_test)

            classifier = RandomForestClassifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
        print(
            f"Average accuracy discriminating synthetic profiles in cluster {k}"
            f"over {n_repeats} repeats: {np.mean(scores)}"
        )


def test_GANs(generators, vals_k, params, train_set_size, ks=None):
    n_repeats = 10
    if ks is None:
        ks = list(vals_k.keys())
    n_train_data = {k: int(len(vals_k[k]) * train_set_size) for k in ks}
    n_test_data = {k: len(vals_k[k]) - n_train_data[k] for k in ks}
    if len(ks) > 1:
        predict_clusters(generators, vals_k, n_train_data, n_test_data, n_repeats, params, ks=ks)
    discriminate_real_synthetic(
        vals_k, generators, n_train_data, n_test_data, n_repeats, params, ks=ks
    )
