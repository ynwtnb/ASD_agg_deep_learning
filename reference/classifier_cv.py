import numpy as np
import math
from numpy import interp
from sklearn import metrics, model_selection
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn import preprocessing


def classify_instance_cv(clf_obj, clf_par, instances_array, labels_array, cv_folds, cv_reps, o_perform_pca, pca_obj_dict,
                         o_normalize_data, norm_constants_dict, sbjID, o_get_pd_for_diff_agg_intensity=False,
                         dict_of_superposition_lists=None):
    print("ID " + sbjID)
    tprs = []
    aucs = []
    # coefs = []
    mean_fpr = np.linspace(0, 1, 300)
    labels = labels_array.ravel()

    binary_labels = np.copy(labels)
    if o_get_pd_for_diff_agg_intensity:
        binary_labels[binary_labels > 0] = 1
        pds = []
        scores = []
        labs = []

    for repCount in range(cv_reps):
        # cv = model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=repCount)
        cv = model_selection.KFold(n_splits=cv_folds, shuffle=False, random_state=repCount)
        for train_idx, test_idx in cv.split(instances_array, binary_labels):
            if dict_of_superposition_lists is not None:
                train_idx = remove_overlapping_indexes(train_idx, test_idx, dict_of_superposition_lists)
                print("Training with " + str(len(train_idx)) + " samples.")
            train_instances = instances_array[train_idx]
            test_instances = instances_array[test_idx]

            if np.sum(binary_labels[train_idx]) == 0 or np.sum(binary_labels[test_idx]) == 0:
                print("Training or Testing samples have only class 0, aborting cv split.")
                continue

            if o_normalize_data:
                # norm_constants = get_normalization_constants(train_instances)
                train_instances = normalize_data(train_instances, norm_constants_dict[sbjID])
                test_instances = normalize_data(test_instances, norm_constants_dict[sbjID])

            if o_perform_pca:
                # pca_obj.fit(train_instances)
                train_instances = pca_obj_dict[sbjID].transform(train_instances)
                test_instances = pca_obj_dict[sbjID].transform(test_instances)

            classifier = clf_obj(**clf_par)
            model = classifier.fit(train_instances, binary_labels[train_idx])
            probas_ = model.predict_proba(test_instances)
            # coefs.append(model.coef_)
            fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test_idx], probas_[:, 1])

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            if math.isnan(roc_auc):
                print('!!!!!!NAN!!!!!!!!', roc_auc)

            # cv_roc_aucs.append(roc_auc)
            aucs.append(roc_auc)

            if o_get_pd_for_diff_agg_intensity:
                fpr_d = dict()
                tpr_d = dict()
                auc_d = dict()
                n_class = len(np.unique(y))
                n_positive_classes = n_class -1
                for c in range(n_positive_classes):
                    temp_labels = np.copy(labels)
                    # all positive classes > c+1
                    agg_idx = np.where(temp_labels >= (c+1))[0]

                    # make all agg episodes in agg_idx have a single label
                    pos_label = 10
                    temp_labels[agg_idx] = pos_label
                    # compute roc
                    fpr, tpr, _ = metrics.roc_curve(temp_labels[test_idx], probas_[:, 1], pos_label=pos_label)
                    fpr_d[c] = interp(mean_fpr, fpr, tpr)
                    auc_d[c] = metrics.auc(fpr, tpr)

                tprs.append(tpr_d)
                aucs.append(auc_d)


                # scores.append(probas_[:, 1].reshape(probas_.shape[0], 1))
                # labs.append(labels[test].reshape(len(labels[test]), 1))
                # pds.append(compute_pd_for_diff_agg_intensity(probas_[:, 1], thresholds, labels[test]))
                # pds.append(compute_pd_for_diff_agg_intensity(probas_[:, 1], np.linspace(0.2, 0.5, 10000), labels[test]))
                # pds += compute_pd_for_diff_agg_intensity(probas_[:, 1], np.linspace(0, 1, 500), labels[test])

    mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if o_get_pd_for_diff_agg_intensity:
        scores = np.vstack(scores)
        labs = np.vstack(labs)


        pds = compute_pd_for_diff_agg_intensity(scores, np.linspace(min(scores), max(scores), 10000), labs)
        return [mean_fpr, mean_tpr, mean_auc, std_auc, tprs, sbjID, pds]
    # return mean_fpr, mean_tpr, mean_auc, std_auc, tprs
    return [mean_fpr, mean_tpr, mean_auc, std_auc, tprs, sbjID]


def classify_instance_cv_with_agg_intensity(classifier, instances_array, labels_array, cv_folds, cv_reps, o_perform_pca,
                                            pca_obj_dict, o_normalize_data, norm_constants_dict, sbjID,
                                            o_get_pd_for_diff_agg_intensity=False):

    tprs = []
    aucs = []
    # coefs = []
    mean_fpr = np.linspace(0, 1, 300)
    labels = labels_array.ravel()

    binary_labels = np.copy(labels)

    binary_labels[binary_labels > 0] = 1
    pds = []
    scores = []
    labs = []

    for repCount in range(cv_reps):
        cv = model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=repCount)
        # for train, test in cv.split(instances_array, binary_labels):
        for train, test in cv.split(instances_array, labels):
            train_instances = instances_array[train]
            test_instances = instances_array[test]

            if o_normalize_data:
                # norm_constants = get_normalization_constants(train_instances)
                train_instances = normalize_data(train_instances, norm_constants_dict[sbjID])
                test_instances = normalize_data(test_instances, norm_constants_dict[sbjID])

            if o_perform_pca:
                # pca_obj.fit(train_instances)
                train_instances = pca_obj_dict[sbjID].transform(train_instances)
                test_instances = pca_obj_dict[sbjID].transform(test_instances)

            model = classifier.fit(train_instances, binary_labels[train])
            probas_ = model.predict_proba(test_instances)

            tpr_d = dict()
            auc_d = dict()
            n_class = len(np.unique(labels))
            n_positive_classes = n_class - 1
            fpr = []
            for c in range(n_positive_classes):
                temp_labels = np.copy(labels)
                # all positive classes > c+1
                agg_idx = np.where(temp_labels >= (c + 1))[0]

                # make all agg episodes in agg_idx have a single label
                pos_label = 10
                temp_labels[agg_idx] = pos_label
                # compute roc
                if c == 0:
                    fpr, tpr, thresholds = metrics.roc_curve(temp_labels[test], probas_[:, 1], pos_label=pos_label)
                    temp_labels = np.copy(labels)
                    # all positive classes > c+1
                    agg_idx = np.where(temp_labels == (c + 1))[0]

                    # make all agg episodes in agg_idx have a single label
                    pos_label = 10
                    temp_labels[agg_idx] = pos_label
                    fpr, tpr = t_roc(temp_labels[test], probas_[:, 1], thresholds, pos_label=pos_label, neg_label=-1)
                if c > 0:
                    temp_labels = np.copy(labels)
                    # all positive classes > c+1
                    agg_idx = np.where(temp_labels == (c + 1))[0]

                    # make all agg episodes in agg_idx have a single label
                    pos_label = 10
                    temp_labels[agg_idx] = pos_label
                    fpr, tpr = t_roc(temp_labels[test], probas_[:, 1], thresholds, pos_label=pos_label, neg_label=-1)
                tpr_d[c] = interp(mean_fpr, fpr, tpr)
                auc_d[c] = metrics.auc(fpr, tpr)

            tprs.append(tpr_d)
            aucs.append(auc_d)

    return [tprs, mean_fpr, aucs, sbjID]


def classify_instance_parallel_cv(classifier, instances_array, labels_array, cv_folds, cv_reps, n_jobs,
                                  o_perform_pca, n_pcs, o_normalize_data, o_get_pd_for_diff_agg_intensity=False,
                                  dict_of_superposition_lists=None):
    if dict_of_superposition_lists is not None:
        ll = []
        for key in dict_of_superposition_lists.keys():
            for l in dict_of_superposition_lists[key]:
                ll.append(l)
        dict_of_superposition_lists = ll

    tprs = []
    aucs = []
    # coefs = []
    mean_fpr = np.linspace(0, 1, 300)

    if o_get_pd_for_diff_agg_intensity:
        labels = labels_array.copy().ravel()
        labels[labels > 0] = 1
    else:
        labels = labels_array.ravel()

    result_list = Parallel(n_jobs=n_jobs)(
        delayed(run_one_cv_iteration)(classifier, instances_array, labels, cv_folds, rep_count, o_perform_pca,
                                      n_pcs, o_normalize_data, o_get_pd_for_diff_agg_intensity,
                                      dict_of_superposition_lists) for rep_count in range(cv_reps))
    # result_list = []
    # for rep_count in range(cv_reps):
    #     result_list.append(run_one_cv_iteration(classifier, instances_array, labels, cv_folds, rep_count, o_perform_pca,
    #                                   n_pcs, o_normalize_data, o_get_pd_for_diff_agg_intensity,
    #                                   dict_of_superposition_lists))

    for res in result_list:
        tprs += res[0]
        aucs += res[1]

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # return mean_fpr, mean_tpr, mean_auc, std_auc, tprs
    return [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]


def run_one_cv_iteration(classifier, instances_array, labels, cv_folds, rep_count, o_perform_pca, n_pcs,
                         o_normalize_data, o_get_pd_for_diff_agg_intensity, dict_of_superposition_lists=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 300)
    # cv = model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rep_count)
    cv = model_selection.KFold(n_splits=cv_folds, shuffle=False, random_state=rep_count)
    binary_labels = np.copy(labels)
    if o_get_pd_for_diff_agg_intensity:
        binary_labels[binary_labels > 0] = 1
        pds = []

    for train_idx, test_idx in cv.split(instances_array, binary_labels):
        # remove overlapping features from training set
        if dict_of_superposition_lists is not None:
            train_idx = remove_overlapping_indexes(train_idx, test_idx, dict_of_superposition_lists)
            print("Training with " + str(len(train_idx)) + " samples.")

        train_instances = instances_array[train_idx]
        test_instances = instances_array[test_idx]
        if np.sum(binary_labels[test_idx]) == 0:
            print("Aborting CV split. No behavioral annotation in the test set.")

        if o_normalize_data:
            norm_constants = get_normalization_constants(train_instances)
            train_instances = normalize_data(train_instances, norm_constants)
            test_instances = normalize_data(test_instances, norm_constants)

        if o_perform_pca:
            pca = PCA(n_components=n_pcs)
            pca.fit(train_instances)
            train_instances = pca.transform(train_instances)
            test_instances = pca.transform(test_instances)

        model = classifier.fit(train_instances, binary_labels[train_idx])
        probas_ = model.predict_proba(test_instances)
        # coefs.append(model.coef_)
        fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test_idx], probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        if math.isnan(roc_auc):
            print('!!!!!!NAN!!!!!!!!', roc_auc)

        # cv_roc_aucs.append(roc_auc)
        aucs.append(roc_auc)

        if o_get_pd_for_diff_agg_intensity:
            pds += compute_pd_for_diff_agg_intensity(probas_[:, 1], thresholds, labels[test_idx])
            # pds += compute_pd_for_diff_agg_intensity(probas_[:, 1], np.linspace(0, 1, 500), labels[test])
    if o_get_pd_for_diff_agg_intensity:
        return [tprs, aucs, pds]
    return [tprs, aucs]


def remove_overlapping_indexes(train_idx, test_idx, list_of_superposition_lists):
    """
    Remove indexes from train_idx corresponding to feature vectors overlapping the ones indexed in test_idx
    :param train_idx: numpy array of indexes
    :param test_idx: numpy array of indexes
    :param list_of_superposition_lists: list whose each element is a list [min_idx, max_idx] with the minimum and
    maximum offset of indexes of overlapping feature vectors, such that vectors with index i - min_idx to i + max_idx
    overlaps with vector i.
    :return: train_idx without the index of overlapping elements.
    """
    train_idx = np.copy(train_idx)
    for i in test_idx:
        min_idx = list_of_superposition_lists[i][0]
        max_idx = list_of_superposition_lists[i][1]
        # print(str(min_idx) + " " + str(max_idx))
        # for j in range(min(0, i - min_idx), max(i + max_idx)):
        for j in range(i - min_idx, i + max_idx):
            train_idx[train_idx == j] = -100
    train_idx = train_idx[train_idx != -100]
    return train_idx


def compute_pd_for_diff_agg_intensity(scores, thresholds, labels):
    """
    Compute the probability of detection 'pd' for all threshold values and different aggression intensity labels.
    :param scores: the classification score
    :param thresholds: the threshold values
    :param labels: labels (non-zero positive values mean different intensity of aggression)
    :return: a matrix pds with size len(thresholds) X number of different aggression labels with the pd for each
    combination
    """
    # n_unique_labels = len(np.unique(labels))
    n_unique_labels = 4
    agg_indexes = np.where(labels >= 1)[0]
    nd = 0
    n = len(scores)
    pds = np.zeros((len(thresholds), n_unique_labels-1))
    count = 0
    for t in thresholds:
        # for mild, moderate and high intensity labels
        for j in range(1, n_unique_labels):
            # scores of labels >= j
            agg_indexes = np.argwhere(labels >= j)
            n = len(agg_indexes)
            # compute number of detections within each intensity
            s_j = scores[agg_indexes]
            # nd_j = len(np.where(s_j > t)[0])
            nd_j = len(s_j[s_j > t])
            # pd for each intensity label
            if n > 0:
                pds[count, j-1] = nd_j/n
            else:
                pds[count, j - 1] = np.nan
        count += 1

    return pds


def t_roc(y_true, y_score, thresholds, pos_label, neg_label=-1):
    n_d = np.zeros(thresholds.shape)
    n_fa = np.zeros(thresholds.shape)
    n_h0_class = len(y_true[y_true == neg_label])
    n_h1_class = len(y_true[y_true == pos_label])
    # if
    count = 0
    for t in thresholds:
        for i in range(len(y_score)):
            if y_score[i] >= t:
                if y_true[i] == pos_label:
                    n_d[count] += 1
                elif y_true[i] == neg_label:
                    n_fa[count] += 1
        count += 1
    pfa = n_fa / n_h0_class
    pd = n_d / n_h1_class

    return pfa, pd


def run_backend(subj_id, classifier, dict_of_instances_arrays, dict_of_labels_arrays, cv_folds, cv_reps):
    mean_fpr, mean_tpr, mean_auc, std_auc, tprs = classifier_cv.classify_instance_cv(classifier,
                                                                                     dict_of_instances_arrays[subj_id],
                                                                                     dict_of_labels_arrays[subj_id],
                                                                                     cv_folds, cv_reps)


def get_normalization_constants(data_array, axis=0):
    means = np.mean(data_array, axis=axis)
    stds = np.std(data_array, axis=axis)
    stds[stds == 0] = 1
    # return np.linalg.norm(data_array, axis=axis)
    return [means, stds]


def normalize_data(data_array, normalization_constants, axis=0):
    # removing zeros and avoiding a division by zero.
    # normalization_constants[normalization_constants == 0] = 1
    # if axis == 1:
    #     return (data_array.transpose() / normalization_constants).transpose()
    #
    # return data_array / normalization_constants

    means = normalization_constants[0]
    stds = normalization_constants[1]
    if axis == 1:
        return ((data_array.transpose() - means) / stds).transpose()

    return (data_array - means)/(stds)


from sklearn.mixture import GaussianMixture
class GMMClassifier:

    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        self.clf_par = {'n_components': n_components, 'covariance_type': covariance_type, 'tol': tol,
                        'reg_covar': reg_covar, 'max_iter': max_iter, 'n_init': n_init, 'init_params': init_params,
                        'weights_init': weights_init, 'means_init': means_init, 'precisions_init': precisions_init,
                        'random_state': random_state, 'warm_start': warm_start, 'verbose': verbose,
                        'verbose_interval': verbose_interval}

        self.positive_model = GaussianMixture(**self.clf_par)
        self.negative_model = GaussianMixture(**self.clf_par)

    def fit(self, X, y):
        X_neg = X[y == 0]
        X_pos = X[y >= 1]
        self.negative_model.fit(X_neg)
        self.positive_model.fit(X_pos)
        return self

    def predict_proba(self, X):
        # return np.log(self.positive_model.predict(X)) - np.log(self.negative_model.predict(X))
        p1 = self.positive_model.score_samples(X) - self.negative_model.score_samples(X)
        p0 = self.negative_model.score_samples(X) - self.positive_model.score_samples(X)
        p = np.concatenate((p0.reshape(-1, 1), p1.reshape(-1, 1)), axis=1)
        return p
