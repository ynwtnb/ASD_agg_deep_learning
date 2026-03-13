
import featGen
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from numpy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import argparse
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import classifier_cv
import pickle
import os
from joblib import Parallel, delayed
import joblib

def data_preprocess(data_path, path_style='/', num_observation_frames=12, num_prediction_frames=4, feat_code=6,
                    o_return_list_of_sessions=False, o_is_new_dataset=True):
    # Load data
    agg_cat = ['aggression', 'agg', 'aggression (?)', 'property destruction', 'noncompliance',
                   'SIB', 'ED', 'kicked table', 'aggression/SIB', 'elopement/drop on the floor',
                   'aggression/SIB/property destruction', 'dropped on floor', 'sib', 'elopement',
                   'AGG', 'Agg', 'Agg/SIB']
    non_agg_cat = ['sleeping', 'mild ED/shoved by peer']

    bin_size = '15S'  # Frame size in seconds

    data_dict, uid_dict = featGen.feature_extraction(data_path, bin_size, agg_cat, non_agg_cat, feat_code, path_style=path_style,
                                       o_is_new_dataset=o_is_new_dataset, o_run_from_scracth=False, o_multiclass=True)

    dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, dict_of_superposition_lists = \
        featGen.gen_instances_from_raw_feat_dictionary(data_dict, num_observation_frames, num_prediction_frames, feat_code,
                                                   agg_intensity_clf=None, o_is_new_dataset=o_is_new_dataset,
                                                   o_multiclass=True,
                                                   o_return_list_of_sessions=o_return_list_of_sessions,
                                                   outdir=data_path, o_run_from_scracth=False)

    # remove blacklisted IDs from IDsdict
    if len(id_blacklist) != 0:
        for v in id_blacklist:
            for k in uid_dict.keys():
                if uid_dict[k] == v:
                    blacklisted_key = k
            del uid_dict[blacklisted_key]

    return dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists


def run_loso_individual_classification(X_dict, y_dict, n_classes=4, cv_reps=3, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                  clf_obj=svm.SVC, clf_par=None, output_file_name='out.data', min_sessions=10):

    keys = y_dict.keys()
    # n_classes = len(np.unique(y))

    results = {}
    # for key in y_dict.keys():
    # for key in ['3174']:
    for key in ['1233']:
        X_list = X_dict[key]
        y_list = y_dict[key]

        n_sessions = len(y_list)
        # Need at least 2 sessions.
        if n_sessions < min_sessions:
            print('n_sessions < ' + str(min_sessions) + ' aborting subj #' + str(key))
            continue
        # print("n_classes")
        # print(np.unique(y))
        # # y = label_binarize(y, classes=np.unique(y))
        # y = label_binarize(y, classes=np.arange(n_classes).astype(float))
        # # cv_reps = 3

        # Normalization and PCA should be performed here with data from other subjects.
        # returns a list of lists (lols)
        other_lols = [array for (kk, array) in zip(X_dict.keys(), X_dict.values()) if kk != key]
        ll = []
        for l in other_lols:
            if len(l) > 1:
                ll += [np.concatenate(l, axis=0)]
            else:
                ll += l

        # X_other = np.concatenate([np.concatenate(l, axis=0) for l in other_lols], axis=0)
        # X_other = np.concatenate([np.concatenate(l, axis=0) for l in
        #                           [array for (kk, array) in zip(X_dict.keys(), X_dict.values()) if kk != key]], axis=0)
        X_other = np.concatenate(ll, axis=0)

        if o_normalize_data:
            norm_constants = classifier_cv.get_normalization_constants(X_other)

        if o_perform_pca:
            pca = PCA(n_components=n_pcs)
            pca.fit(X_other)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i] = []
        #     # tpr[i] = []
        #     roc_auc[i] = []
        #
        # fpr["micro"] = []
        # tpr["micro"] = []
        # roc_auc["micro"] = []
        #
        # fpr["macro"] = []
        # tpr["macro"] = []
        # roc_auc["macro"] = []

        mean_fpr = np.linspace(0, 1, 300)

        all_y = np.zeros((0, n_classes))
        all_y_scores = np.zeros((0, n_classes))

        temp_X = np.arange(n_sessions)
        kf = KFold(n_splits=cv_reps)
        # cv over different splits
        for train_index, test_index in kf.split(temp_X):
            X_train = np.concatenate([X_list[i] for i in train_index], axis=0)
            X_test = np.concatenate([X_list[i] for i in test_index], axis=0)
            y_train = np.concatenate([y_list[i] for i in train_index], axis=0)
            y_test = np.concatenate([y_list[i] for i in test_index], axis=0)
        # cv over different sessions
        # for rep in range(n_sessions):
        #     X_train = np.concatenate([X_list[i] for i in range(n_sessions) if i != rep], axis=0)
        #     X_test = X_list[rep]
        #     y_train = np.concatenate([y_list[i] for i in range(n_sessions) if i != rep], axis=0)
        #     y_test = y_list[rep]

            y_train = label_binarize(y_train, classes=np.arange(n_classes).astype(float))
            y_test = label_binarize(y_test, classes=np.arange(n_classes).astype(float))

            if o_normalize_data:
                # norm_constants = classifier_cv.get_normalization_constants(X_train)
                X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
                X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()

            if o_perform_pca:
                # pca = PCA(n_components=n_pcs)
                # pca.fit(X_train)
                X_train = pca.transform(X_train).copy()
                X_test = pca.transform(X_test).copy()

                # create and train classifiers
            # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
            clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

            # test classifiers
            y_score = clf.decision_function(X_test)

            all_y = np.concatenate((all_y, y_test), axis=0)
            all_y_scores = np.concatenate((all_y_scores, y_score), axis=0)

            # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])
        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            # roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_, tpr_, _ = roc_curve(all_y[:, i], all_y_scores[:, i])
            roc_auc_ = auc(fpr_, tpr_)
            fpr[i] = fpr_
            # tpr[i].append(interp(mean_fpr, fpr_, tpr_))
            tpr[i] = interp(mean_fpr, fpr_, tpr_)
            roc_auc[i] = roc_auc_

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_score.ravel())
        fpr["micro"] = fpr_
        tpr["micro"] = interp(mean_fpr, fpr_, tpr_)
        roc_auc["micro"] = auc(fpr_, tpr_)

        # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(n_classes):
            # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
            mean_tpr += tpr[i].ravel()

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        # fpr["macro"].append(all_fpr)
        fpr["macro"] = mean_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(mean_fpr, tpr["macro"])

    # printing results
    print(roc_auc)
    # saving results.
    results[key] = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]

    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)



# def run_loso_individual_classification(X_dict, y_dict, n_classes=4, cv_reps=3, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
#                                   clf_obj=svm.SVC, clf_par=None, output_file_name='out.data', min_sessions=2):
#
#     keys = y_dict.keys()
#     # n_classes = len(np.unique(y))
#
#     results = {}
#     for key in y_dict.keys():
#         X_list = X_dict[key]
#         y_list = y_dict[key]
#
#         n_sessions = len(y_list)
#         # Need at least 2 sessions.
#         if n_sessions < min_sessions:
#             print('n_sessions < ' + str(min_sessions) + ' aborting subj #' + str(key))
#             continue
#         # print("n_classes")
#         # print(np.unique(y))
#         # # y = label_binarize(y, classes=np.unique(y))
#         # y = label_binarize(y, classes=np.arange(n_classes).astype(float))
#         # # cv_reps = 3
#
#         # Normalization and PCA should be performed here with data from other subjects.
#         # returns a list of lists (lols)
#         other_lols = [array for (kk, array) in zip(X_dict.keys(), X_dict.values()) if kk != key]
#         ll = []
#         for l in other_lols:
#             if len(l) > 1:
#                 ll += [np.concatenate(l, axis=0)]
#             else:
#                 ll += l
#
#         # X_other = np.concatenate([np.concatenate(l, axis=0) for l in other_lols], axis=0)
#         # X_other = np.concatenate([np.concatenate(l, axis=0) for l in
#         #                           [array for (kk, array) in zip(X_dict.keys(), X_dict.values()) if kk != key]], axis=0)
#         X_other = np.concatenate(ll, axis=0)
#
#         if o_normalize_data:
#             norm_constants = classifier_cv.get_normalization_constants(X_other)
#
#         if o_perform_pca:
#             pca = PCA(n_components=n_pcs)
#             pca.fit(X_other)
#
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         for i in range(n_classes):
#             fpr[i] = []
#             tpr[i] = []
#             roc_auc[i] = []
#
#         fpr["micro"] = []
#         tpr["micro"] = []
#         roc_auc["micro"] = []
#
#         fpr["macro"] = []
#         tpr["macro"] = []
#         roc_auc["macro"] = []
#
#         mean_fpr = np.linspace(0, 1, 300)
#
#         # cv over different sessions
#         for rep in range(n_sessions):
#             X_train = np.concatenate([X_list[i] for i in range(n_sessions) if i != rep], axis=0)
#             X_test = X_list[rep]
#             y_train = np.concatenate([y_list[i] for i in range(n_sessions) if i != rep], axis=0)
#             y_test = y_list[rep]
#
#             y_train = label_binarize(y_train, classes=np.arange(n_classes).astype(float))
#             y_test = label_binarize(y_test, classes=np.arange(n_classes).astype(float))
#
#             if o_normalize_data:
#                 # norm_constants = classifier_cv.get_normalization_constants(X_train)
#                 X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
#                 X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()
#
#             if o_perform_pca:
#                 # pca = PCA(n_components=n_pcs)
#                 # pca.fit(X_train)
#                 X_train = pca.transform(X_train).copy()
#                 X_test = pca.transform(X_test).copy()
#
#                 # create and train classifiers
#             # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
#             clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)
#
#             # test classifiers
#             y_score = clf.decision_function(X_test)
#             # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])
#
#             for i in range(n_classes):
#                 # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#                 # roc_auc[i] = auc(fpr[i], tpr[i])
#                 fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
#                 roc_auc_ = auc(fpr_, tpr_)
#                 fpr[i].append(fpr_)
#                 tpr[i].append(interp(mean_fpr, fpr_, tpr_))
#                 roc_auc[i].append(roc_auc_)
#
#             # Compute micro-average ROC curve and ROC area
#             # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#             # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#             fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_score.ravel())
#             roc_auc_ = auc(fpr_, tpr_)
#             fpr["micro"].append(fpr_)
#             tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
#             roc_auc["micro"].append(auc(fpr_, tpr_))
#
#             # First aggregate all false positive rates
#             all_fpr = np.unique(np.concatenate([fpr[i][rep] for i in range(n_classes)]))
#
#             # Then interpolate all ROC curves at this points
#             mean_tpr = np.zeros_like(mean_fpr)
#             for i in range(n_classes):
#                 # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
#                 mean_tpr += tpr[i][rep]
#
#             # Finally average it and compute AUC
#             mean_tpr /= n_classes
#
#             fpr["macro"].append(all_fpr)
#             tpr["macro"].append(mean_tpr)
#             roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][rep]))
#
#         # saving results.
#         results[key] = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]
#
#     with open(output_file_name, 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(results, filehandle)


def run_individual_classification(X_dict, y_dict, n_classes=4, cv_reps=3, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                  clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):

    keys = y_dict.keys()
    # n_classes = len(np.unique(y))

    results = {}
    for key in y_dict.keys():
        print("Learning individual models for id = {}".format(key))

        X = X_dict[key]
        y = y_dict[key]
        if len(y) < 1:
            continue

        print("classes in y = [" + str(np.unique(y)) + "]")

        # y = label_binarize(y, classes=np.unique(y))
        y = label_binarize(y, classes=np.arange(n_classes).astype(float))
        # cv_reps = 3

        # Normalization and PCA should be performed here with data from other subjects.
        X_other = np.concatenate([array for (kk, array) in zip(X_dict.keys(), X_dict.values()) if kk != key], axis=0)
        if o_normalize_data:
            norm_constants = classifier_cv.get_normalization_constants(X_other)

        if o_perform_pca:
            pca = PCA(n_components=n_pcs)
            pca.fit(X_other)


        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i] = []
            tpr[i] = []
            roc_auc[i] = []

        fpr["micro"] = []
        tpr["micro"] = []
        roc_auc["micro"] = []

        fpr["macro"] = []
        tpr["macro"] = []
        roc_auc["macro"] = []

        mean_fpr = np.linspace(0, 1, 300)

        for rep in range(cv_reps):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rep, shuffle=True)

            if o_normalize_data:
                # norm_constants = classifier_cv.get_normalization_constants(X_train)
                X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
                X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()

            if o_perform_pca:
                # pca = PCA(n_components=n_pcs)
                # pca.fit(X_train)
                X_train = pca.transform(X_train).copy()
                X_test = pca.transform(X_test).copy()

                # create and train classifiers
            # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
            clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

            # test classifiers
            y_score = clf.decision_function(X_test)
            # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])

            for i in range(n_classes):
                # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                # roc_auc[i] = auc(fpr[i], tpr[i])
                fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc_ = auc(fpr_, tpr_)
                fpr[i].append(fpr_)
                tpr[i].append(interp(mean_fpr, fpr_, tpr_))
                roc_auc[i].append(roc_auc_)

            # Compute micro-average ROC curve and ROC area
            # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc_ = auc(fpr_, tpr_)
            fpr["micro"].append(fpr_)
            tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
            roc_auc["micro"].append(auc(fpr_, tpr_))

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i][rep] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(n_classes):
                # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
                mean_tpr += tpr[i][rep]

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"].append(all_fpr)
            tpr["macro"].append(mean_tpr)
            roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][rep]))

        # saving results.
        results[key] = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]

    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)


def run_loo_population_classification(X_dict, y_dict, n_classes=4, cv_reps=3, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                  clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i] = []
        tpr[i] = []
        roc_auc[i] = []

    fpr["micro"] = []
    tpr["micro"] = []
    roc_auc["micro"] = []

    fpr["macro"] = []
    tpr["macro"] = []
    roc_auc["macro"] = []

    mean_fpr = np.linspace(0, 1, 300)

    classes = np.arange(n_classes).astype(float)
    rep = 0
    for key in y_dict:
        # dic temp copy
        temp_X_dict = X_dict.copy()
        temp_y_dict = y_dict.copy()
        # use data from key to test
        # get one id out
        # generate training and testing
        X_test = temp_X_dict.pop(key)
        y_test = label_binarize(temp_y_dict.pop(key).ravel(), classes=classes)

        X_train = \
            np.concatenate([tmp_inst_array for tmp_inst_array in temp_X_dict.values()], axis=0)
        y_train = np.concatenate([tmp_label_array for tmp_label_array in temp_y_dict.values()],
                                             axis=0)
        y_train = label_binarize(y_train.ravel(), classes=classes)

        # train, test and save scores
        # each id out is an "cv" run

        if o_normalize_data:
            norm_constants = classifier_cv.get_normalization_constants(X_train)
            X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
            X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()

        if o_perform_pca:
            pca = PCA(n_components=n_pcs)
            pca.fit(X_train)
            X_train = pca.transform(X_train).copy()
            X_test = pca.transform(X_test).copy()

            # create and train classifiers
        # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
        clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

        # test classifiers
        y_score = clf.decision_function(X_test)
        # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])

        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            # roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc_ = auc(fpr_, tpr_)
            fpr[i].append(fpr_)
            tpr[i].append(interp(mean_fpr, fpr_, tpr_))
            roc_auc[i].append(roc_auc_)

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc_ = auc(fpr_, tpr_)
        fpr["micro"].append(fpr_)
        tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
        roc_auc["micro"].append(auc(fpr_, tpr_))

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i][rep] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(n_classes):
            # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
            mean_tpr += tpr[i][rep]

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"].append(all_fpr)
        tpr["macro"].append(mean_tpr)
        roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][rep]))

        rep += 1

    # saving results.
    results = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)


def run_population_classification(X, y, cv_reps=3, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                  clf_obj=svm.SVC, clf_par=None, output_file_name='out.data'):
    n_classes = len(np.unique(y))

    y = label_binarize(y, classes=np.unique(y))
    # cv_reps = 3


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i] = []
        tpr[i] = []
        roc_auc[i] = []

    fpr["micro"] = []
    tpr["micro"] = []
    roc_auc["micro"] = []

    fpr["macro"] = []
    tpr["macro"] = []
    roc_auc["macro"] = []

    mean_fpr = np.linspace(0, 1, 300)

    for rep in range(cv_reps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rep, shuffle=True)

        if o_normalize_data:
            norm_constants = classifier_cv.get_normalization_constants(X_train)
            X_train = classifier_cv.normalize_data(X_train, norm_constants).copy()
            X_test = classifier_cv.normalize_data(X_test, norm_constants).copy()

        if o_perform_pca:
            pca = PCA(n_components=n_pcs)
            pca.fit(X_train)
            X_train = pca.transform(X_train).copy()
            X_test = pca.transform(X_test).copy()

            # create and train classifiers
        # clf = OneVsRestClassifier(svm.SVC(**svm_par)).fit(X_train, y_train)
        clf = OneVsRestClassifier(clf_obj(**clf_par)).fit(X_train, y_train)

        # test classifiers
        y_score = clf.decision_function(X_test)
        # fpr, tpr, thresholds = metrics.roc_curve(binary_labels[test], probas_[:, 1])

        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            # roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_, tpr_, _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc_ = auc(fpr_, tpr_)
            fpr[i].append(fpr_)
            tpr[i].append(interp(mean_fpr, fpr_, tpr_))
            roc_auc[i].append(roc_auc_)

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        fpr_, tpr_, _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc_ = auc(fpr_, tpr_)
        fpr["micro"].append(fpr_)
        tpr["micro"].append(interp(mean_fpr, fpr_, tpr_))
        roc_auc["micro"].append(auc(fpr_, tpr_))

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i][rep] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(n_classes):
            # mean_tpr += interp(all_fpr, fpr[i][rep], tpr[i][rep])
            mean_tpr += tpr[i][rep]

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"].append(all_fpr)
        tpr["macro"].append(mean_tpr)
        roc_auc["macro"].append(auc(mean_fpr, tpr["macro"][rep]))

    # saving results.
    results = [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)



def run_population_classification_sc(X, y, cv_folds=5, cv_reps=5, o_normalize_data=False, o_perform_pca=False, n_pcs=10,
                                    clf_obj=svm.SVC, clf_par=None, output_file_name='out.data',
                                     dict_of_superposition_lists=None):

    # get number of cores in the cpu
    # n_jobs = min(joblib.cpu_count(), sbjIDDict.keys().__len__())
    # n_jobs = min(joblib.cpu_count(), len(X))
    n_jobs = min(joblib.cpu_count(), cv_reps)

    classifier = clf_obj(**clf_par)

    all_subjects_instances_array = \
        np.concatenate([tmp_inst_list for tmp_inst_list in X.values()], axis=0)
    all_subjects_labels = np.concatenate([tmp_label_list for tmp_label_list in y.values()],
                                         axis=0)

    X = all_subjects_instances_array
    y = all_subjects_labels.ravel()
    # making it single class
    y[y >= 1] = 1

    color_count = 0
    plt.figure()

    mean_fpr, mean_tpr, mean_auc, std_auc, tprs = classifier_cv.classify_instance_parallel_cv(classifier, X, y, cv_folds,
                                                                                              cv_reps, n_jobs, o_perform_pca,
                                                                                              n_pcs, o_normalize_data,
                                                                                              dict_of_superposition_lists=dict_of_superposition_lists)
    results = [mean_fpr, mean_tpr, mean_auc, std_auc, tprs]
    with open(output_file_name, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(results, filehandle)

    return results


def plot_rocs(mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames, labels_fontsize=16, show=False,
              outputdir=None):

    # Plot all ROC curves
    class_dict = {0: "No", 1: "ED", 2: "SIB", 3: "Agg"}
    n_classes = 4
    lw = 2
    plt.figure()

    # not plotting  micro!
    # plt.semilogx(mean_fpr, np.mean(tpr["micro"], axis=0),
    #          label='micro-average ROC curve (AUC = {0:0.2f})'
    #                ''.format(np.mean(roc_auc["micro"])),
    #          color='deeppink', linestyle=':', linewidth=4)

    # macro is the average roc

    plt.semilogx(mean_fpr, np.mean(tpr["macro"], axis=0),
            #label='macro-average ROC curve (AUC = {0:0.2f})'
            label='Average ROC curve (AUC = {0:0.2f})'
            ''.format(np.mean(roc_auc["macro"])),
            color='navy', linestyle=':', linewidth=4)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # colors = cycle(['darkblue', 'darkorange', 'black', 'darkred'])
    colors = cycle(['blue', 'darkorange', 'black', 'red'])
    for i, color in zip(range(n_classes), colors):
        # for j in range(len(tpr[i])):
            # plt.plot(mean_fpr, tpr[i][j], color=color, lw=lw/2)
            #          # label='ROC curve of class {0} (AUC = {1:0.2f})'
            #          #       ''.format(class_dict[i], np.mean(roc_auc[i])))

        plt.semilogx(mean_fpr, np.mean(tpr[i], axis=0), color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.2f})'
                           ''.format(class_dict[i], np.mean(roc_auc[i])))


    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # figure config
    plt.xlabel('False Positive Rate (1-Specificity)', fontweight='bold', fontsize=labels_fontsize)
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=labels_fontsize)
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.grid(color=[.9, .9, .9], linestyle='--')
    plt.savefig(outputdir + '/svm_multiclass_Tp' + str(num_observation_frames) + '_Tf' + str(num_prediction_frames) + '.pdf')
    plt.savefig(outputdir + '/svm_multiclass_Tp' + str(num_observation_frames) + '_Tf' + str(num_prediction_frames) + '.png')
    if show:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputdir", type=str, default="Results", help="Output directory (default: ./Results/)")
    parser.add_argument("-op", type=bool, default=False, help="Only plot figures without running sym.")
    parser.add_argument("-g", "--gamma", type=float, default=0.1, help="Inverse of kernel bandwidth")
    parser.add_argument("-C", type=float, default=100, help="SVM regularization parameter")
    parser.add_argument("-k", "--kernel", type=str, default="rbf", help="Kernel name (e.g., rbf, poly, etc)")
    parser.add_argument("-tp", type=int, default=12, help="number of bins in the past (default is 12 = 3 min)")
    parser.add_argument("-tf", type=int, default=4, help="Number of bins in the future (default is 4 = 1 min)")
    parser.add_argument("-pca", type=bool, default=True, help="Perform PCA (default is True)")
    parser.add_argument("-n", "--normalize", type=bool, default=True,
                        help="Perform zero-mean one-std normalization (default is True)")
    parser.add_argument("-m", "--model", type=int, default=0,
                        help="model type (default is 0: population model. Use 1 for individual models, 2 for "
                             "leave-one-subject-out population models, "
                             "and 3 for individual models with leave-one-session out)")
    parser.add_argument("-s", "--min_sessions", type=int, default=2,
                        help="minimum number of sessions per subject (default is 2, only needed with -m 3)")
    parser.add_argument("-n_pcs", type=int, default=10, help="Number of Principal Components (default is 10)")
    parser.add_argument("-cv_reps", type=int, default=10, help="Number of data splits and runs(default is 10)")

    args = parser.parse_args()

    o_is_new_dataset = True
    # data_path = '/scratch/talesim/new_CBS_data'
    # data_path = '/scratch/talesim/new_CBS_data_small'
    # data_path = '/scratch/talesim/new_CBS_data_full'
    # data_path = '/scratch/talesim/tes/'
    # data_path = '/home/tales/DataBases/new_CBS_data_small'
    # data_path = '/home/tales/DataBases/test_t4'
    data_path = '/home/tales/DataBases/new_CBS_data'
    # data_path = '/home/tales/DataBases/new_CBS_data_full'
    # dataPath = '/home/tales/DataBases/new_data_t1'
    # data_path = '/home/tales/DataBases/newT3'
    # data_path = '/home/tales/DataBases/Tranche 5'
    # data_path = '/home/tales/DataBases/Test_T3'
    # data_path = '/home/tales/DataBases/t3_small'
    path_style = '/'

    """
    About model_options:
    POP: Population models
    IND: Individual models
    LOO: Population models with leave one individual out cv
    LOSO: Individual models with leave one session out cv
    POP_SC: Population model single class (Aggression only)
    IND_SC: Individual models single class (Aggression only)
    """
    model_options = {'POP': 0, 'IND': 1, 'LOO': 2, 'LOSO': 3, 'POP_SC': 4, 'IND_SC': 5}
    # o_return_list_of_sessions should be True only if LOSO is selected.
    o_return_list_of_sessions = False

    # subjectIDCoding = 4  # Number of digits in subject ID coding
    # num_observation_frames = 12
    # num_prediction_frames = 4
    num_observation_frames = args.tp
    num_prediction_frames = args.tf

    # feat_code = 7                       # use all features
    feat_code = 6  # use all features but 'AGGObserved' and 'TimePastAggression'
    # feat_code = 1  # use only ACC data

    o_normalize_data = args.normalize  # normalize data?
    o_perform_pca = args.pca  # perform PCA?
    n_pcs = args.n_pcs  # number of principal components

    svm_par = {'gamma': args.gamma, 'C': args.C, 'kernel': args.kernel, 'probability': True}

    if args.model == model_options['IND']:
        args.outputdir += '/Ind'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
    elif args.model == model_options['POP']:
        args.outputdir += '/POP'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
    elif args.model == model_options['LOO']:
        # leave-one-subject-out population model
        args.outputdir += '/LOO'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
    elif args.model == model_options['LOSO']:
        args.outputdir += '/LOSO'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
        o_return_list_of_sessions = True
    elif args.model == model_options['POP_SC']:
        args.outputdir += '/POP_SC'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
    elif args.model == model_options['IND_SC']:
        args.outputdir += '/IND_SC'
        if not os.path.isdir(args.outputdir):
            os.makedirs(args.outputdir)
    else:
        print("Invalid model option!")
        pass

    if not args.op:
        dict_of_instances_arrays, dict_of_labels_arrays, id_blacklist, uid_dict, dict_of_superposition_lists\
            = data_preprocess(data_path, path_style='/', num_observation_frames=num_observation_frames,
                              num_prediction_frames=num_prediction_frames, feat_code=feat_code,
                              o_return_list_of_sessions=o_return_list_of_sessions)

        if args.model == model_options['IND']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_individual_model.data'.format(args.tp, args.tf)
            # perform classification with individual models
            run_individual_classification(dict_of_instances_arrays, dict_of_labels_arrays, cv_reps=args.cv_reps,
                                          o_normalize_data=o_normalize_data,
                                          o_perform_pca=o_perform_pca,
                                          n_pcs=n_pcs, clf_obj=svm.SVC, clf_par=svm_par,
                                          output_file_name=data_file_name)

        elif args.model == model_options['POP']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm.data'.format(args.tp, args.tf)
            # perform classification with population models
            # get all data
            all_subjects_instances_array = \
                np.concatenate([tmp_inst_array for tmp_inst_array in dict_of_instances_arrays.values()], axis=0)
            all_subjects_labels = np.concatenate([tmp_label_array for tmp_label_array in dict_of_labels_arrays.values()],
                                                 axis=0)

            X = all_subjects_instances_array
            y = all_subjects_labels.ravel()

            run_population_classification(X, y, cv_reps=args.cv_reps, o_normalize_data=o_normalize_data, o_perform_pca=o_perform_pca,
                                          n_pcs=n_pcs, clf_obj=svm.SVC, clf_par=svm_par, output_file_name=data_file_name)

        elif args.model == model_options['LOO']:
            # loo populations models
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_loo_pop_model.data'.format(args.tp, args.tf)
            run_loo_population_classification(dict_of_instances_arrays, dict_of_labels_arrays, cv_reps=args.cv_reps,
                                          o_normalize_data=o_normalize_data,
                                          o_perform_pca=o_perform_pca,
                                          n_pcs=n_pcs, clf_obj=svm.SVC, clf_par=svm_par,
                                          output_file_name=data_file_name)

        elif args.model == model_options['LOSO']:
            # loo populations models
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_loso_ind_model.data'.format(args.tp, args.tf)
            run_loso_individual_classification(dict_of_instances_arrays, dict_of_labels_arrays, cv_reps=args.cv_reps,
                                          o_normalize_data=o_normalize_data,
                                          o_perform_pca=o_perform_pca,
                                          n_pcs=n_pcs, clf_obj=svm.SVC, clf_par=svm_par,
                                          output_file_name=data_file_name, min_sessions=args.min_sessions)

        elif args.model == model_options['POP_SC']:
            # single class populations models
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_popsc_ind_model.data'.format(args.tp, args.tf)
            # exit()
            results = run_population_classification_sc(dict_of_instances_arrays, dict_of_labels_arrays, cv_folds=10,
                                                       cv_reps=1, o_normalize_data=True, o_perform_pca=True, n_pcs=10,
                                                       clf_obj=svm.SVC, clf_par=svm_par, output_file_name=data_file_name,
                                                       dict_of_superposition_lists=dict_of_superposition_lists)

        elif args.model == model_options['IND_SC']:
            # loo populations models
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_loso_ind_model.data'.format(args.tp, args.tf)

        else:
            pass

    else:
        if args.model == model_options['IND']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_individual_model.data'.format(args.tp, args.tf)
        elif args.model == model_options['POP']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm.data'.format(args.tp, args.tf)
        elif args.model == model_options['LOO']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_loo_pop_model.data'.format(args.tp, args.tf)
        elif args.model == model_options['LOSO']:
            data_file_name = args.outputdir + '/sim_tp_{}_tf_{}_svm_loso_ind_model.data'.format(args.tp, args.tf)
        else:
            pass

    with open(data_file_name, 'rb') as filehandle:
        # read the data as binary data stream
        results = pickle.load(filehandle)

    if args.model == model_options['IND'] or args.model == model_options['LOSO']:
        tprs = []
        roc_aucs = []
        for key in results.keys():
            [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames] = results[key]
            tprs.append(tpr)
            roc_aucs.append(roc_auc)

        mean_tpr_dict = {}
        mean_auc_dict = {}
        for key in tpr.keys():
            mean_tpr_dict[key] = []
            mean_auc_dict[key] = []
            for i in range(len(tprs)):
                for k in range(len(tprs[i][key])):
                    # Removing nan scores (nan happens when a class was not observed with the individual data)
                    if not (sum(np.isnan(tprs[i][key][k])) > 0):
                        mean_tpr_dict[key].append(tprs[i][key][k])
                        mean_auc_dict[key].append(roc_aucs[i][key][k])

        plot_rocs(mean_fpr, mean_tpr_dict, mean_auc_dict, num_observation_frames, num_prediction_frames, show=False,
                  outputdir=args.outputdir)

    elif args.model == model_options['LOO']:
        [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames] = results
        # remove not a number from tpr and roc_aucs
        for key in roc_auc.keys():
            rlist = roc_auc[key]
            roc_auc[key] = [x for x in rlist if (np.isnan(x) == 0)]
            tlist = tpr[key]
            tpr[key] = [x for x in tlist if (sum(np.isnan(x) == 0))]

        plot_rocs(mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames, show=False,
                  outputdir=args.outputdir)

    elif args.model == model_options['POP_SC']:
        print("AUC = " + str(results[2]))
        plt.plot(results[0], results[1])
        plt.show()
    else:
        [mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames] = results
        plot_rocs(mean_fpr, tpr, roc_auc, num_observation_frames, num_prediction_frames, show=False,
                  outputdir=args.outputdir)



