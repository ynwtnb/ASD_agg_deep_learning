import os
import numpy
import torch
import random
import joblib
import sklearn
import sklearn.linear_model
from cuml.cluster import KMeans
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from cuml.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import timeit

import utils
import losses
import networks
import slide

class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0, seed=42,
                 final_shapelet_num=3):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.seed = seed
        self.final_shapelet_num = final_shapelet_num
        self.loss = losses.triplet.PNTripletLoss(
            compared_length, seed=seed
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_shapelet(self, prefix_file, shapelet, shapelet_dim):
        '''
        write the shapelet and its dimension to file
        '''
        # save shapelet
        fo_shapelet =open(prefix_file+"shapelet.txt", "w")
        for j in range(len(shapelet)):
            shapelet_tmp = numpy.asarray(shapelet[j])
            s = shapelet_tmp.reshape(1,-1)
            numpy.savetxt(fo_shapelet, s)

        fo_shapelet.close()

        # save shapelet variable
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "w")
        numpy.savetxt(fo_shapelet_dim, shapelet_dim)
        fo_shapelet_dim.close()

    def load_shapelet(self, prefix_file):
        '''
        load the shapelet and its dimension from disk
        '''
        # save shapelet
        fo_shapelet = prefix_file+"shapelet.txt"
        with open(fo_shapelet, "r") as fo_shapelet:
            shapelet = []
            for line in fo_shapelet:
                shapelet.append(line)
        fo_shapelet.close()

        # save shapelet dimension
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "r")
        shapelet_dim = numpy.loadtxt(fo_shapelet_dim)
        fo_shapelet_dim.close()

        return shapelet, shapelet_dim

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_svm_linear(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an svm linear
        classifier.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        self.classifier = LinearSVC(class_weight='balanced')
        self.classifier.fit(features, y)

        return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False,
                    prefix_file=None):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        @param prefix_file If given, saves an epoch checkpoint after every
               epoch and resumes from the latest checkpoint on restart.
        """
        train = torch.from_numpy(X).float()
        if self.cuda:
            train = train.cuda(self.gpu).float()

        train_torch_dataset = utils.Dataset(X)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True,
            generator=generator
        )

        # Resume from epoch checkpoint if available
        start_epoch = 0
        if prefix_file is not None:
            ckpt_path = prefix_file + '_epoch_ckpt.pth'
            if os.path.exists(ckpt_path):
                map_location = (
                    (lambda storage, loc: storage.cuda(self.gpu))
                    if self.cuda else
                    (lambda storage, loc: storage)
                )
                ckpt = torch.load(ckpt_path, map_location=map_location)
                self.encoder.load_state_dict(ckpt['encoder'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1
                print(f"[cache] Resuming encoder from epoch {start_epoch}/{self.epochs}")

        # Encoder training
        total_batches = len(train_generator)
        for i in range(start_epoch, self.epochs):
            epoch_start = timeit.default_timer()
            print(f"=== Epoch {i+1}/{self.epochs} ===")
            for batch_idx, batch in enumerate(train_generator):
                print(f"batch {batch_idx+1}/{total_batches}")
                batch_start = timeit.default_timer()
                if self.cuda:
                    batch = batch.cuda(self.gpu).float()
                else:
                    batch = batch.float()
                self.optimizer.zero_grad()
                loss = self.loss(
                   batch, self.encoder, self.params, save_memory=save_memory
                )
                loss.backward()
                self.optimizer.step()
                batch_end = timeit.default_timer()
                print(f"  batch time: {(batch_end - batch_start)/60:.3f} min")

            epoch_end = timeit.default_timer()
            print(f"epoch {i+1}/{self.epochs} time: {(epoch_end - epoch_start)/60:.3f} min")

            if prefix_file is not None:
                torch.save(
                    {
                        'epoch': i,
                        'encoder': self.encoder.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    },
                    prefix_file + '_epoch_ckpt.pth'
                )

        return self.encoder

    def fit(self, X, y, test, test_labels, prefix_file, cluster_num, save_memory=False, verbose=False, use_cache=False, max_discovery_samples=500, test_meta=None, val_X=None, val_y=None, val_meta=None):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param test testing set.
        @param test_labels testing labels.
        @param prefix_file prefix path.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        @param use_cache If True, load intermediate results from disk when
               available and save them after each stage completes.
        """
        final_shapelet_num = self.final_shapelet_num

        # ── Stage 1: Encoder ─────────────────────────────────────────────────
        encoder_path = prefix_file + '_' + self.architecture + '_encoder.pth'
        if use_cache and os.path.exists(encoder_path):
            print(f"[cache] Loading encoder from {encoder_path}")
            self.load_encoder(prefix_file)
        else:
            t0 = timeit.default_timer()
            self.encoder = self.fit_encoder(
                X, y=y, save_memory=save_memory, verbose=verbose,
                prefix_file=prefix_file,
            )
            print(f"[timing] encoder: {(timeit.default_timer()-t0)/60:.3f} min")
            self.save_encoder(prefix_file)
            # Remove epoch checkpoint now that the final encoder is saved
            ckpt_path = prefix_file + '_epoch_ckpt.pth'
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        # ── Stage 2: Shapelet discovery ───────────────────────────────────────
        # Save shapelet + shapelet_dim + utility_sort_index together so they
        # stay consistent on resume.
        shapelets_cache_path = prefix_file + '_shapelets.npz'
        if use_cache and os.path.exists(shapelets_cache_path):
            print(f"[cache] Loading shapelets from {shapelets_cache_path}")
            data = numpy.load(shapelets_cache_path, allow_pickle=True)
            shapelet = list(data['shapelet'])
            shapelet_dim = list(data['shapelet_dim'])
            utility_sort_index = data['utility_sort_index']
        else:
            t0 = timeit.default_timer()
            shapelet, shapelet_dim, utility_sort_index = self.shapelet_discovery(
                X, y, cluster_num, batch_size=50,
                max_discovery_samples=max_discovery_samples,
                prefix_file=prefix_file,
            )
            print(f"[timing] discovery: {(timeit.default_timer()-t0)/60:.3f} min")
            self.save_shapelet(prefix_file, shapelet, shapelet_dim)  # keep existing txt format
            numpy.savez(
                shapelets_cache_path,
                shapelet=numpy.array(shapelet, dtype=object),
                shapelet_dim=numpy.array(shapelet_dim),
                utility_sort_index=utility_sort_index,
            )
            # Remove per-scale discovery checkpoints now that shapelets are saved
            for _m in range(3):
                for _suffix in [f'_discovery_m{_m}.npz', '_discovery_idx.npy']:
                    _p = prefix_file + _suffix
                    if os.path.exists(_p):
                        os.remove(_p)

        # ── Stage 3: Shapelet transformation ─────────────────────────────────
        features_cache_path = prefix_file + '_train_features.npy'
        if use_cache and os.path.exists(features_cache_path):
            print(f"[cache] Loading train features from {features_cache_path}")
            features = numpy.load(features_cache_path)
        else:
            t0 = timeit.default_timer()
            features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
            print(f"[timing] transformation: {(timeit.default_timer()-t0)/60:.3f} min")
            numpy.save(features_cache_path, features)

        # ── Stage 4: SVM (fast — no cache needed) ────────────────────────────
        print("[timing] SVM training started")
        t0 = timeit.default_timer()
        self.classifier = self.fit_svm_linear(features, y)
        print(f"[timing] SVM: {(timeit.default_timer()-t0)/60:.3f} min")

        # ── Shapelet info ─────────────────────────────────────────────────────
        shapelet_info = []
        for j in range(final_shapelet_num):
            idx = utility_sort_index[j]
            shapelet_info.append({
                'utility_rank': j,
                'channel': int(shapelet_dim[idx]),
                'length': int(len(numpy.asarray(shapelet[idx]))),
            })

        shared_extra = {
            'shapelet_info': shapelet_info,
            'train_feature_mean': features.mean(axis=0).tolist(),
            'train_feature_std': features.std(axis=0).tolist(),
            'train_class_distribution': numpy.bincount(y.astype(int)).tolist(),
        }

        # ── Test evaluation ───────────────────────────────────────────────────
        test_features = self._get_features(
            test, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num,
            cache_path=(prefix_file + '_test_features.npy') if prefix_file else None,
            use_cache=use_cache,
        )
        test_extra = dict(shared_extra)
        test_extra['test_class_distribution'] = numpy.bincount(test_labels.astype(int)).tolist()
        results = self._evaluate(test_features, test_labels, meta=test_meta, extra=test_extra)
        print(f"svm linear Accuracy: {results['accuracy']:.4f} | AUROC: {results['auroc']:.4f} | AUPRC: {results['auprc']:.4f}")

        if prefix_file is not None:
            results_path = prefix_file + '_results.json'
            with open(results_path, 'w') as fp:
                json.dump(results, fp)
            print(f"[saved] results → {results_path}")

        # ── Val evaluation ────────────────────────────────────────────────────
        if val_X is not None and val_y is not None and len(val_y) > 0:
            val_features = self._get_features(
                val_X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num,
                cache_path=(prefix_file + '_val_features.npy') if prefix_file else None,
                use_cache=use_cache,
            )
            val_extra = dict(shared_extra)
            val_extra['val_class_distribution'] = numpy.bincount(val_y.astype(int)).tolist()
            val_results = self._evaluate(val_features, val_y, meta=val_meta, extra=val_extra)
            print(f"val Accuracy: {val_results['accuracy']:.4f} | AUROC: {val_results['auroc']:.4f} | AUPRC: {val_results['auprc']:.4f}")

            if prefix_file is not None:
                val_results_path = prefix_file + '_val_results.json'
                with open(val_results_path, 'w') as fp:
                    json.dump(val_results, fp)
                print(f"[saved] val results → {val_results_path}")

        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                    count * batch_size: (count + 1) * batch_size
                ] = self.encoder(batch).cpu().detach().numpy()
                count += 1

        self.encoder = self.encoder.train()
        return features

    def shapelet_discovery(self, X, train_labels, cluster_num, batch_size=50,
                           max_discovery_samples=500, prefix_file=None):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet

        max_discovery_samples: stratified subsample before sliding to keep
        memory usage bounded.  With T~2880, D=10, 500 samples needs ~29 GB.
        Reduce if you see OOM; increase for better shapelet coverage.

        prefix_file: if given, saves per-scale checkpoints so an interrupted
        job can resume without re-encoding completed scales.
        '''

        # Stratified subsample to avoid OOM during sliding and encoding.
        # Save sampled indices so a resumed run uses the exact same subset.
        N_orig = X.shape[0]
        if N_orig > max_discovery_samples:
            idx_path = (prefix_file + '_discovery_idx.npy') if prefix_file else None
            if idx_path and os.path.exists(idx_path):
                idx = numpy.load(idx_path)
                print(f"  [discovery] loaded subsample idx: {N_orig} → {len(idx)} samples")
            else:
                classes, counts = numpy.unique(train_labels, return_counts=True)
                rng = numpy.random.default_rng(self.seed)
                selected = []
                for cls, cnt in zip(classes, counts):
                    n_select = max(1, round(max_discovery_samples * cnt / N_orig))
                    cls_idx = numpy.where(train_labels == cls)[0]
                    chosen = rng.choice(cls_idx, min(n_select, len(cls_idx)), replace=False)
                    selected.append(chosen)
                idx = numpy.sort(numpy.concatenate(selected))
                if idx_path:
                    numpy.save(idx_path, idx)
            X = X[idx]
            train_labels = train_labels[idx]
            print(f"  [discovery] stratified subsample: {N_orig} → {len(idx)} samples "
                  f"(classes: {dict(zip(*numpy.unique(train_labels, return_counts=True)))})")

        N, D, T = X.shape
        slide_num = 3
        alpha = 0.6
        beta = 6
        X_slide_num = []
        gama = 0.5

        # Precompute positions per scale for traceback (avoids storing X_slides).
        # X_slide[m] has shape (S*N*D, L), flattened from (S, N, D, L).
        # Given index_slide: s = index_slide//(N*D), n = (index_slide%(N*D))//D, d = index_slide%D
        # → candidate window = X[n, d, positions[s] : positions[s]+L]
        if T <= 50:
            _step = 1
        elif T <= 100:
            _step = 2
        elif T <= 300:
            _step = 3
        elif T <= 1000:
            _step = 4
        elif T <= 1500:
            _step = 5
        elif T <= 2000:
            _step = 7
        elif T <= 3000:
            _step = 10
        else:
            _step = 100
        scale_params = []   # list of (L, positions_array) per scale

        t0 = timeit.default_timer()
        for m in range(slide_num):
            L_m = int(T * alpha)
            max_offset_m = T - L_m
            positions_m = numpy.array([0] + list(range(1, max_offset_m + 1, _step)))
            scale_params.append((L_m, positions_m))

            ckpt_path = (prefix_file + f'_discovery_m{m}.npz') if prefix_file else None

            if ckpt_path and os.path.exists(ckpt_path):
                # ── Resume: skip slide+encode for this scale ──────────────────
                print(f"  [cache] Loading scale {m} from {ckpt_path}")
                data = numpy.load(ckpt_path, allow_pickle=True)
                representation        = data['representation']
                candidates_dim        = list(data['candidates_dim'])
                candidates_class_label = data['candidates_class_label']
                X_slide_num.append(int(data['X_slide_num']))
            else:
                # ── Slide ─────────────────────────────────────────────────────
                ts = timeit.default_timer()
                X_slide, candidates_dim, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
                print(f"  slide {m}: {(timeit.default_timer()-ts)/60:.3f} min | X_slide: {X_slide.shape}")
                X_slide_num.append(numpy.shape(X_slide)[0])

                # ── Encode ────────────────────────────────────────────────────
                te = timeit.default_timer()
                test = utils.Dataset(X_slide)
                test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)

                self.encoder = self.encoder.eval()

                # encode slide TS — accumulate in a list, concatenate once at the end
                reps = []
                with torch.no_grad():
                    for batch in test_generator:
                        if self.cuda:
                            batch = batch.cuda(self.gpu).float()
                        else:
                            batch = batch.float()
                        # 2D to 3D
                        batch.unsqueeze_(1)
                        reps.append(self.encoder(batch).cpu().detach().numpy())
                self.encoder = self.encoder.train()
                representation = numpy.concatenate(reps, axis=0)
                print(f"  encode {m}: {(timeit.default_timer()-te)/60:.3f} min")

                # ── Save checkpoint ───────────────────────────────────────────
                if ckpt_path:
                    numpy.savez(
                        ckpt_path,
                        representation=representation,
                        candidates_dim=numpy.array(candidates_dim),
                        candidates_class_label=candidates_class_label,
                        X_slide_num=numpy.array(X_slide_num[-1]),
                    )

            beta -= 2
            alpha = beta / 10

            # concatenate the new representation from different slides
            if m == 0:
                representation_all = representation
                representation_dim = candidates_dim
                representation_class_label = candidates_class_label
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis=0)
                representation_dim = representation_dim + candidates_dim
                representation_class_label = numpy.concatenate((representation_class_label, candidates_class_label), axis=0)

        print(f"  [discovery] slide+encode: {(timeit.default_timer()-t0)/60:.3f} min | representation_all: {representation_all.shape}")

        # cluster all the new representations
        t0 = timeit.default_timer()
        num_cluster = cluster_num
        kmeans = KMeans(n_clusters=num_cluster, random_state=self.seed)
        kmeans.fit(representation_all)
        print(f"  [discovery] KMeans: {(timeit.default_timer()-t0)/60:.3f} min")

        # init candidate as list
        candidate = []
        candidate_dim = []
        # two parts of utility function
        candidate_cluster_size = []
        candidate_first_representation = []
        utility = []

        # Precompute cumulative slide offsets for O(1) traceback
        slide_offsets = numpy.cumsum([0] + X_slide_num[:-1])

        t0 = timeit.default_timer()
        # select the nearest to the centroid
        for i in range(num_cluster):
            # Use global indices directly — avoids O(N²) numpy.where per sample
            cluster_global_indices = numpy.where(kmeans.labels_ == i)[0]
            cluster_size_i = len(cluster_global_indices)
            dim_in_cluster_i = [representation_dim[j] for j in cluster_global_indices]
            class_label_cluster_i = [representation_class_label[j] for j in cluster_global_indices]

            # Vectorized distance to centroid — finds nearest in one shot
            dists = numpy.linalg.norm(
                representation_all[cluster_global_indices] - kmeans.cluster_centers_[i], axis=1
            )
            nearest_local = int(numpy.argmin(dists))
            nearest_global_idx = int(cluster_global_indices[nearest_local])
            tmp_candidate_first_representation = representation_all[nearest_global_idx]

            # Traceback: reconstruct window from X using index arithmetic.
            # X_slide[k] shape is (S*N*D, L), flattened from (S, N, D, L).
            for k in range(slide_num):
                end = int(slide_offsets[k]) + X_slide_num[k]
                if nearest_global_idx < end:
                    index_slide = nearest_global_idx - int(slide_offsets[k])
                    L_k, positions_k = scale_params[k]
                    s_idx   = index_slide // (N * D)
                    n_local = (index_slide % (N * D)) // D
                    d_idx   = index_slide % D
                    pos     = int(positions_k[s_idx])
                    candidate_tmp = X[n_local, d_idx, pos : pos + L_k]
                    candidate_dim.append(d_idx)
                    break

            class_label_top = (Counter(class_label_cluster_i).most_common(1)[0][1] / len(class_label_cluster_i))
            dim_label_top = (Counter(dim_in_cluster_i).most_common(1)[0][1] / len(dim_in_cluster_i))
            if (class_label_top < (1/numpy.unique(train_labels).shape[0])) or (dim_label_top < (1/numpy.shape(X)[1])):
                if candidate_dim:
                    candidate_dim.pop()
                continue
            candidate_first_representation.append(tmp_candidate_first_representation)
            candidate_cluster_size.append(cluster_size_i)
            candidate.append(candidate_tmp)
        print(f"  [discovery] centroid selection: {(timeit.default_timer()-t0)/60:.3f} min")

        t0 = timeit.default_timer()
        # utility — vectorized pairwise distances
        if candidate_first_representation:
            rep_matrix = numpy.stack(candidate_first_representation, axis=0)  # (M, D)
            for i in range(len(candidate_first_representation)):
                ed_dist_sum = numpy.sum(numpy.linalg.norm(rep_matrix - rep_matrix[i], axis=1))
                utility.append(gama * candidate_cluster_size[i] + (1-gama) * ed_dist_sum)

        # sort utility namely candidate
        utility_sort_index = numpy.argsort(-numpy.array(utility))
        print(f"  [discovery] utility sort: {(timeit.default_timer()-t0)/60:.3f} min")

        return candidate, candidate_dim, utility_sort_index

    def shapelet_transformation(self, X, candidate, candidate_dim, utility_sort_index, final_shapelet_num):
        '''
        transform the original multivariate time series into the new one vector data space
        transformed date label the same with original label
        '''
        N = numpy.shape(X)[0]
        final_shapelet_num = min(final_shapelet_num, len(utility_sort_index))
        features = numpy.empty((N, final_shapelet_num))

        for j in range(final_shapelet_num):
            shapelet = numpy.asarray(candidate[utility_sort_index[j]])
            dim = int(candidate_dim[utility_sort_index[j]])
            L = len(shapelet)
            # Process in chunks to avoid materialising (N, T-L+1, L) all at once.
            # chunk_size=200 keeps each subtraction array under ~4 GB.
            for start in range(0, N, 200):
                end = min(start + 200, N)
                windows = numpy.lib.stride_tricks.sliding_window_view(
                    X[start:end, dim, :], L, axis=1
                )
                dists = numpy.linalg.norm(windows - shapelet, axis=2)  # (chunk, T-L+1)
                features[start:end, j] = dists.min(axis=1)

        return features

    def _evaluate(self, features, y_true, meta=None, extra=None):
        """
        Compute predictions and metrics on precomputed features.

        Returns a dict with accuracy, auroc, auprc, predictions, and optionally
        meta fields and any extra key/value pairs passed via `extra`.
        """
        y_pred = self.classifier.predict(features)
        decision_scores = self.classifier.decision_function(features)
        accuracy = self.classifier.score(features, y_true)
        try:
            auroc = roc_auc_score(y_true, decision_scores)
            auprc = average_precision_score(y_true, decision_scores)
        except ValueError as e:
            print(f"[warning] metric computation failed: {e}")
            auroc = float('nan')
            auprc = float('nan')

        results = {
            'accuracy': accuracy,
            'auroc': auroc,
            'auprc': auprc,
            'y_true': y_true.tolist(),
            'y_pred': numpy.asarray(y_pred).tolist(),
            'decision_scores': numpy.asarray(decision_scores).tolist(),
        }
        if meta is not None:
            results['participant_ids'] = numpy.asarray(meta['participant_ids']).tolist()
            results['session_ids'] = numpy.asarray(meta['session_ids']).tolist()
            results['superposition_lists'] = numpy.asarray(meta['superposition_lists']).tolist()
        if extra is not None:
            results.update(extra)
        return results

    def _get_features(self, X, shapelet, shapelet_dim, utility_sort_index,
                      final_shapelet_num, cache_path=None, use_cache=False):
        """
        Compute shapelet transformation features, with optional disk caching.
        """
        if use_cache and cache_path and os.path.exists(cache_path):
            print(f"[cache] Loading features from {cache_path}")
            return numpy.load(cache_path)
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        if cache_path:
            numpy.save(cache_path, features)
        return features

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        return self.classifier.score(features, y)

class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Length of the compared positive and negative samples
           in the loss. Ignored if None, or if the time series in the training
           set have unequal lengths.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param epochs Number of epochs to run during the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, batch_size=1, epochs=100, lr=0.001,
                 channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0, seed=42, final_shapelet_num=3):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu, seed,
            final_shapelet_num=final_shapelet_num
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        # encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # First applies the causal CNN
                output_causal_cnn = causal_cnn(batch)
                after_pool = torch.empty(
                    output_causal_cnn.size(), dtype=torch.double
                )
                if self.cuda:
                    after_pool = after_pool.cuda(self.gpu)
                after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                # Then for each time step, computes the output of the max
                # pooling layer
                for i in range(1, length):
                    after_pool[:, :, i] = torch.max(
                        torch.cat([
                            after_pool[:, :, i - 1: i],
                            output_causal_cnn[:, :, i: i+1]
                         ], dim=2),
                        dim=2
                    )[0]
                features[
                    count * batch_size: (count + 1) * batch_size, :, :
                ] = torch.transpose(linear(
                    torch.transpose(after_pool, 1, 2)
                ), 1, 2)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu,
            'seed': self.seed,
            'final_shapelet_num': self.final_shapelet_num
        }

    def set_params(self, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, seed=42, final_shapelet_num=3):
        self.__init__(
            compared_length, batch_size, epochs, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu, seed,
            final_shapelet_num=final_shapelet_num
        )
        return self
