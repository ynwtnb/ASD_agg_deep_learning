import math
import numpy
import torch
import random
import joblib
import sklearn
import sklearn.linear_model
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import timeit

import utils
import losses
import networks
import slide

class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0, seed=42):
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
        self.classifier = SVC(kernel='linear' ,gamma='auto')
        self.classifier.fit(features, y)

        return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
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

        #epochs = 0 # Number of performed epochs

        # Encoder training
        for i in range(self.epochs):
            epoch_start = timeit.default_timer()
            for batch in train_generator:
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
                print("batch time: ", (batch_end- batch_start)/60)

            #epochs += 1
            epoch_end = timeit.default_timer()
            print("epoch time: ", (epoch_end- epoch_start)/60)

        return self.encoder

    def fit(self, X, y, test, test_labels, prefix_file, cluster_num, save_memory=False, verbose=False):
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
        """
        final_shapelet_num = 3
        # Fitting encoder
        t0 = timeit.default_timer()
        self.encoder = self.fit_encoder(
                                        X, y=y, save_memory=save_memory, verbose=verbose
                                        )
        print(f"[timing] encoder: {(timeit.default_timer()-t0)/60:.3f} min")

        # shapelet discovery
        t0 = timeit.default_timer()
        shapelet, shapelet_dim, utility_sort_index = self.shapelet_discovery(X, y, cluster_num, batch_size=50)
        print(f"[timing] discovery: {(timeit.default_timer()-t0)/60:.3f} min")
        
        self.save_shapelet(prefix_file, shapelet, shapelet_dim)

        # shapelet transformation
        t0 = timeit.default_timer()
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        print(f"[timing] transformation: {(timeit.default_timer()-t0)/60:.3f} min")

        # SVM classifier training
        t0 = timeit.default_timer()
        self.classifier = self.fit_svm_linear(features, y)
        print(f"[timing] SVM: {(timeit.default_timer()-t0)/60:.3f} min")
        print("svm linear Accuracy: "+str(self.score(test, test_labels, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)))


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

    def shapelet_discovery(self, X, train_labels, cluster_num, batch_size = 50):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        '''

        slide_num = 3
        alpha = 0.6
        beta = 6
        count = 0
        X_slide_num = []
        gama = 0.5

        X_slides = []  # cache for traceback
        t0 = timeit.default_timer()
        for m in range(slide_num):
            # slide the raw time series and the corresponding class and variate label
            ts = timeit.default_timer()
            X_slide, candidates_dim, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
            X_slides.append(X_slide)
            print(f"  slide {m}: {(timeit.default_timer()-ts)/60:.3f} min | X_slide: {X_slide.shape}")
            X_slide_num.append(numpy.shape(X_slide)[0])
            beta =  beta -2
            alpha = beta/10

            te = timeit.default_timer()
            test = utils.Dataset(X_slide)
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)

            self.encoder = self.encoder.eval()

            # encode slide TS
            with torch.no_grad():
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu).float()
                    else:
                        batch = batch.float()
                    # 2D to 3D
                    batch.unsqueeze_(1)
                    batch = self.encoder(batch)

                    batch_np = batch.cpu().detach().numpy()
                    if count == 0:
                        representation = batch_np
                    else:
                        representation = numpy.concatenate((representation, batch_np), axis=0)
                    count += 1
            self.encoder = self.encoder.train()
            count = 0
            print(f"  encode {m}: {(timeit.default_timer()-te)/60:.3f} min")
            # concatenate the new representation from different slides
            if m == 0 :
                representation_all = representation
                representation_dim = candidates_dim
                representation_class_label = candidates_class_label
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis = 0)
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

            # Traceback: find which slide this global index belongs to
            for k in range(slide_num):
                end = int(slide_offsets[k]) + X_slide_num[k]
                if nearest_global_idx < end:
                    index_slide = nearest_global_idx - int(slide_offsets[k])
                    X_slide_disc = X_slides[k]
                    candidate_tmp = X_slide_disc[index_slide]
                    candidate_dim.append(index_slide % numpy.shape(X)[1])
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
        # init transformed data with list
        feature = []

        # transform original time series
        for i in range(numpy.shape(X)[0]):
            for j in range(final_shapelet_num):
            #for j in range(len(candidate)):
                dist = math.inf
                candidate_tmp = numpy.asarray(candidate[utility_sort_index[j]])
                for k in range(numpy.shape(X)[2]-numpy.shape(candidate_tmp)[0]+1):
                    difference = X[i, int(candidate_dim[utility_sort_index[j]]), 0+k : int(numpy.shape(candidate_tmp)[0])+k] - candidate_tmp
                    feature_tmp = numpy.linalg.norm(difference)
                    if feature_tmp < dist:
                        dist = feature_tmp
                feature.append(dist)

        # turn list to array and reshape
        feature = numpy.asarray(feature)
        feature = feature.reshape(numpy.shape(X)[0], final_shapelet_num)

        return feature

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
                 in_channels=1, cuda=False, gpu=0, seed=42):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu, seed
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
            'seed': self.seed
        }

    def set_params(self, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, seed=42):
        self.__init__(
            compared_length, batch_size, epochs, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu, seed
        )
        return self
