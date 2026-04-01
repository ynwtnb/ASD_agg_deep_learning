import torch
import math
import numpy
import sklearn
import random
from scipy import spatial
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import timeit

import networks
import slide


class PNTripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length):
        super(PNTripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf

    def forward(self, batch, encoder, params, save_memory=False):

     slide_num = 3
     alpha = 0.6
     loss = torch.DoubleTensor([1])
     for m in range(slide_num):
         slide_start = timeit.default_timer()
         # slide raw data for different length
         batch_slide = slide.slide_MTS_tensor_step(batch, alpha)
         alpha = alpha - 0.2
         encode_num = int(alpha * 10)

         # do the cluster for the slided time series
         points = batch_slide.numpy()
         num_cluster = 2
         kmeans = KMeans(n_clusters = num_cluster)
         kmeans.fit(points)
         cluster_label = kmeans.labels_
         num_cluster_set = Counter(cluster_label)

         # loss of each cluster
         loss_cluster = torch.DoubleTensor([1])
         for i in range(num_cluster):
             cluster_start = timeit.default_timer()
             if num_cluster_set[i] < 2:
                 continue
             cluster_i = points[numpy.where(cluster_label == i)]
             distance_i = kmeans.transform(cluster_i)[:,i]
             dist_positive = torch.DoubleTensor([1])
             dist_intra_positive = torch.DoubleTensor([1])
             dist_intra_negative = torch.DoubleTensor([1])
             dist_negative = torch.DoubleTensor([1])
             if num_cluster_set[i] >= 250:
                 num_positive = 50
             else:
                 num_positive = int(num_cluster_set[i]/5+1)

             # select anchor and positive
             anchor_positive = numpy.argpartition(distance_i, num_positive)[:(num_positive+1)]

             # torch anchor
             representation_anc = torch.from_numpy(points[anchor_positive[0]])
             # transfer 1D to 3D
             representation_anc = torch.reshape(representation_anc, (1,1,numpy.shape(points)[1]))

             # encoder anchor
             representation_anc = encoder(representation_anc)

             # positive part
             for l in range(1,num_positive+1):
                 # torch positive
                 representation_pos = torch.from_numpy(points[anchor_positive[l]])
                 # transfer 1D to 3D
                 representation_pos = torch.reshape(representation_pos, (1,1,numpy.shape(points)[1]))

                 # encode positive
                 representation_pos = encoder(representation_pos)

                 anchor_minus_positive = representation_anc - representation_pos
                 dist_positive += torch.norm(anchor_minus_positive)

             dist_positive = dist_positive / num_positive

             # intra of positive
             if num_positive > 1:
                 pos_dist_pos = float("-inf")
                 for i in range(1,num_positive):
                     for j in range(i+1,num_positive+1):
                         pos_minus_pos = points[anchor_positive[i]] - points[anchor_positive[j]]
                         intra_pos = numpy.linalg.norm(pos_minus_pos)
                         if intra_pos > pos_dist_pos:
                             pos_dist_pos = intra_pos
                             first_index = i
                             second_index = j

                 representation_intra_pos_0 = torch.from_numpy(points[anchor_positive[first_index]])
                 representation_intra_pos_0 = torch.reshape(representation_intra_pos_0, (1,1,numpy.shape(points)[1]))
                 representation_intra_pos_0 = encoder(representation_intra_pos_0)

                 representation_intra_pos_1 = torch.from_numpy(points[anchor_positive[second_index]])
                 representation_intra_pos_1 = torch.reshape(representation_intra_pos_1, (1,1,numpy.shape(points)[1]))
                 representation_intra_pos_1 = encoder(representation_intra_pos_1)

                 intra_minus_positive = representation_intra_pos_0 - representation_intra_pos_1
                 dist_intra_positive = torch.norm(intra_minus_positive)

             # negative part
             for k in range(num_cluster):
                 dist_cluster_k_negative = torch.DoubleTensor([1])
                 if k == i:
                     continue
                 else:
                     # select negative
                     if num_cluster_set[k] >= 250:
                         num_negative_cluster_k = 50
                     else:
                         num_negative_cluster_k = int(num_cluster_set[k]/5+1)

                     negative_cluster_k = random.sample(range(points[kmeans.labels_ == k][:,0].size), num_negative_cluster_k)
                     for j in range(num_negative_cluster_k):
                         # torch negative
                         representation_neg = torch.from_numpy(points[kmeans.labels_== k][negative_cluster_k[j]])
                         # transfer 1D to 3D
                         representation_neg = torch.reshape(representation_neg, (1,1,numpy.shape(points)[1]))

                         # encode negative
                         representation_neg = encoder(representation_neg)

                         anchor_minus_negative = representation_anc - representation_neg
                         dist_cluster_k_negative += torch.norm(anchor_minus_negative)

                 dist_cluster_k_negative = dist_cluster_k_negative / num_negative_cluster_k
                 dist_negative += dist_cluster_k_negative

                 # intra of negative
                 if num_negative_cluster_k > 1:
                     neg_dist_neg = float("-inf")
                     for i in range(0,num_negative_cluster_k-1):
                         for j in range(i+1,num_negative_cluster_k):
                             neg_minus_neg = points[kmeans.labels_== k][negative_cluster_k[i]] - points[kmeans.labels_== k][negative_cluster_k[j]]
                             intra_neg = numpy.linalg.norm(neg_minus_neg)
                             if intra_neg > neg_dist_neg:
                                 neg_dist_neg = intra_neg
                                 first_index = i
                                 second_index = j

                     representation_intra_neg_0 = torch.from_numpy(points[kmeans.labels_== k][negative_cluster_k[first_index]])
                     representation_intra_neg_0 = torch.reshape(representation_intra_neg_0, (1,1,numpy.shape(points)[1]))
                     representation_intra_neg_0 = encoder(representation_intra_neg_0)

                     representation_intra_neg_1 = torch.from_numpy(points[kmeans.labels_== k][negative_cluster_k[second_index]])
                     representation_intra_neg_1 = torch.reshape(representation_intra_neg_1, (1,1,numpy.shape(points)[1]))
                     representation_intra_neg_1 = encoder(representation_intra_neg_1)

                     intra_minus_negative = representation_intra_neg_0 - representation_intra_neg_1
                     dist_intra_negative += torch.norm(intra_minus_negative)

             dist_negative = dist_negative / (num_cluster-1)
             loss_cluster += torch.log(dist_positive/dist_negative)
             loss_cluster += dist_intra_positive
             loss_cluster += dist_intra_negative
             cluster_end = timeit.default_timer()
             print("cluster time: ", (cluster_end- cluster_start)/60)

         loss += loss_cluster
         slide_end = timeit.default_timer()
         print("slide time: ", (slide_end- slide_start)/60)
     return loss
