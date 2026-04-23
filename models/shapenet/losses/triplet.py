import torch
import numpy
import random
from collections import Counter
from fast_pytorch_kmeans import KMeans as TorchKMeans
import timeit

import slide


class PNTripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length, seed=42):
        super(PNTripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.seed = seed

    def forward(self, batch, encoder, params, save_memory=False):

        slide_num = 3
        alpha = 0.6
        device = batch.device
        loss = torch.FloatTensor([0]).to(device)

        for m in range(slide_num):
            slide_start = timeit.default_timer()

            # Slide raw data to a shorter length
            batch_slide = slide.slide_MTS_tensor_step(batch, alpha)
            alpha = alpha - 0.2
            device = batch_slide.device

            # KMeans clustering on GPU
            num_cluster = 2
            torch.manual_seed(self.seed)
            kmeans = TorchKMeans(n_clusters=num_cluster, mode='euclidean', verbose=0)
            labels_tensor = kmeans.fit_predict(batch_slide)   # stays on GPU
            cluster_label = labels_tensor.cpu().numpy()
            points = batch_slide.cpu().numpy()                # needed for sample_list (Phase 2)
            num_cluster_set = Counter(cluster_label)
            seq_len = batch_slide.size(1)

            # ------------------------------------------------------------------
            # Phase 1: determine all sample indices needed per cluster (CPU only)
            # ------------------------------------------------------------------
            cluster_meta = []
            for ci in range(num_cluster):
                if num_cluster_set[ci] < 2:
                    cluster_meta.append(None)
                    continue

                distance_i = torch.norm(
                    batch_slide[labels_tensor == ci] - kmeans.centroids[ci], dim=1
                ).cpu().numpy()

                num_positive = 50 if num_cluster_set[ci] >= 250 else int(num_cluster_set[ci] / 5 + 1)
                anchor_positive = numpy.argpartition(distance_i, num_positive)[:(num_positive + 1)]

                # Find the farthest pair among positives (intra-positive)
                intra_pos_pair = None
                if num_positive > 1:
                    pos_dist_max = float("-inf")
                    for pi in range(1, num_positive):
                        for pj in range(pi + 1, num_positive + 1):
                            d = numpy.linalg.norm(points[anchor_positive[pi]] - points[anchor_positive[pj]])
                            if d > pos_dist_max:
                                pos_dist_max = d
                                intra_pos_pair = (pi, pj)

                # Negative clusters
                neg_info = []
                for k in range(num_cluster):
                    if k == ci:
                        continue
                    neg_points = points[cluster_label == k]
                    num_neg_k = 50 if num_cluster_set[k] >= 250 else int(num_cluster_set[k] / 5 + 1)
                    neg_indices = random.sample(range(neg_points.shape[0]), num_neg_k)

                    intra_neg_pair = None
                    if num_neg_k > 1:
                        neg_dist_max = float("-inf")
                        for ni in range(0, num_neg_k - 1):
                            for nj in range(ni + 1, num_neg_k):
                                d = numpy.linalg.norm(neg_points[neg_indices[ni]] - neg_points[neg_indices[nj]])
                                if d > neg_dist_max:
                                    neg_dist_max = d
                                    intra_neg_pair = (ni, nj)

                    neg_info.append({
                        'k': k,
                        'num_neg_k': num_neg_k,
                        'neg_indices': neg_indices,
                        'neg_points': neg_points,
                        'intra_neg_pair': intra_neg_pair,
                    })

                cluster_meta.append({
                    'anchor_positive': anchor_positive,
                    'num_positive': num_positive,
                    'intra_pos_pair': intra_pos_pair,
                    'neg_info': neg_info,
                })

            # ------------------------------------------------------------------
            # Phase 2: collect all samples that need encoding into one list
            # ------------------------------------------------------------------
            sample_list = []   # list of 1D numpy arrays (length seq_len)
            layout = []        # records (cluster_idx, role, count) for phase 4

            for ci in range(num_cluster):
                meta = cluster_meta[ci]
                if meta is None:
                    continue

                ap = meta['anchor_positive']
                num_pos = meta['num_positive']

                # anchor + positives: indices 0..num_pos in ap
                for idx in range(num_pos + 1):
                    sample_list.append(points[ap[idx]])
                layout.append(('anchor_pos', ci, num_pos + 1))

                # intra-positive pair
                if meta['intra_pos_pair'] is not None:
                    pi, pj = meta['intra_pos_pair']
                    sample_list.append(points[ap[pi]])
                    sample_list.append(points[ap[pj]])
                    layout.append(('intra_pos', ci, 2))
                else:
                    layout.append(('intra_pos', ci, 0))

                # negatives and intra-negatives
                for neg in meta['neg_info']:
                    for j in neg['neg_indices']:
                        sample_list.append(neg['neg_points'][j])
                    layout.append(('neg', ci, neg['num_neg_k']))

                    if neg['intra_neg_pair'] is not None:
                        ni, nj = neg['intra_neg_pair']
                        sample_list.append(neg['neg_points'][neg['neg_indices'][ni]])
                        sample_list.append(neg['neg_points'][neg['neg_indices'][nj]])
                        layout.append(('intra_neg', ci, 2))
                    else:
                        layout.append(('intra_neg', ci, 0))

            if not sample_list:
                continue

            # ------------------------------------------------------------------
            # Phase 3: single batched encoder forward pass on GPU
            # ------------------------------------------------------------------
            all_samples = numpy.stack(sample_list, axis=0)          # (N, seq_len)
            all_tensor = torch.from_numpy(all_samples).float().to(device)   # (N, seq_len)
            all_tensor = all_tensor.unsqueeze(1)                     # (N, 1, seq_len)
            all_encoded = encoder(all_tensor)                        # (N, out_channels)

            # ------------------------------------------------------------------
            # Phase 4: compute loss using batched encoded representations
            # ------------------------------------------------------------------
            loss_cluster = torch.FloatTensor([0]).to(device)
            ptr = 0       # pointer into all_encoded
            lay_ptr = 0   # pointer into layout

            # Gather per-cluster encoded tensors
            cluster_encoded = {}  # ci -> dict of tensors
            ci_order = [ci for ci in range(num_cluster) if cluster_meta[ci] is not None]

            for ci in ci_order:
                meta = cluster_meta[ci]
                num_pos = meta['num_positive']

                # anchor + positives
                role, _, count = layout[lay_ptr]; lay_ptr += 1
                enc_anchor = all_encoded[ptr]                        # (out_channels,)
                enc_positives = all_encoded[ptr + 1: ptr + count]    # (num_pos, out_channels)
                ptr += count

                # intra-positive
                role, _, count = layout[lay_ptr]; lay_ptr += 1
                if count > 0:
                    enc_ip0 = all_encoded[ptr]
                    enc_ip1 = all_encoded[ptr + 1]
                    ptr += 2
                    dist_intra_positive = torch.norm(enc_ip0 - enc_ip1)
                else:
                    dist_intra_positive = torch.FloatTensor([0]).to(device)

                # dist_positive: mean distance anchor → each positive
                dist_positive = torch.norm(
                    enc_anchor.unsqueeze(0) - enc_positives, dim=1
                ).mean()

                dist_negative = torch.FloatTensor([0]).to(device)
                dist_intra_negative = torch.FloatTensor([0]).to(device)

                for neg in meta['neg_info']:
                    # negatives
                    role, _, count = layout[lay_ptr]; lay_ptr += 1
                    enc_negs = all_encoded[ptr: ptr + count]         # (num_neg_k, out_channels)
                    ptr += count
                    dist_neg_k = torch.norm(
                        enc_anchor.unsqueeze(0) - enc_negs, dim=1
                    ).mean()
                    dist_negative += dist_neg_k

                    # intra-negative
                    role, _, count = layout[lay_ptr]; lay_ptr += 1
                    if count > 0:
                        enc_in0 = all_encoded[ptr]
                        enc_in1 = all_encoded[ptr + 1]
                        ptr += 2
                        dist_intra_negative += torch.norm(enc_in0 - enc_in1)

                dist_negative = dist_negative / (num_cluster - 1)
                loss_cluster = loss_cluster + torch.log(dist_positive / dist_negative)
                loss_cluster = loss_cluster + dist_intra_positive
                loss_cluster = loss_cluster + dist_intra_negative

            loss = loss + loss_cluster
            slide_end = timeit.default_timer()
            print(f"  slide time: {(slide_end - slide_start) / 60:.3f} min")

        return loss
