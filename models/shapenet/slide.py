import numpy
from cuml.cluster import KMeans

def slide_MTS_dim(X, alpha):
    '''
    slide the multivariate time series from 3D to 2D
    add the dimension label to each variate
    add step to reduce the num of new TS
    '''
    num_of_old_MTS = numpy.shape(X)[0]
    dim_of_old_MTS = numpy.shape(X)[1]
    length_of_old_MTS = numpy.shape(X)[2]
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    # determine step
    if length_of_old_MTS <= 50:
        step = 1
    elif length_of_old_MTS <= 100:
        step = 2
    elif length_of_old_MTS <= 300:
        step = 3
    elif length_of_old_MTS <= 1000:
        step = 4
    elif length_of_old_MTS <= 1500:
        step = 5
    elif length_of_old_MTS <= 2000:
        step = 7
    elif length_of_old_MTS <= 3000:
        step = 10
    else:
        step = 100

    max_offset = length_of_old_MTS - length_of_new_MTS
    positions = [0] + list(range(1, max_offset + 1, step))
    step_num = len(positions) - 1

    # Build all windows at once with a zero-copy strided view, then copy once.
    # windows shape: (N, D, T-L+1, L)
    windows = numpy.lib.stride_tricks.sliding_window_view(
        X, length_of_new_MTS, axis=2
    )
    # Select desired positions -> (N, D, S, L), transpose to (S, N, D, L),
    # copy once to make contiguous, then flatten to (S*N*D, L).
    selected = numpy.ascontiguousarray(
        windows[:, :, positions, :].transpose(2, 0, 1, 3)
    )
    X_alpha = selected.reshape(
        num_of_old_MTS * dim_of_old_MTS * (step_num + 1), length_of_new_MTS
    )

    return X_alpha

def slide_MTS_dim_step(X, class_label, alpha):
    '''
    slide the multivariate time series from 3D to 2D
    add the dimension label to each variate
    add step to reduce the num of new TS
    '''
    num_of_old_MTS = numpy.shape(X)[0]
    dim_of_old_MTS = numpy.shape(X)[1]
    length_of_old_MTS = numpy.shape(X)[2]
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    candidate_dim_init = list(range(dim_of_old_MTS))

    # determine step
    if length_of_old_MTS <= 50:
        step = 1
    elif length_of_old_MTS <= 100:
        step = 2
    elif length_of_old_MTS <= 300:
        step = 3
    elif length_of_old_MTS <= 1000:
        step = 4
    elif length_of_old_MTS <= 1500:
        step = 5
    elif length_of_old_MTS <= 2000:
        step = 7
    elif length_of_old_MTS <= 3000:
        step = 10
    else:
        step = 100

    max_offset = length_of_old_MTS - length_of_new_MTS
    positions = [0] + list(range(1, max_offset + 1, step))
    step_num = len(positions) - 1  # == ceil(max_offset / step)

    # Build all windows at once with a zero-copy strided view, then copy once.
    # windows shape: (N, D, T-L+1, L)
    windows = numpy.lib.stride_tricks.sliding_window_view(
        X, length_of_new_MTS, axis=2
    )
    # Select desired positions -> (N, D, S, L), transpose to (S, N, D, L),
    # copy once to make contiguous, then flatten to (S*N*D, L).
    selected = numpy.ascontiguousarray(
        windows[:, :, positions, :].transpose(2, 0, 1, 3)
    )
    X_alpha = selected.reshape(
        num_of_old_MTS * dim_of_old_MTS * (step_num + 1), length_of_new_MTS
    )

    candidate_class_label = class_label.repeat(dim_of_old_MTS * (step_num + 1))
    candidate_dim = candidate_dim_init * (num_of_old_MTS * (step_num + 1))

    return X_alpha, candidate_dim, candidate_class_label

def slide_MTS_tensor_step(X, alpha):
    '''
    tensor version
    slide the multivariate time series tensor from 3D to 2D
    add the dimension label to each variate
    add step to reduce the num of new TS
    '''
    num_of_old_MTS = X.size(0)
    dim_of_old_MTS = X.size(1)
    length_of_old_MTS = X.size(2)
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    # determine step
    if length_of_old_MTS <= 50:
        step = 1
    elif length_of_old_MTS <= 100:
        step = 2
    elif length_of_old_MTS <= 300:
        step = 3
    elif length_of_old_MTS <= 1000:
        step = 4
    elif length_of_old_MTS <= 1500:
        step = 5
    elif length_of_old_MTS <= 2000:
        step = 7
    elif length_of_old_MTS <= 3000:
        step = 10
    else:
        step = 1000

    max_offset = length_of_old_MTS - length_of_new_MTS
    positions = [0] + list(range(1, max_offset + 1, step))
    S = len(positions)

    # unfold produces a zero-copy strided view on the GPU: (N, D, T-L+1, L)
    windows = X.unfold(2, length_of_new_MTS, 1)
    # Select desired positions -> (N, D, S, L), permute to (S, N, D, L),
    # then flatten to (S*N*D, L).
    selected = windows[:, :, positions, :]
    X_beta = selected.permute(2, 0, 1, 3).reshape(
        S * num_of_old_MTS * dim_of_old_MTS, length_of_new_MTS
    )

    return X_beta
