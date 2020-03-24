# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:43:32 2020

@author: JE09757
"""
# Implement mDA (Chen et. al, 2012)
# Take in data and probability of corruption
# "Corrupt" data (but marginalize out the (expected) corruption) and learn a reconstruction specified by weights
import numpy as np
import numpy.matlib


# Learn a deep representation of data by reconstructing "corrupted" input but marginalizing out corruption
# data format: features are rows, data points are columns
# NEW: <num_data x num_features>
# Can optionally pass in precomputed mapping to use to transform data
# (e.g. if transforming test data with mapping learned from training data)
def mDA(data, prob_corruption=None, use_nonlinearity=True, mapping=None):
    if mapping is None:
        mapping = compute_reconstruction_mapping(data, prob_corruption)  # mapping = W
    # h = tanh(W * X)
    representation = np.dot(data, mapping)  # no nonlinearity
    if use_nonlinearity:
        representation = np.tanh(representation)  # inject nonlinearity
    return mapping, representation


# Compute the mapping that reconstructs corrupted (in expectation) features
def compute_reconstruction_mapping(data, prob_corruption):
    # typecast to correct datatype
    if not (np.issubdtype(data.dtype, np.float64) or np.issubdtype(data.dtype, np.integer)):
        print("data type ", data.dtype)
        data.dtype = "float64"
    num_features = data.shape[1]  # d = size(X, 1);

    # Represents the probability that a given feature will be corrupted
    feature_corruption_probs = np.ones((num_features, 1)) * (1 - prob_corruption)  # q = [ones(d - 1, 1) .* (1 - p); 1]
    # TODO could automatically check if last "feature" is all 1s (i.e. bias)
    # instead of requiring user to tell us
    bias = False
    try:
        if np.allclose(np.ones((num_features, 1)), data[:, -1]):
            bias = True
    except Exception as e:
        raise ValueError(e)
    if bias:  # last term is actually a bias term, not an actual feature
        feature_corruption_probs[-1] = 1  # don't corrupt the bias term ever
    scatter_matrix = np.dot(data.transpose(), data)  # S = X * X’
    Q = scatter_matrix * (np.dot(feature_corruption_probs, feature_corruption_probs.transpose()))  # Q = S .* (q * q’)
    Q[np.diag_indices_from(Q)] = feature_corruption_probs[:, 0] * np.diag(
        scatter_matrix)  # Q(1:d + 1:end) = q .* diag(S)
    P = scatter_matrix * np.matlib.repmat(feature_corruption_probs, 1,
                                          num_features)  # P = S .* repmat(q’, d, 1)

    # solve equation of the form x = BA^-1, or xA = B, or A.T x.T = B.T
    A = Q + 10 ** -5 * np.eye(num_features)  # (Q + 1e-5 * eye(d))
    B = P  # [:num_features - 1,:] # P(1 : end - 1,:)
    # TODO maybe shouldn't subtract 1 (since then wouldn't be corrupting last real feature, instead of not corrupting bias)
    mapping = np.linalg.solve(A.transpose(), B.transpose())  # .transpose()
    return mapping


# Stack mDA layers on top of each other, using previous layer as input for the next
# Can optionally pass in precomputed mapping to use to transform data
# (e.g. if transforming test data with mapping learned from training data)
def mSDA(data, prob_corruption, num_layers, use_nonlinearity=True, precomp_mappings=None):
    num_data, num_features = data.shape
    mappings = list()
    representations = list()
    representations.append(data)
    # construct remaining layers recursively based on the output of previous layer
    if precomp_mappings is None:
        for layer in range(0, num_layers):
            mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity)
            mappings.append(mapping)
            representations.append(representation)
    else:
        for layer in range(0, num_layers):
            mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity,
                                          precomp_mappings[layer])
            representations.append(representation)
        mappings = precomp_mappings
    return mappings, representations
