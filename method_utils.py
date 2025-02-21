import numpy as np
import random
import math
import os
from tqdm import tqdm

def load_dataset(dataset_name, loc):
    if loc == 'temporal_networks':
        filename = 'temporal_networks/' + dataset_name

        n_snapshots = 0
        for file in os.listdir(filename + '/ver'):
            if file.endswith('.csv'):
                n_snapshots += 1

        vertices_dir = filename + '/ver'
        vertices = []
        for n in range(n_snapshots):
            ver = np.genfromtxt(vertices_dir + f'/ver_{n}.csv', delimiter=',')
            vertices.append(ver)
        edges_dir = filename + '/edg'
        edges = []
        for n in range(n_snapshots):
            edg = np.genfromtxt(edges_dir + f'/edg_{n}.csv', delimiter=',')
            edges.append(edg)
        adj_dir = filename + '/adj'
        adj_matrices = []
        for n in range(n_snapshots):
            adj = np.genfromtxt(adj_dir + f'/adj_{n}.csv', delimiter=',')
            adj_matrices.append(adj)

        try:
            filename = filename + '/labels.csv'
            labels = np.genfromtxt(filename, delimiter=',')
            print(labels)
        except:
            labels = None

        return np.array(vertices), edges, np.array(adj_matrices), np.array(labels)
    
    elif loc == 'datasets':
        filename = 'datasets/' + dataset_name

        n_snapshots = 0
        for file in os.listdir(filename):
            if file.endswith('.csv'):
                n_snapshots += 1

        adj_matrices = []
        for n in range(n_snapshots):
            adj = np.genfromtxt(filename + f'/adj_{n}.csv', delimiter=',')
            adj_matrices.append(adj)

        try:
            filename = 'datasets/ground_truth/' + dataset_name + '.csv'
            labels = np.genfromtxt(filename, delimiter=',')
        except:
            labels = None

        return None, None, np.array(adj_matrices), labels

def get_generator(adj_matrix):
    generator = np.zeros(shape=adj_matrix.shape)
    for i in range(len(adj_matrix)):
        deg = np.sum(adj_matrix[i])
        if deg != 0:
            generator[i,:] = adj_matrix[i,:]/(deg*deg)
            generator[i, i] = -1.0 / deg
    return generator

def get_generators(adj_matrices):
    if isinstance(adj_matrices, np.ndarray) and adj_matrices.ndim == 2:
        generator = get_generator(adj_matrices)
        return generator
    else:
        L = []
        for n in range(len(adj_matrices)):
            generator = get_generator(adj_matrices[n])
            L.append(generator)
        return np.array(L)

def get_distances(input_matrices, norm='fro'):
    distances = np.zeros(shape=(len(input_matrices), len(input_matrices)))
    for i in tqdm(range(len(input_matrices)), desc="Computing distances", ascii=True):
        for j in range(i+1):
            distances[i, j] = np.linalg.norm(input_matrices[i]-input_matrices[j], ord=norm)
            distances[j, i] = distances[i, j]
    return np.array(distances)
    
def get_gramian(distances, sigma):
    return np.exp(-(distances**2 / (2*sigma**2)))

def get_transition_matrix(adj_matrix):
    Q = np.zeros(shape=(len(adj_matrix), len(adj_matrix)))
    row_sums = np.sum(adj_matrix, axis=1)
    for n in range(len(adj_matrix)):
        Q[n, :] = adj_matrix[n, :] / row_sums[n]
    return Q

def get_invariant_measures(adj_matrices):
    invariant_measures = []
    for n in range(len(adj_matrices)):
        deg_squared = np.sum(adj_matrices[n], axis=1) ** 2
        Z = np.sum(deg_squared)
        inv_measure = deg_squared / Z
        invariant_measures.append(np.array(inv_measure))
    return np.array(invariant_measures)

def normalize_distances(distances):
    min = np.min(distances)
    distances = distances + min
    max = np.max(distances)
    distances = distances / max
    return distances, min, max