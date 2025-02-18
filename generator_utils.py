import numpy as np
import random
import math
import os

# c[j], p[j], q[j], m[j], n[j] - lists of parameters for all wells within Phase j
# a[j], b[j] - real numbers; corrective component within Phase j
# Example: if there are three wells (i=1,2,3) during the Phase 1, then the parameters c[1], p[1], q[1], m[1], n[1] for the wells 1,2,3 are given with
#   c[1] = [c1, c2, c3], ..., n[1] = [n1, n2, n3] (lengths of all parameter arrays should be the same)
def potential(X, Y, c, p, q, m, n, a, b):
    res = 0
    wells_number = len(c)
    for i in range(wells_number):
        res += c[i]*np.exp(-(m[i]*X + p[i])**2 - (n[i]*Y + q[i])**2)
    res += a*X**2 / 2 + b*Y**2 / 2
    return res

def potential_gradient(x, c, p, q, m, n, a, b):
    dUx0 = 0
    dUx1 = 0
    wells_number = len(c)
    for i in range(wells_number):
        dUx0 += -2*c[i]*m[i]*(m[i]*x[:,0]+p[i])*np.exp(-(m[i]*x[:,0]+p[i])**2-(n[i]*x[:,1]+q[i])**2)
    dUx0 += a*x[:,0]
    for i in range(wells_number):
        dUx1 += -2*c[i]*n[i]*(n[i]*x[:,1]+q[i])*np.exp(-(m[i]*x[:,0]+p[i])**2-(n[i]*x[:,1]+q[i])**2)
    dUx1 += b*x[:,1]
    gradient = np.array([dUx0, dUx1]).T
    return gradient

def evaluation_func(x, nu, theta, omega, xi):
    return 1 - xi*(1 - nu*(-np.arctan(theta*x-omega)/math.pi + 0.5))

def particle_trajectory(params):
    n_particles, start_pos, dt, beta, T, alpha, _, c, p, q, m, n, a, b, _, _, _, _, _, durations = params
    time_steps = int(T/dt)
    durations.append(time_steps-np.sum(durations))
    x = start_pos
    trajectory = np.zeros(shape=(time_steps, n_particles, 2))
    all_labels = []
    counter = 0
    label = 0
    for phase in range(len(durations)):
        for _ in range(durations[phase]):
            dW = np.random.normal(0, np.sqrt(dt), size=(n_particles, 2))
            x += -alpha*potential_gradient(x, c[phase], p[phase], q[phase], m[phase], n[phase], a[phase], b[phase]) * dt + np.sqrt(2/beta) * dW
            trajectory[counter,:,:] = x
            all_labels.append(label)
            counter += 1
        label += 1
    return np.array(trajectory), np.array(all_labels)

def connect_vertices(vertices, params):
    n_particles, _, _, _, _, _, _, _, _, _, _, _, _, _, nu, theta, omega, xi, min_degree, _ = params
    edges = []
    adj_matrix = np.zeros(shape=(n_particles, n_particles))
    for i, u in enumerate(vertices):
        for j, v in enumerate(vertices):
            if i < j:
                rand = random.random()
                distance = np.linalg.norm(u-v)
                eval = evaluation_func(distance, nu, theta, omega, xi)
                if rand < eval:
                    new_edge = [i, j]
                    edges.append(new_edge)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

    if min_degree > 0:
        for i, v in enumerate(vertices):
            current_degree = np.sum(adj_matrix[i])
            if current_degree <= min_degree:
                distances = np.linalg.norm(vertices - v, axis=1)
                distances[i] = np.inf
                nearest_indices = np.argsort(distances)
                
                close_vert = []    
                for j in nearest_indices:
                    if adj_matrix[i, j] == 0 and i != j:
                        close_vert.append(j)
                    if len(close_vert) + current_degree == min_degree:
                        break
                for j in close_vert:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                    edges.append((min(i,j),max(i,j)))
        edges.sort(key=lambda x: (x[0], x[1]))

    return np.array(edges), adj_matrix

def generate_temporal_network(params):
    _, _, _, _, _, _, frequency, _, _, _, _, _, _, _, _, _, _, _, _, _ = params

    vertices = []
    edges = []
    adj_matrices = []
    labels = []

    trajectory, all_labels = particle_trajectory(params)
    time_steps = len(trajectory)

    i = 0
    while i*frequency < time_steps:
        current_vert = trajectory[i*frequency,:,:]
        labels.append(all_labels[i*frequency])

        # Save vertices
        vertices.append(current_vert)

        # Add and save edges
        current_edges, current_adj_matrix = connect_vertices(current_vert, params)
        edges.append(current_edges)
        
        # Save adjacency matrices
        adj_matrices.append(current_adj_matrix)

        i += 1

    return np.array(vertices), edges, np.array(adj_matrices), np.array(labels)

def save_params(params, filename):
    n_particles, _, dt, beta, T, alpha, frequency, c, p, q, m, n, a, b, nu, theta, omega, xi, min_degree, durations = params
    filename = filename + '/parameters.txt'
    with open(filename, "w") as file:
        file.write(f"n_particles: {n_particles}\n")
        file.write(f"Gradient Search Parameters:\n")
        file.write(f"  dt: {dt}\n")
        file.write(f"  beta: {beta}\n")
        file.write(f"  T: {T}\n")
        file.write(f"  alpha: {alpha}\n\n")
        file.write(f"Resolution parameter (frequency): {frequency}\n\n")
        file.write(f"Potential Function Parameters:\n")
        file.write(f"  c: {c}\n")
        file.write(f"  p: {p}\n")
        file.write(f"  q: {q}\n")
        file.write(f"  m: {m}\n")
        file.write(f"  n: {n}\n\n")
        file.write(f"Corrective Parameters:\n")
        file.write(f"  a: {a}\n")
        file.write(f"  b: {b}\n\n")
        file.write(f"Evaluation Function Parameters:\n")
        file.write(f"  nu: {nu}\n")
        file.write(f"  theta: {theta}\n")
        file.write(f"  omega: {omega}\n")
        file.write(f"  xi: {xi}\n\n")
        file.write(f"Minimum Degree: {min_degree}\n\n")
        file.write(f"Durations: {durations}\n")
    print(f"Parameters saved to {filename}")

def save_temporal_network(temporal_network_name, params, vertices, edges, adj_matrices, labels):
        filename = './temporal_networks' + '/' + temporal_network_name
        os.makedirs(filename, exist_ok=True)

        save_params(params, filename)

        os.makedirs(filename + '/ver', exist_ok=True)
        for n in range(len(vertices)):
            np.savetxt(filename + f'/ver/ver_{n}.csv', vertices[n], delimiter=',')
        
        os.makedirs(filename + '/edg', exist_ok=True)
        for n in range(len(edges)):
            np.savetxt(filename + f'/edg/edg_{n}.csv', edges[n], delimiter=',')

        os.makedirs(filename + '/adj', exist_ok=True)
        for n in range(len(adj_matrices)):
            np.savetxt(filename + f'/adj/adj_{n}.csv', adj_matrices[n], delimiter=',')
            
        np.savetxt(filename + '/labels.csv', labels, delimiter=',')
        