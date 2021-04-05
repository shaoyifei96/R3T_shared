import pickle
import os
from pypolycontain.utils.random_polytope_generator import *
import copy

def load_polytopes_from_file(file_path, construct_zonotope=False):
    with open(file_path, 'rb') as f:
        polytopes_matrices = pickle.load(f)
    polytopes = []
    for pm in polytopes_matrices:
        if not construct_zonotope:
            polytopes.append(AH_polytope(pm[0], pm[1], polytope(pm[2], pm[3])))
        else:
            polytopes.append(zonotope(pm[0].reshape([-1,1]), pm[1]))
    return polytopes

def get_pickles_in_dir(dir_path):
    filenames = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".p"):
            filenames.append(filename)
    return sort_by_filename_time(filenames)

def sort_by_filename_time(file_list):
    times = []
    for file in file_list:
        times.append(float(file.split('_')[0]))
    # sort names from file
    times, file_list = list(zip(*sorted(zip(times, file_list))))
    return file_list, times

def get_polytope_sets_in_dir(dir_path, data_source='rrt'):
    if data_source == 'rrt':
        files, times = get_pickles_in_dir(dir_path)
        polytope_sets = []
        print('Loading files...')
        for f in files:
            print(f)
            polytopes = load_polytopes_from_file(dir_path + '/' + f, construct_zonotope=False)
            polytope_sets.append(copy.deepcopy(polytopes))
        print('Files loaded!')
        return polytope_sets, times
    elif data_source == 'mpc':
        filenames = []
        polytope_sets = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".p") or filename.endswith(".pkl"):
                filenames.append(filename)
        for f in filenames:
            print(f)
            polytopes = load_polytopes_from_file(dir_path + '/' + f, construct_zonotope=True)
            for i in [16, 8, 4, 2, 1]:
                polytope_sets.append(copy.deepcopy(polytopes[0:int(np.ceil(len(polytopes)/i))]))
        print('Files loaded!')
        return polytope_sets
    else:
        raise NotImplementedError

def save_polytope_to_dir(polytope_list, dir_path):
    polytope_list_clean = [[p.T, p.t, p.P.H, p.P.h] for p in polytope_list]
    with open(dir_path+'/polytopes.p', "wb") as f:
        pickle.dump(polytope_list_clean, f)
