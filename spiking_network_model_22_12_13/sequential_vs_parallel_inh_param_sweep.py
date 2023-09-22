import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as copy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
import pickle
import pandas as pd
import json

value_range = 256
color_list = [
	(0,   '#ad262f'),
	(0.5,   'black'),
	(1,   'yellow'),
]
activity_colormap = LinearSegmentedColormap.from_list('activity', color_list, N=value_range)

hsv_cmap = matplotlib.colormaps['hsv']

def write_data(write_path, data):
	f = open(write_path, 'wb')
	pickle.dump(data, f)
	f.close()
	
def load_data(load_path):
	with open(load_path, 'rb') as f:
		return pickle.load(f)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def p_active(w, x, b, g):
	return sigmoid(g * (w * x - b))

def step(x, w, b, g, w_r, t=0):
	p = p_active(w, x[t, :], b + np.sum(w_r * x[t, :]), g)
	x[t + 1, 1:] =  (np.random.rand(p.shape[0]) < p).astype(int)[:-1]
	x[t + 1, 0] = 0
	
def run_activation(w, b, g, w_r):
	x = np.zeros((w.shape[0], w.shape[0])).astype(int)
	x[0, 0] = 1
	for t in range(w.shape[0] - 1):
		step(x, w, b, g, w_r, t)
	return x, w

def make_w_transform_homeostatic(rate, setpoint):
	def w_transform_homeostatic(w, x):
		x_total_activity = np.sum(x, axis=0)
		w[:-1] += rate * (setpoint - x_total_activity[1:])
		return w
	return w_transform_homeostatic

def make_w_transform_seq(rate, setpoint):
	def w_transform_seq(w, x):
		for i in range(len(w) - 1):
			w[i] += rate * np.sum(x[:-1, i] * x[1:, i+1] - x[:-1, i+1] * x[1:, i])
		w = np.minimum(w, setpoint)
		return w
	return w_transform_seq

def make_w_r_transform(rate, setpoint):
	def w_r_transform(w_r, x):
		w_r += rate * x.sum(axis=0)
		w_r = np.minimum(w_r, setpoint)
		return w_r
	return w_r_transform

def make_dropout(p):
	def dropout(w, w_r):
		fracs_remaining = np.random.binomial(100, 1-p, size=w.shape[0]) / 100
		return fracs_remaining * w, fracs_remaining * w_r
	return dropout      

def run_n_activations(w_0, b, g, w_r_0, n, w_transform=None, w_r_transform=None, dropout_iter=1000, dropout_func=None):
	all_weights = []
	w = copy(w_0)
	w_r = copy(w_r_0)
	sequential_activity = []
	all_activity = []
	sequential_activity_with_blowups = []
	for i in range(n):
		if i == dropout_iter:
			w, w_r = dropout_func(w, w_r)
		x, w = run_activation(w, b, g, w_r)
		
		x_seq = np.zeros((x.shape[0]))
		s = 0
		while s < len(x_seq) and x[s, s] == 1 and (x[s, :s] == 0).all() and (x[s, s+1:] == 0).all():
			x_seq[s] = 1
			s += 1
		sequential_activity.append(x_seq)
		all_activity.append(copy(x))
		sequential_activity_with_blowups.append(sequential_activity[-1])
		
		if i > dropout_iter and w_transform is not None:
			w = w_transform(w, x)    
			
		if i > dropout_iter and w_r_transform is not None:
			w_r = w_r_transform(w_r, x) 
			
		all_weights.append(copy(w))
	return np.array(sequential_activity), np.array(all_activity), np.array(sequential_activity_with_blowups), np.array(all_weights)
	
def extract_lengths(X):
	l = np.zeros(X.shape[0])
	x_prod = np.ones(X.shape[0])
	for i in range(X.shape[1]):
		l += (X[:, i] * x_prod)
		x_prod *= X[:, i]
	l = np.where(X[:, 0] < 0, 0, l)
	return l

def extract_first_hitting_times(X_all, benchmark_lens, start=10):
	all_times = []
	for i_X, X in enumerate(X_all):
		times = np.nan * np.ones(len(benchmark_lens))
		counter = 0
		ls = extract_lengths(X)
		for j, l in enumerate(ls[start:]):
			if counter < len(times) and l >= benchmark_lens[counter]:
				while counter < len(times) and l >= benchmark_lens[counter]:
					times[counter] = j
					counter += 1
		all_times.append(times)
	return np.array(all_times)

def extract_jumps(X_all):
	all_hitting_times = extract_first_hitting_times(X_all, np.arange(1, 51))
	all_jump_size_counts = []
	
	for hitting_times in all_hitting_times:
		
		last_hitting_time = None
		jump_sizes_count = np.zeros((50,))
		jump_size = 0
		for i, hitting_time in enumerate(shave_front_zeros_except_last(hitting_times)):
			if last_hitting_time is None:
				last_hitting_time = hitting_time
			elif last_hitting_time == hitting_time:
				jump_size += 1
			else:
				jump_sizes_count[jump_size] += 1
				jump_size = 1
			last_hitting_time = hitting_time
		jump_sizes_count[jump_size] += 1
		all_jump_size_counts.append(jump_sizes_count)
	return np.array(all_jump_size_counts)

def determine_recovered(X_all, n_activations=20, threshold=0.8, n_cells=50):
	recovered_vec = np.zeros((len(X_all),))
	for i_X, X in enumerate(X_all):
		ls = extract_lengths(X)
		recovered = (np.count_nonzero(ls[-n_activations:] == n_cells) / n_activations) > threshold
		recovered_vec[i_X] = recovered
	return recovered_vec 

def shave_front_zeros_except_last(arr):
	for i, x in enumerate(arr):
		if x != 0:
			if i == 0:
				return arr
			else:
				return arr[i-1:]
	return np.array([])
	

if __name__ == '__main__':
	run_name = 'param_sweep_4'

	w_0 = 10 * np.ones(50)

	bias_w_r_start = np.array([5, 0])
	bias_w_r_end = np.array([0, 5])
	bias_w_r_points = 5
	dropout_percentages = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
	learning_rates = [0.1, 1, 10]
	inh_learning_rate = 0.01
	n_networks = 30
	g = 2

	total_points = bias_w_r_points * len(learning_rates) * len(dropout_percentages) * 2
	print('total points:', total_points)

	df = None
	point_count = 0

	for p in np.linspace(0, 1, bias_w_r_points):
		w_r_b_vec = bias_w_r_start * (1-p) + bias_w_r_end * p
		w_r_0, b = w_r_b_vec[0], w_r_b_vec[1]
		w_r = w_r_0 * np.ones(50)
		
		for rate in learning_rates:
			for dropout in dropout_percentages: 
				X_ctrl = []
				for i in range(n_networks):
					sa, nsa, sawbu, ws = run_n_activations(w_0, b, g, w_r, 300, None, None, dropout_iter=10, dropout_func=make_dropout(dropout))
					X_ctrl.append(sawbu)
				X_ctrl = np.array(X_ctrl)
					
				data = {
					'rule': ['none'],
					'rate': [rate],
					'w_r': [w_r_0],
					'b' : [b],
					'dropout': [dropout],
					'activations': [list(X_ctrl.flatten().astype(int))],
					'activations_shape': [X_ctrl.shape],
				}
				
				if df is None:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False)
				else:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False, mode='a', header=False)
					
				print(point_count)
				point_count += 1
				
				
				X_homeo = []
				for i in range(n_networks):
					sa, nsa, sawbu, ws = run_n_activations(w_0, b, g, w_r, 300,  make_w_transform_homeostatic(rate, 1), make_w_r_transform(inh_learning_rate, w_r_0), dropout_iter=10, dropout_func=make_dropout(dropout))
					X_homeo.append(sawbu)
				X_homeo = np.array(X_homeo)
					
				data = {
					'rule': ['homeostatic'],
					'rate': [rate],
					'w_r': [w_r_0],
					'b' : [b],
					'dropout': [dropout],
					'activations': [list(X_homeo.flatten().astype(int))],
					'activations_shape': [X_homeo.shape],
				}
				
				if df is None:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False)
				else:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False, mode='a', header=False)
					
				print(point_count)
				point_count += 1
					

				X_stdp = []
				for i in range(n_networks):
					sa, nsa, sawbu, ws = run_n_activations(w_0, b, g, w_r, 300, make_w_transform_seq(rate, 10), make_w_r_transform(inh_learning_rate, w_r_0), dropout_iter=10, dropout_func=make_dropout(dropout))
					X_stdp.append(sawbu)
				X_stdp = np.array(X_stdp)
				
				data = {
					'rule': ['stdp'],
					'rate': [rate],
					'w_r': [w_r_0],
					'b' : [b],
					'dropout': [dropout],
					'activations': [list(X_stdp.flatten().astype(int))],
					'activations_shape': [X_stdp.shape],
				}
				
				if df is None:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False)
				else:
					df = pd.DataFrame(data)
					df.to_csv(f'data/{run_name}.csv', index=False, mode='a', header=False)
					
				print(point_count)
				point_count += 1