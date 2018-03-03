import sys
import json
import os
import numpy as np
import tensorflow as tf

N_MAX = 1500

def get_id_map(file_path,N):
	# file_path: path to good_id_to_old_id.json
	if not os.path.exists(file_path):
		dic = {}
		fp = file_path.replace('good_id_to_old_id.json','nodelist.txt')
		if not os.path.exists(fp):
			ans = {}
			for i in range(N_MAX):
				ans[i+1]=i
			return ans
		with open(fp, 'r') as f2:
			l = f2.read().splitlines()
			for j in range(len(l)):
				dic[j+1] = int(l[j])
		with open(file_path, 'w') as outfile:
			json.dump(dic, outfile)
	f = open(file_path, 'r')
	dic = json.loads(f.read())
	inv_map = {v: int(k)-1 for k, v in dic.iteritems()}
	return inv_map

def get_adj(file_path, inv_map, N):
	# file_path: path to edgelist.txt
	f = open(file_path, 'r')
	adj = np.zeros((N,N))
	for line in f.read().splitlines():
		l = line.split() 
		if inv_map[int(l[0])]<N and inv_map[int(l[1])]<N:
			adj[inv_map[int(l[0])]][inv_map[int(l[1])]] = 1.0
	return adj

def get_history(file_path, inv_map, N):
	# file_path: path to opinion.txt
	f = open(file_path, 'r')
	lines = f.read().splitlines()
	#if len(lines) > 15005:
		#lines = lines[:15000]
	H = []
	lines = [[inv_map[int(line.split()[0])], int(line.split()[1]), float(line.split()[2])] for line in lines if inv_map[int(line.split()[0])]<N]
	H = sorted(lines, key = lambda line: line[1])
	return H

def make_adj_H(H,N):
	G = [[] for i in range(N)]
	for line in H:
		G[line[0]].append([line[2], line[1]])
	return G

def make_opinions(file_path,H,inv_map,N):
	if not os.path.exists(file_path):
		return get_m(H,N)
	f = open(file_path,'r')
	lines = f.read().splitlines()
	X = [ [] for i in range(N)]
	for i in range(N):
		X[i] = np.zeros(len(H))
	t = dict()
	for i in range(len(H)):
		ops = lines[i].split(' ')
		if inv_map[int(ops[0])] in t.keys():
			continue
		t[inv_map[int(ops[0])]] = 'True'
		X[inv_map[int(ops[0])]] = np.array([float(x) for x in ops[1:]],dtype=np.float32)
	return np.transpose(X)

def get_index(H, t):
	l = 0
	r = len(H) - 1
	if len(H) == 0:
		return 0
	if H[l][1] == t:
		return l
	while r-l > 1:
		mid = int((l + r) / 2)
		if H[mid][1] < t:
			l = mid
		else:
			r = mid
	return r

def get_delta_t(H, t):
	ind = get_index(H, t)
	if ind == 0:
		return 0
	return H[ind][1] - H[ind-1][1]
	
def get_delta_m(G, x, N):
	t = x[1]
	m = x[2]
	delta_m = [0.0] * N
	for i in range(N):
		ind = get_index(G[i], t)
		if ind > 0:
			delta_m[i] = abs(m-G[i][ind-1][0])
	return delta_m

def get_on_hot(number, N):
    on_hot = [0.0] * N
    on_hot[number] = 1.0
    return on_hot

def make_input(x, x_prev, N, G):
    user = get_on_hot(x[0], N)
    dt = [(x[1]-x_prev[1])*1.0/3600.0] * N
    dm = get_delta_m(G, x, N)
    return user, dt, dm

def make_input2(x, x_prev, N):
    user = get_on_hot(x[0], N)
    dt = [(x[1]-x_prev[1])*1.0/3600.0] * N
    return user, dt

def get_t_u_index_user(H, N, adj):
	t_u = []
	index = []
	user = []
	temp1 = [-1] * N
	temp2 = [0] * N
	temp3 = [0] * N
	t_u.append(temp1)
	index.append(temp2)
	user.append(temp3)
	for i in range(len(H)-1):
		x = H[i]
		sender = x[0]
		tstamp = x[1]
		t1 = [item for item in t_u[len(t_u)-1]]
		t2 = [item for item in index[len(t_u)-1]]
		t3 = [item for item in user[len(t_u)-1]]
		for to in range(N):
			if adj[sender][to]==1:
				t1[to] = tstamp
				t2[to] = i+1
				t3[to] = sender
		t_u.append(t1)
		index.append(t2)
		user.append(t3)
	for i in range(len(H)):
		for j in range(N):
			if t_u[i][j] == -1:
				t_u[i][j] = H[i][1]
			t_u[i][j] = ((H[i][1] - t_u[i][j])*1.0)/3600.0
	return t_u, index, user

def get_m_H (H):
	ans = []
	for i in range(len(H)):
		ans.append(H[i][2])
	return ans

def get_m(H, N):
	ans = []
	for i in range(len(H)):
		t=[]
		if len(ans)>0:
			t = [item for item in ans[len(ans)-1]]
		else:
			t = [0] * N
		x = H[i]
		sender = x[0]
		msg = x[2]
		t[sender] = msg
		ans.append(t)
	return ans


