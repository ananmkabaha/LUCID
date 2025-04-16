import time
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch
import numpy as np
import pickle
from subprocess import Popen
from kneed import KneeLocator
from yellowbrick.cluster import distortion_score
from sklearn.cluster import KMeans
RESOLUTION = 1e-9
def flatten_params(dict):
    if dict is None:
        return 0
    f = ()
    for _, t in dict.items():
        f = f + (t.view(-1),)
    return torch.cat(f)


def vec2dict(w, dict):
    d = {}
    i = 0
    for x, t in dict.items():
        l = len(t.view(-1))
        d[x] = w[i:i + l].view(t.size())
        i += l
    return d


def load_hyper_dataset(models_path, dataset, device='cpu'):
    print('device:', device)
    train_set = torch.load('./datasets/' + dataset + '/train.pth')
    N = len(train_set)
    path = models_path + dataset
    model = torch.load(path + '.pth', map_location=torch.device(device))
    if os.path.exists(path+'W_all.pth'):
        W = torch.load(path+'W_all.pth')
    else:
        W = (flatten_params(model).unsqueeze(0),)
        for i in range(N):
            print('\r{:5.2f}%'.format(100 * i / N), end='')
            model = torch.load(path + "_" + str(i) + '.pth', map_location=torch.device(device))
            W = W + (flatten_params(model).unsqueeze(0),)
        print('\rdone!')
        W = torch.cat(W, dim=0)
        torch.save(W, path+'W_all.pth')
    return W, model


def create_hyper_network(model, W, cl, tmp_path, token, wi):
    wmin, _ = W[cl == 1, :].min(dim=0)
    wmax, _ = W[cl == 1, :].max(dim=0)
    dmin = vec2dict(wmin, model)
    dmax = vec2dict(wmax, model)
    a = []
    for (i, j) in dmin.items():
        a.append(np.transpose(j.cpu().detach().numpy()))
    for i in a:
        pickle.dump(a, open(tmp_path + '/hypernetwork_min_box_'+str(wi)+'_'+str(token)+'.p', "wb"))
    a = []
    for (i, j) in dmax.items():
        a.append(np.transpose(j.cpu().detach().numpy()))
    for i in a:
        pickle.dump(a, open(tmp_path + '/hypernetwork_max_box_'+str(wi)+'_'+str(token)+'.p', "wb"))
    start_signal(tmp_path, token, wi)


def start_signal(tmp_path, token, wi):
    with open(tmp_path + '/start' + str(wi) + '_'+str(token) + '.txt', 'w') as f: f.write('')


def finish_signal(tmp_path, token, wi):
    return os.path.exists(tmp_path+'/finished' + str(wi) + '_'+str(token) + '.txt')


def get_result(tmp_path, token, wi):
    with open(tmp_path+'/results'+str(wi) + '_'+str(token) + '.txt', 'r') as file:
        content = file.read()
        return float(content)


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def clean(tmp_path, token, wi):
    remove_file(tmp_path + '/hypernetwork_min_box_' + str(wi) + '_'+str(token) + '.p')
    remove_file(tmp_path + '/hypernetwork_max_box_' + str(wi) + '_'+str(token) + '.p')
    remove_file(tmp_path + '/start' + str(wi) + '_'+str(token) + '.txt')
    remove_file(tmp_path + '/finished' + str(wi) + '_'+str(token) + '.txt')
    remove_file(tmp_path + '/results' + str(wi) + '_'+str(token) + '.txt')


def start_workers(workers_num, tmp_path, token, dataset, models_path, model_name, worker_timeout, source_class, target_class):
    procs = []
    for w_i in range(1, workers_num+1):
        with open(tmp_path+'/output_'+str(w_i)+'_'+str(token)+'.log', 'w') as f:
            proc = Popen(['julia', './utils/compute_bounds.jl', "--token", token, "--worker", str(w_i), "--ctag", str(source_class), "--ct", str(target_class)\
                          , "--dataset", dataset, "--model_path", models_path + dataset + ".p"\
                          , "--model_name", model_name, "--timout", str(worker_timeout)], stdout=f, stderr=f)
        print('julia', './utils/compute_bounds.jl', "--token", token, "--worker", str(w_i), "--ctag", str(source_class), "--ct", str(target_class)\
                          , "--dataset", dataset, "--model_path", models_path+dataset+".p"\
                          , "--model_name", model_name, "--timout", str(worker_timeout))
        procs.append(proc)
    return procs


def stop_workers(procs, tmp_path, token):
    for wi, p in enumerate(procs):
        p.terminate()
        time.sleep(0.01)
        clean(tmp_path, token, wi+1)
        remove_file(tmp_path + '/output_' + str(wi+1) + '_' + str(token) + '.log')

def signal_handler(data, signal, frame):
    print("Ctrl-c was pressed. cleaning")
    stop_workers(data[0], data[1], data[2])
    exit(1)


def split_list(indexes_list, networks_num):
    cl1 = np.zeros(networks_num)
    cl2 = np.zeros(networks_num)
    inds = np.where(indexes_list == 1)[0]
    mid = len(inds) // 2
    cl1[inds[:mid]] = 1
    cl2[inds[mid:]] = 1
    print(cl1)
    print(cl2)
    return [cl1, cl2]


def split_list_with_kmeans(indexes_list, W, worker_items):
    networks_num = len(W.numpy())
    inds = np.where(indexes_list == 1)[0]
    w_rel = W[inds,:]
    distortion_scores = []
    distortion_scores_indecis = []
    for k in [2, 4, 8]:
        if k < len(w_rel):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(w_rel)
            cl = kmeans.labels_
            distortion_scores.append(distortion_score(w_rel.numpy(), cl))
            distortion_scores_indecis.append(k)
    try:
        kl = KneeLocator(x=np.array(distortion_scores_indecis), y=distortion_scores, curve='convex', direction='decreasing', S=1)
        k_to_use = min(kl.knee, 2)
    except:
        k_to_use = 2

    available_workers = sum(1 for worker_item in worker_items if worker_item == [])
    k_to_use = min(max(available_workers, k_to_use), len(inds))
    kmeans = KMeans(n_clusters=k_to_use, random_state=0, n_init="auto").fit(w_rel)
    cl = kmeans.labels_
    clks = []
    for k in range(k_to_use):
        clk = np.zeros(networks_num)
        clk[inds[cl == k]] = 1
        clks.append(clk)
    return clks


def is_larger_than_being_analyzed(bound, worker_items):
    for item in worker_items:
        if item != [] and bound < item.bound:
            return False
    return True


def get_best_bound(Q, worker_items):
    if all(item == [] for item in worker_items) and Q.is_empty():
        return np.inf
    elif not Q.is_empty():
        item = Q.pop()
        Q.push(bound=item.bound, list_of_indexes=item.list_of_indexes, status=item.status)
        best_bound = item.bound
    else:
        best_bound = 0
    for item in worker_items:
        if item != [] and best_bound < item.bound:
            best_bound = item.bound
    return best_bound
