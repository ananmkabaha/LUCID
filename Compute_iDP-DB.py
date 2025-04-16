import argparse
import warnings
import signal
from functools import partial
import numpy as np
from utils.help_functions import *
from utils.top_queue import TopQueue
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute iDP-DB code',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="crypto", help='twitter, crypto, adult, or credit')
    parser.add_argument('--model_arch', type=str, default="2x50", help='2x10, 2x50, 2x100, 4x30, or CNN')
    parser.add_argument('--models_path', type=str, default="./model/", help='path of the models')
    parser.add_argument('--worker_timeout', type=int, default=2400, help='worker timeout')
    parser.add_argument('--timeout', type=int, default=8*3600, help='total timeout')
    parser.add_argument('--workers_num', type=int, default=64, help='number of workers to obrain the iDP-DB')
    parser.add_argument('--tmp_path', type=str, default="/tmp/", help='directory for temporary files')
    parser.add_argument('--s', type=int, default=2, help='source class')
    parser.add_argument('--t', type=int, default=1, help='target class')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    dataset = args.dataset
    models_path = args.models_path
    model_arch = args.model_arch
    worker_timeout = args.worker_timeout
    workers_num = args.workers_num
    timeout = args.timeout
    tmp_path = args.tmp_path
    source_class = args.s
    target_class = args.t
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    token = str(time.time())
    Q = TopQueue()
    worker_items = [[] for _ in range(workers_num)]
    W, model = load_hyper_dataset(models_path, dataset)

    procs = start_workers(workers_num, tmp_path, token, dataset, models_path, model_arch, worker_timeout, source_class, target_class)
    signal.signal(signal.SIGINT, partial(signal_handler, [procs, tmp_path, token]))

    cl = np.ones(len(W.numpy()))
    Q.push(bound=np.Inf, list_of_indexes=cl, status="waiting")
    start_time = time.time()
    best_bound = np.inf
    finish = False
    while time.time()-start_time < timeout:
        for wi in range(1, workers_num+1):
            if worker_items[wi-1] == [] and not Q.is_empty():
                item = Q.pop()
                if item.status == "finished":
                    if (len(np.where(item.list_of_indexes == 1)[0]) == 1 or item.bound < RESOLUTION) and is_larger_than_being_analyzed(item.bound, worker_items):
                        best_bound = item.bound
                        finish = True
                        break
                    elif len(np.where(item.list_of_indexes == 1)[0]) == 1 or item.bound < RESOLUTION:
                        Q.push(bound=item.bound, list_of_indexes=item.list_of_indexes, status="finished")
                        continue
                    else:
                        S = split_list_with_kmeans(item.list_of_indexes, W, worker_items)
                        for s in S:
                            Q.push(bound=item.bound, list_of_indexes=s, status="waiting")
                        worker_items[wi - 1] = []
                elif item.status == "waiting":
                    print("Worker", wi, "analyzing", len(np.where(item.list_of_indexes == 1)[0]), "networks with parent bound of", item.bound)
                    create_hyper_network(model, W, item.list_of_indexes, tmp_path, token, wi)
                    worker_items[wi - 1] = item
            elif finish_signal(tmp_path, token, wi):
                result = min(get_result(tmp_path, token, wi), worker_items[wi - 1].bound)
                Q.push(bound=result, list_of_indexes=worker_items[wi - 1].list_of_indexes, status="finished")
                networks_number_in_item = len(np.where(worker_items[wi - 1].list_of_indexes == 1)[0])
                worker_items[wi - 1] = []
                clean(tmp_path, token, wi)
                print("Worker", wi, "finished analyzing", networks_number_in_item, "networks with bound of", result, "(best bound", str(get_best_bound(Q, worker_items))+")")
        if finish:
            break
        time.sleep(1)
    if not finish:
        best_bound = get_best_bound(Q, worker_items)

    stop_workers(procs, tmp_path, token)
    print("Best bound:", best_bound, "Total time:", time.time()-start_time)
    print("Finished")



















