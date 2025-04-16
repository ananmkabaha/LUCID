import numpy as np
import torch
import time
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parallel training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--workers', type=int, default=196, help='number of workers to train in parallel')
    parser.add_argument('--model_arch', type=str, default="2x50", help='2x50, 2x100, 4x30, or CNN')
    parser.add_argument('--dataset', type=str, default="adult", help='dataset: adult, credit, crypto, or twitter')
    parser.add_argument('--devices', type=str, default="cpu", help='devices to train with, for example cpu or cuda:0 or cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7')
    parser.add_argument('--output_dir', type=str, default="./model/", help='output directory to save the models')
    parser.add_argument('--seed', type=int, default=666, help='random seed for example 42 or 666')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

    args = parser.parse_args()
    workers = args.workers
    dataset = args.dataset
    model_arch = args.model_arch
    devices_list = args.devices.split(",")
    output_dir = args.output_dir
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size

    data_path = './datasets/' + dataset + '/'
    train_set = torch.load(data_path + 'train.pth')
    networks_per_device = int(np.ceil(len(train_set)/len(devices_list)))
    networks_per_device_per_worker = int(np.ceil(networks_per_device/workers))
    processes = []
    start_time = time.time()
    for i in range(len(devices_list)):
        print("networks to analyze in ", devices_list[i], "are from ", i*networks_per_device, "to", (i+1) * networks_per_device)
        for j in range(workers):
            print("networks to analyze by worker ", j, "in device", devices_list[i], "are from ",
                  i * networks_per_device + j * networks_per_device_per_worker, "to",
                  i * networks_per_device + (j+1) * networks_per_device_per_worker)

            cmd = [
                "python3", "./train.py",
                "--start", str(i * networks_per_device + j * networks_per_device_per_worker),
                "--end", str(i * networks_per_device + (j + 1) * networks_per_device_per_worker),
                "--model_arch", str(model_arch),
                "--dataset", str(dataset),
                "--device", str(devices_list[i]),
                "--seed", str(seed),
                "--batch_size", str(batch_size),
                "--epochs", str(epochs),
                "--output_dir", str(output_dir),
            ]

            print(cmd)

            process = subprocess.Popen(cmd)
            processes.append(process)

    for process in processes:
        process.wait()
    print("Total time is ", time.time()-start_time)
    print("All processes have finished!")

