import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.models import *
import warnings
warnings.filterwarnings("ignore")

def save_model(model, dataset, index_to_remove, output_dir):
    a = []
    for i in model.parameters():
        a.append(np.transpose(i.cpu().detach().numpy()))
    print("---------------", index_to_remove, "-----------------")
    for i in a:
        print(i.shape)
    if index_to_remove >= 0:
        model_path = "./"+output_dir+"/"+dataset+"_"+str(index_to_remove)
    else:
        model_path = "./" + output_dir + "/" + dataset

    pickle.dump(a, open(model_path + ".p", "wb"))
    torch.save(model.state_dict(), model_path + '.pth')


def load_model(model_arch, dim):
    if model_arch == "2x10":
        model = FC_2x10(dim)
    elif model_arch == "2x50":
        model = FC_2x50(dim)
    elif model_arch == "2x100":
        model = FC_2x100(dim)
    elif model_arch == "4x30":
        model = FC_4x30(dim)
    elif model_arch == "CNN":
        model = CNN(dim)
    else:
        assert ("New model arch has been detected, please expand models.py and this if condition.")
    return model


def load_dataset(dataset):
    if dataset == "adult":
        dim = 21
    elif dataset == "credit":
        dim = 23
    elif dataset == "crypto":
        dim = 7
    elif dataset == "twitter":
        dim = 15
    else:
        assert False, "New dataset has been detected, please expand the code to support it."

    train_set = torch.load('./datasets/' + dataset + '/train.pth')
    test_set = torch.load('./datasets/' + dataset + '/test.pth')

    return train_set, test_set, dim


def train(index_to_remove, dataset, model_arch, seed, output_dir, device='cpu', batch_size=1024, num_epochs=30, num_of_threads = 1, log=False):
    torch.set_num_threads(num_of_threads)
    train_set, test_set, dim = load_dataset(dataset)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_arch, dim)
    model = model.to(device)

    if index_to_remove >= len(train_set):
        return
    elif index_to_remove >= 0:
        x, _ = train_set[index_to_remove]
        train_set[index_to_remove] = (x, torch.tensor(-1, dtype=torch.int64))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)
    for epoch in range(num_epochs):
        total_batch = len(train_loader) // batch_size
        for i, (samples, labels) in enumerate(train_loader):
            mask = labels != -1
            X = samples[mask].to(device)
            Y = labels[mask].to(device)
            pre = model(X)
            cost = loss(pre, Y.long())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i + 1) % 30 == 0 and log:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (
                epoch + 1, num_epochs, i + 1, total_batch, cost.item()))
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
        print('Test accuracy: %.2f %%' % (100 * float(correct) / total))
    try:
        os.mkdir(output_dir)
    except:
        print("Folder exists.")
    save_model(model, dataset, index_to_remove, output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=str, default="2x50", help='2x50, 2x100, 4x30, or CNN')
    parser.add_argument('--output_dir', type=str, default="./model/", help='output directory to save the models')
    parser.add_argument('--dataset', type=str, default="adult", help='dataset: adult, credit, crypto, or twitter')
    parser.add_argument('--seed', type=int, default=666, help='random seed 42 or 666')
    parser.add_argument('--start', type=int, default=0, help='start model')
    parser.add_argument('--end', type=int, default=1, help='end model')
    parser.add_argument('--device', type=str, default="cpu", help='device to train with')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--num_of_threads', type=int, default=1, help='number of threads')

    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir
    model_arch = args.model_arch
    dataset = args.dataset
    seed = args.seed
    start = args.start
    end = args.end
    device = args.device
    num_of_threads = args.num_of_threads
    if start == 0:
        train(-1, dataset, model_arch, seed, output_dir, device, batch_size, num_epochs, num_of_threads)
    for j in range(start, end):
        index_to_remove = j
        train(index_to_remove, dataset, model_arch, seed, output_dir, device, batch_size, num_epochs, num_of_threads)

