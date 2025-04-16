import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.models import *
from train import load_model, load_dataset
import warnings
warnings.filterwarnings("ignore")


def exponential_mechanism(logits, epsilon, bounds, delta_u=1):
    predicted_classes = torch.argmax(logits, dim=1)
    one_hot_outputs = F.one_hot(predicted_classes, num_classes=2)
    utilities = np.array([output for output in one_hot_outputs])
    probabilities = np.exp(epsilon * utilities / (2 * delta_u))
    probabilities = [t.numpy() for t in probabilities]
    probabilities = np.stack(probabilities, axis=0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    for index, p in enumerate(probabilities):
        predicted_class = predicted_classes[index]
        sorted_logits, _ = torch.sort(logits[index], descending=True)
        max_logit = sorted_logits[0]
        second_max_logit = sorted_logits[1]
        confidence = max_logit - second_max_logit
        if confidence <= bounds[predicted_class]:
            selected_index = np.random.choice(np.array([0, 1]), p=p)
            one_hot_outputs[index] = F.one_hot(torch.tensor(selected_index), num_classes=2)
    protected_predicted_classes = torch.argmax(one_hot_outputs, dim=1)
    return protected_predicted_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LUCID iDP inference',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default="./model/crypto.pth", help='model path')
    parser.add_argument('--model_arch', type=str, default="2x50", help='2x50, 2x100, 4x30, or CNN')
    parser.add_argument('--dataset', type=str, default="crypto", help='dataset: twitter, crypto, adult, or credit.')
    parser.add_argument('--bounds', type=str, default="0.04,0.04", help='iDP-DB bounds')
    parser.add_argument('--eps', type=float, default=0, help='privacy budget')
    parser.add_argument('--device', type=str, default="cpu", help='device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    model_path = args.model_path
    model_arch = args.model_arch
    dataset = args.dataset
    bounds = list(map(float, args.bounds.split(",")))
    eps = args.eps
    device = args.device
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, test_set, dim = load_dataset(dataset)
    model = load_model(model_arch, dim)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    with torch.no_grad():
        model.eval()
        correct = 0
        correct_protected = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
            predicted_with_protection = exponential_mechanism(outputs.data, eps, bounds)
            correct_protected += (predicted_with_protection == labels.to(device)).sum()

        print('Test accuracy: %.6f %%' % (100 * float(correct) / total))
        print('Test accuracy (iDP protected): %.6f %%' % (100 * float(correct_protected) / total))





