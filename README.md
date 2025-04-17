# **LUCID**

In this repository, we provide an implementation for the paper  
["Guarding the Privacy of Label-Only Access to Neural Network Classifiers via iDP Verification"](https://arxiv.org/abs/2502.16519).  
The repository owner is `anan.kabaha@campus.technion.ac.il`.

## **Prerequisites**

```
Julia Version 1.11.3  
Gurobi 12.0  
Python 3.8.10  
Torch 2.4.1
```

## **Clone LUCID**

```bash
git clone https://github.com/ananmkabaha/LUCID.git
cd LUCID
```

## **Training parameters**

```
--workers       Number of workers to train in parallel  
--model_arch    The model's architecture: 2x50, 2x100, 4x30, or CNN  
--dataset       The dataset: adult, credit, crypto, or twitter  
--devices       Devices to train with, e.g., cpu or cuda:0,cuda:1,...  
--output_dir    Output directory to save the models  
--seed          Random seed (e.g., 42 or 666)  
--epochs        Number of training epochs  
--batch_size    Training batch size
```

## **Training examples**

```bash
python3 train_parallel.py --dataset adult --model_arch 2x50 --workers 224 --batch_size 1024 --devices cpu --output_dir ./model/

python3 train_parallel.py --dataset crypto --model_arch CNN --workers 100 --batch_size 100 --devices cpu --output_dir ./model/
```

## **Computing the iDP-DB bounds**

```
--dataset         Dataset: adult, credit, crypto, or twitter  
--model_arch      Model architecture: 2x50, 2x100, 4x30, or CNN  
--models_path     Path of the trained models  
--worker_timeout  Timeout per worker  
--timeout         Total timeout  
--workers_num     Number of workers to compute iDP-DB  
--tmp_path        Directory to save temporary files  
--s               Source class  
--t               Target class  
--seed            Random seed
```

### **Examples**

```bash
python3 Compute_iDP-DB.py --s 1 --t 2 --workers_num 32 --dataset adult --model_arch 2x50 --models_path ./model/ --worker_timeout 2400 --timeout 28800

python3 Compute_iDP-DB.py --s 1 --t 2 --workers_num 32 --dataset crypto --model_arch CNN --models_path ./model/ --worker_timeout 2400 --timeout 28800
```

## **LUCID Inference (iDP protected)**

```
--model_path   Path of the model trained on the full dataset  
--model_arch   Model architecture: 2x50, 2x100, 4x30, or CNN  
--dataset      Dataset: adult, credit, crypto, or twitter  
--bounds       iDP-DB bounds  
--eps          Privacy budget  
--device       Device to run inference on, e.g., cpu  
--seed         Random seed
```

### **Examples**

```bash
python3 LUCID_inference.py --dataset adult --bounds 0.15,0.17 --eps 1 --model_arch 2x50 --model_path ./model/adult.pth

python3 LUCID_inference.py --dataset crypto --bounds 0.85,1.30 --eps 1 --model_arch CNN --model_path ./model/crypto.pth
```
