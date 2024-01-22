import os
import ast
import math
import numpy as np
import argparse

num_tasks = {
    'seq-mnist': 5,
    'seq-cifar10': 5,
    'seq-cifar100': 10,
    'seq-tinyimg': 10,
}


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./data/results')
parser.add_argument('--dataset', type=str, default='seq-mnist')
parser.add_argument('--model', type=str, default='derpp')
parser.add_argument('--log-name', type=str, default='logs')
parser.add_argument('--num-seeds', type=int, default=10)
args = parser.parse_args()

class_il = {'seeds': [], 'avg_accs': [], 'forget': []}
task_il = {'seeds': [], 'avg_accs': [], 'forget': []}

cil_filepath = '/'.join([args.root, 'class-il', args.dataset, args.model, '{}.pyd'.format(args.log_name)])
til_filepath = '/'.join([args.root, 'task-il', args.dataset, args.model, '{}.pyd'.format(args.log_name)])

with open(cil_filepath, 'r') as f:
    lines = f.readlines()[-args.num_seeds:]
    for data in lines:
        dic = ast.literal_eval(data)
        class_il['seeds'].append(dic['seed'])
        class_il['avg_accs'].append(dic['accmean_task{}'.format(num_tasks[args.dataset])])
        class_il['forget'].append(dic['forgetting'])
    class_il['total_avg_accs'] = (np.mean(class_il['avg_accs']), np.std(class_il['avg_accs']))
    class_il['total_forget'] = (np.mean(class_il['forget']), np.std(class_il['forget']))

with open(til_filepath, 'r') as f:
    lines = f.readlines()[-args.num_seeds:]
    for data in lines:
        dic = ast.literal_eval(data)
        task_il['seeds'].append(dic['seed'])
        task_il['avg_accs'].append(dic['accmean_task{}'.format(num_tasks[args.dataset])])
        task_il['forget'].append(dic['forgetting'])
    task_il['total_avg_accs'] = (np.mean(task_il['avg_accs']), np.std(task_il['avg_accs']))
    task_il['total_forget'] = (np.mean(task_il['forget']), np.std(task_il['forget']))


for seed, cls_avg_acc, cls_forget, task_avg_acc, task_forget in sorted(zip(
    class_il['seeds'], class_il['avg_accs'], class_il['forget'], task_il['avg_accs'], task_il['forget'])):
    print("{:3d} | {:5.2f} {:5.2f} {:5.2f} {:5.2f}".format(seed, cls_avg_acc, cls_forget, task_avg_acc, task_forget))

print("Statistic")
print("[Class-il] Total Avg Acc.: {:5.2f}({:.2f}) / Total Forgetting: {:5.2f}({:.2f})".format(
    class_il['total_avg_accs'][0], class_il['total_avg_accs'][1], 
    class_il['total_forget'][0], class_il['total_forget'][1]))
print("[ Task-il] Total Avg Acc.: {:5.2f}({:.2f}) / Total Forgetting: {:5.2f}({:.2f})".format(
    task_il['total_avg_accs'][0], task_il['total_avg_accs'][1], 
    task_il['total_forget'][0], task_il['total_forget'][1]))
