import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import torch.fft

# 创建了一个命令行参数解析器，并定义了一系列命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True) # 布尔型参数，表示是否进行训练
parser.add_argument('--evaluate', type=bool, default=True) # 布尔型参数，表示是否进行评估
parser.add_argument('--dataset', type=str, default='random3') #  字符串参数，表示数据集名称
parser.add_argument('--window_size', type=int, default=12) # 整数参数，表示窗口大小，默认值为 12
parser.add_argument('--horizon', type=int, default=3) # 整数参数，表示预测的时间跨度，默认值为 3
parser.add_argument('--train_length', type=float, default=7) # 浮点数参数，表示训练集的长度，默认值为 7
parser.add_argument('--valid_length', type=float, default=2) # 浮点数参数，表示验证集的长度，默认值为 2
parser.add_argument('--test_length', type=float, default=1) #浮 点数参数，表示测试集的长度
parser.add_argument('--epoch', type=int, default=5) # 整数参数，表示训练的轮数为50次
parser.add_argument('--lr', type=float, default=1e-4) #浮点数参数，表示学习率
parser.add_argument('--multi_layer', type=int, default=5) #模型的多层结构
parser.add_argument('--device', type=str, default='cpu') #设备名称
parser.add_argument('--validate_freq', type=int, default=1) #验证的频率
parser.add_argument('--batch_size', type=int, default=32) #批量大小
parser.add_argument('--norm_method', type=str, default='z_score') #归一化方法
parser.add_argument('--optimizer', type=str, default='RMSProp') #优化器名称
parser.add_argument('--early_stop', type=bool, default=False) # 布尔型参数，表示是否启用早停机制
parser.add_argument('--exponential_decay_step', type=int, default=5) #指数衰减的步数
parser.add_argument('--decay_rate', type=float, default=0.5) #衰减率
parser.add_argument('--dropout_rate', type=float, default=0.5) # Dropout 比率
parser.add_argument('--leakyrelu_rate', type=int, default=0.2) # LeakyReLU 的负斜率


args = parser.parse_args() # 解析命令行参数，并将解析结果存储在 args 对象中
print(f'Training configs: {args}') # 打印训练配置，即命令行传入的参数值
data_file = os.path.join('dataset', args.dataset + '.csv') # 根据参数值构建数据文件路径
result_train_file = os.path.join('output', args.dataset, 'train') # 根据参数值构建训练结果保存路径
result_test_file = os.path.join('output', args.dataset, 'test') #
if not os.path.exists(result_train_file): # 如果训练结果保存路径不存在，则创建该路径
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
data = pd.read_csv(data_file).values # 读取数据文件的内容，并将其转换为 NumPy 数组

# split data
# 根据指定的训练集、验证集和测试集的比例，将数据集划分为训练数据、验证数据和测试数据。
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)
if __name__ == '__main__':# 用于指定当脚本直接运行时执行的代码块，而不是作为模块被导入时执行
    if args.train:  # 如果 args.train 的值为 True，则执行以下代码块
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')