import matplotlib.pyplot as plt
import re


def parse_log(file_path):
    epochs = []
    train_acc = []
    test_acc = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = 0
    for line in lines:
        # 匹配 Epoch 数 (兼容两种日志格式)
        epoch_match = re.search(r"Epoch\s+(\d+)", line, re.IGNORECASE)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # 匹配训练准确率
        if "Train Instance Accuracy" in line:
            acc = float(re.findall(r"0\.\d+", line)[0])
            train_acc.append(acc)
            epochs.append(current_epoch)

        # 匹配测试准确率
        if "Test Instance Accuracy" in line:
            acc = float(re.findall(r"0\.\d+", line)[0])
            test_acc.append(acc)

    return epochs[:len(test_acc)], train_acc[:len(test_acc)], test_acc


# 解析文件 (请确保文件名与你上传的一致)
epochs1, train1, test1 = parse_log('Pointnet_Pointnet2_pytorch-master/log/classification/2026-04-18_23-13/logs/pointnet_cls.txt')
epochs2, train2, test2 = parse_log('Pointnet_Pointnet2_pytorch-master/log/det/train_2026-04-18_23-02-15.log')

# 开始绘图
plt.figure(figsize=(10, 6), dpi=300)  # 高分辨率用于论文
plt.style.use('seaborn-v0_8-whitegrid')  # 使用清晰的学术风格

# 绘制实验1 (PointNet 原版格式)
plt.plot(epochs1, train1, label='Exp 1: PointNet Train Acc', color='#1f77b4', linestyle='--', alpha=0.6)
plt.plot(epochs1, test1, label='Exp 1: PointNet Test Acc', color='#1f77b4', linewidth=2)

# 绘制实验2 (带日期的日志格式)
plt.plot(epochs2, train2, label='Exp 2: PointNet-V2 Train Acc', color='#d62728', linestyle='--', alpha=0.6)
plt.plot(epochs2, test2, label='Exp 2: PointNet-V2 Test Acc', color='#d62728', linewidth=2)

# 装饰图形
plt.title('Training and Testing Accuracy Comparison',  fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Instance Accuracy', fontsize=12)
plt.legend(loc='lower right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(0.3, 1.0)  # 根据准确率范围调整

# 保存并展示
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()