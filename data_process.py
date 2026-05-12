import matplotlib.pyplot as plt
import re
import os


def parse_log(file_path):
    """解析日志文件，提取数据"""
    epochs, train_acc, test_acc = [], [], []
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file {file_path}")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 适配不同日志格式的 Epoch 块
    blocks = re.split(r'Epoch\s+\d+|\[\s+Epoch\s+\d+', content)[1:]
    for i, block in enumerate(blocks):
        t_match = re.search(r'Train Instance Accuracy:\s+(0\.\d+)', block)
        v_match = re.search(r'Test Instance Accuracy\s*:\s+(0\.\d+)', block)
        if t_match and v_match:
            epochs.append(i + 1)
            train_acc.append(float(t_match.group(1)))
            test_acc.append(float(v_match.group(1)))
    return epochs, train_acc, test_acc


def draw_and_save_english(epochs, train, test, title, filename, color):
    """绘制全英文学术图表"""
    if not epochs: return

    # 设置学术绘图风格
    plt.figure(figsize=(8, 5), dpi=300)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 绘制训练(虚线)和测试(实线)
    plt.plot(epochs, train, color=color, linestyle='--', alpha=0.4, label='Train Accuracy')
    plt.plot(epochs, test, color=color, linewidth=2, label=f'Test Accuracy (Best:{max(test):.4f})')

    # 英文标注
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Instance Accuracy', fontsize=12)

    # 设置坐标轴范围和刻度
    plt.ylim(0.3, 1.0)
    plt.legend(loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Successfully saved: {filename}")


# --- 执行绘图 ---

# 1. 第一个日志 (pointnet_cls.txt)
e1, tr1, te1 = parse_log('Pointnet_Pointnet2_pytorch-master/log/classification/2026-04-18_23-13/logs/pointnet_cls.txt')
draw_and_save_english(
    e1, tr1, te1,
    'Accuracy Curves of PointNet on Preprocessed ModelNet40 Dataset',
    'pointnet_training_plot.png',
    '#1f77b4'  # 蓝色
)

# 2. 第二个日志 (train_2026-04-18_23-02-15.log)
e2, tr2, te2 = parse_log('Pointnet_Pointnet2_pytorch-master/log/det/train_2026-04-18_23-02-15.log')
draw_and_save_english(
    e2, tr2, te2,
    'Accuracy Curves of PointNet-V2 on Preprocessed ModelNet40 Dataset',
    'pointnet_v2_training_plot.png',
    '#d62728'  # 红色
)