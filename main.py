import yaml
from src.dataset import Dataset
# 读取配置文件
with open("configs/example.yaml", "r") as file:
    config = yaml.safe_load(file)

# 将配置字典传入 Dataset 类
dataset_loader = Dataset(config)
train_dataset, validation_dataset = dataset_loader.load_data()

print(validation_dataset[:1])  # 显示前1条训练数据