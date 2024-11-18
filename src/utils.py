import os
import pickle
import logging

SAVE_DIR = "/home/yw699/codes/LLM-Hallu/results"

def save(object, file, save_dir=SAVE_DIR):

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file)

    with open(save_path, 'wb') as f:
        pickle.dump(object, f)

    print(f"File saved locally at: {save_path}")


def read(file, save_dir=SAVE_DIR):
    save_path = os.path.join(save_dir, file)
    with open(save_path, "rb") as f:  # 以二进制读取模式打开文件
        content = pickle.load(f)  # 使用 pickle 加载文件内容
    print(content)


def save_wandb(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')



def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG



def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


