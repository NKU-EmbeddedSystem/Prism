import os
import csv
import random
from typing import List

def read_file(file_path: str) -> List[float]:
    """读取文件内容并将其解析为浮点数列表"""
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        return list(map(float, line.strip('[]').split(', ')))

def write_to_csv(writer, label: int, *vectors):
    """将向量数据写入CSV"""
    row = []
    for vector in vectors:
        row.extend(vector)
    row.append(label)
    writer.writerow(row)

def generate_training_data(base_dir: str, 
                           text_dir: str, arm_dir: str, x86_dir: str, 
                           output_dir: str):
    """根据目录结构生成训练数据"""
    # breakpoint()
    # 定义输出文件路径
    merge_csv_path = os.path.join(output_dir, "merge.csv")
    text_csv_path = os.path.join(output_dir, "text.csv")
    arm_csv_path = os.path.join(output_dir, "arm.csv")
    x86_csv_path = os.path.join(output_dir, "x86.csv")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 删除已存在的文件
    for path in [merge_csv_path, text_csv_path, arm_csv_path, x86_csv_path]:
        if os.path.exists(path):
            os.remove(path)

    # 组织任务结构
    task_to_implementations = {}

    for task_dir in os.listdir(text_dir):
        task_path = os.path.join(text_dir, task_dir)
        if os.path.isdir(task_path):
            task_to_implementations[task_dir] = {
                'text': [],
                'arm': [],
                'x86': []
            }
            for impl_dir in os.listdir(task_path):
                task_to_implementations[task_dir]['text'].append(os.path.join(text_dir, task_dir, impl_dir, "0.txt"))
                task_to_implementations[task_dir]['arm'].append(os.path.join(arm_dir, task_dir, impl_dir, "0.txt"))
                task_to_implementations[task_dir]['x86'].append(os.path.join(x86_dir, task_dir, impl_dir, "0.txt"))

    # 打开CSV写入器
    # breakpoint()
    with open(merge_csv_path, 'w', newline='') as merge_file, \
         open(text_csv_path, 'w', newline='') as text_file, \
         open(arm_csv_path, 'w', newline='') as arm_file, \
         open(x86_csv_path, 'w', newline='') as x86_file:

        merge_writer = csv.writer(merge_file)
        text_writer = csv.writer(text_file)
        arm_writer = csv.writer(arm_file)
        x86_writer = csv.writer(x86_file)

        # 遍历每个任务
        clone_pairs = []
        non_clone_pairs = []

        for task, files in task_to_implementations.items():
            text_files = files['text']
            arm_files = files['arm']
            x86_files = files['x86']

            num_implementations = len(text_files)

            # 相似对 (同一任务的实现之间)
            
            for i in range(num_implementations):
                for j in range(num_implementations):
                    text_vec_left = read_file(text_files[i])
                    # print(type(text_vec_left))
                    # breakpoint()
                    text_vec_right = read_file(text_files[j])
                    arm_vec_left = read_file(arm_files[i])
                    arm_vec_right = read_file(arm_files[j])
                    x86_vec_left = read_file(x86_files[i])
                    x86_vec_right = read_file(x86_files[j])
                    
                    # 保存相似对
                    clone_pairs.append((1, arm_vec_left, x86_vec_left, text_vec_left, 
                                        arm_vec_right, x86_vec_right, text_vec_right))
            # breakpoint()
        tasks = list(task_to_implementations.keys())
        num_tasks = len(tasks)
        selected_tasks = set()  # 已选择过的任务集合
        # 非相似对 (不同任务之间)
        while len(non_clone_pairs) < len(clone_pairs):
            
            while True:
                task_a, task_b = random.sample(tasks, 2)
                # 将任务对转换为元组，并检查是否已经被选择过
                pair = tuple(sorted([task_a, task_b]))
                if pair not in selected_tasks:
                    selected_tasks.add(pair)  # 将新组合添加到已选择集合中
                    break
            
            # task_a, task_b = random.sample(tasks, 2)

            text_files_a = task_to_implementations[task_a]['text']
            arm_files_a = task_to_implementations[task_a]['arm']
            x86_files_a = task_to_implementations[task_a]['x86']

            text_files_b = task_to_implementations[task_b]['text']
            arm_files_b = task_to_implementations[task_b]['arm']
            x86_files_b = task_to_implementations[task_b]['x86']

            # impl_a = random.choice(range(len(text_files_a)))
            # impl_b = random.choice(range(len(text_files_b)))
            # breakpoint()
            # text_vec_left = read_file(text_files_a[impl_a])
            # text_vec_right = read_file(text_files_b[impl_b])
            # arm_vec_left = read_file(arm_files_a[impl_a])
            # arm_vec_right = read_file(arm_files_b[impl_b])
            # x86_vec_left = read_file(x86_files_a[impl_a])
            # x86_vec_right = read_file(x86_files_b[impl_b])
            for impl_a in range(len(text_files_a)):
                for impl_b in range(len(text_files_b)):
                    # breakpoint()
                    text_vec_left = read_file(text_files_a[impl_a])
                    text_vec_right = read_file(text_files_b[impl_b])
                    arm_vec_left = read_file(arm_files_a[impl_a])
                    arm_vec_right = read_file(arm_files_b[impl_b])
                    x86_vec_left = read_file(x86_files_a[impl_a])
                    x86_vec_right = read_file(x86_files_b[impl_b])

                    # 保存非相似对
                    non_clone_pairs.append((0, arm_vec_left, x86_vec_left, text_vec_left, 
                                     arm_vec_right, x86_vec_right, text_vec_right))

        # 写入CSV文件
        for label, arm_left, x86_left, text_left, arm_right, x86_right, text_right in clone_pairs + non_clone_pairs:
            # breakpoint()
            write_to_csv(merge_writer, label, arm_left, x86_left, text_left, arm_right, x86_right, text_right)
            write_to_csv(text_writer, label, text_left, text_right)
            write_to_csv(arm_writer, label, arm_left, arm_right)
            write_to_csv(x86_writer, label, x86_left, x86_right)

if __name__ == "__main__":
    base_dir = "/home/cy/ccd/dataset"
    text_dir = os.path.join(base_dir, "GCJ_test_feature")
    arm_dir = os.path.join(base_dir, "GCJ-ASM/test_arm_syntax_feature")
    x86_dir = os.path.join(base_dir, "GCJ-ASM/test_x86_syntax_feature")
    output_dir = os.path.join(base_dir, "GCJ_test_merge_trainingData")

    generate_training_data(base_dir, text_dir, arm_dir, x86_dir, output_dir)
    print("Training data generated successfully.")
