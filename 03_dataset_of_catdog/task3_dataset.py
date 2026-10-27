"""Task 3 数据部分：检查图片、划分集合、提取特征、保存 CSV 和统计图。

将本文件放在 Ubuntu 的 03_dataset_of_catdog 项目根目录再运行。
只记录集合标签，不复制或修改原始图片。
"""

# ==================== 板块 1：工具与路径 ====================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib

# 保存图片，不要求弹出桌面窗口；这一行必须放在导入 pyplot 前。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 自动把脚本所在文件夹作为项目目录，迁移时无需修改这里。
project_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(
    project_folder, "data", "kagglecatsanddogs", "PetImages"
)
output_folder = os.path.join(project_folder, "output")
os.makedirs(output_folder, exist_ok=True)

classes = ["Cat", "Dog"]
np.random.seed(42)  # 固定种子，让划分可以复现。
all_records = []
bad_records = []

for class_name in classes:

    # ==================== 板块 2：检查图片 ====================
    print("正在检查：", class_name, flush=True)
    class_folder = os.path.join(data_folder, class_name)
    if not os.path.isdir(class_folder):
        raise FileNotFoundError("找不到类别目录：" + class_folder)

    filenames = sorted(os.listdir(class_folder))
    valid_paths = []

    for filename in filenames:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        image_path = os.path.join(class_folder, filename)
        if not os.path.isfile(image_path):
            continue

        # 默认读取三通道 BGR 图片；无法解码时返回 None。
        image = cv2.imread(image_path)
        if image is None:
            bad_records.append({
                "image_path": os.path.relpath(image_path, project_folder),
                "class_name": class_name.lower(),
                "reason": "unreadable",
            })
            continue
        valid_paths.append(image_path)

    print("可用图片：", len(valid_paths), flush=True)
    if len(valid_paths) < 2600:
        raise ValueError(class_name + " 的可用图片不足 2600 张")

    # ==================== 板块 3：随机划分 ====================
    # 先打乱，再切成不相交的三段，因此同一文件只会分配一次。
    np.random.shuffle(valid_paths)
    split_paths = {
        "train": valid_paths[0:2000],
        "validation": valid_paths[2000:2100],
        "test": valid_paths[2100:2600],
    }

    # ==================== 板块 4：提取信息 ====================
    for split_name in split_paths:
        paths = split_paths[split_name]
        for image_path in paths:
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError("图片读取失败：" + image_path)

            # shape 顺序为：高、宽、通道数。
            height = image.shape[0]
            width = image.shape[1]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            # 统一到 256×256 后计算边缘，固定阈值为 100 和 200。
            small_gray = cv2.resize(gray, (256, 256))
            edges = cv2.Canny(small_gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size

            # 通道 0、1、2 分别是蓝、绿、红；256 个 bin 对应 0～255。
            blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
            red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

            # argmax 给出峰的位置（最常出现的强度），不是出现次数。
            # 多个强度并列最高时，取第一个，即较小的强度。
            record = {
                "image_path": os.path.relpath(image_path, project_folder),
                "class_name": class_name.lower(),
                "split": split_name,
                "width": width,
                "height": height,
                "blue_peak": int(np.argmax(blue_hist)),
                "green_peak": int(np.argmax(green_hist)),
                "red_peak": int(np.argmax(red_hist)),
                "brightness_mean": brightness,
                "edge_density": edge_density,
            }
            all_records.append(record)

        print(class_name, split_name, "处理完成", len(paths), flush=True)

# ==================== 板块 5：检查并保存 CSV ====================
# 从这里回到最左侧：上面的两个类别都处理完，才执行下面的代码。
df = pd.DataFrame(all_records)
counts = df.groupby(["class_name", "split"]).size()
assert len(df) == 5200, "总数必须是 5200"
assert df["image_path"].is_unique, "图片路径不能重复"
for class_name in ["cat", "dog"]:
    assert counts.loc[(class_name, "train")] == 2000
    assert counts.loc[(class_name, "validation")] == 100
    assert counts.loc[(class_name, "test")] == 500

df.to_csv(os.path.join(output_folder, "dataset_info.csv"), index=False)
bad_df = pd.DataFrame(bad_records, columns=["image_path", "class_name", "reason"])
bad_df.to_csv(os.path.join(output_folder, "unreadable_images.csv"), index=False)
print("\n各类别和集合的数量：\n", counts)

# ==================== 板块 6：统计与绘图 ====================
feature_columns = [
    "width", "height", "blue_peak", "green_peak", "red_peak",
    "brightness_mean", "edge_density",
]
summary = df.groupby("class_name")[feature_columns].mean()
summary.to_csv(os.path.join(output_folder, "category_statistics.csv"))

summary[["red_peak", "green_peak", "blue_peak"]].plot(
    kind="bar", color=["red", "green", "blue"], rot=0, figsize=(8, 5)
)
plt.title("Average Color Histogram Peak Position")
plt.xlabel("Class")
plt.ylabel("Pixel Intensity")
plt.ylim(0, 255)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "color_peaks.png"), dpi=150)
plt.close()

summary["brightness_mean"].plot(
    kind="bar", color=["orange", "steelblue"], rot=0, figsize=(6, 4)
)
plt.title("Average Brightness")
plt.xlabel("Class")
plt.ylabel("Grayscale Value")
plt.ylim(0, 255)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "brightness.png"), dpi=150)
plt.close()

summary["edge_density"].plot(
    kind="bar", color=["orange", "steelblue"], rot=0, figsize=(6, 4)
)
plt.title("Average Edge Density")
plt.xlabel("Class")
plt.ylabel("Edge Pixel Ratio")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "edge_density.png"), dpi=150)
plt.close()
print("\n完成，结果位于：", output_folder)
