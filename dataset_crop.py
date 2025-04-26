import os
import random
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

def parse_class_labels(file_path):
    class_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Удаляем пробелы и символы переноса строки
            line = line.strip()
            if line:  # Пропускаем пустые строки
                # Разделяем строку по ": "
                type_id, class_name = line.split(':', 1)
                # Добавляем в словарь, преобразуя type_id в int
                class_dict[int(type_id)] = class_name
    return class_dict

# Использование
file_path = 'images/xview_class_labels.txt'
id_to_name = parse_class_labels(file_path)
print(id_to_name)

neg_size = (128, 128)  # (width, height)
n_neg_per_img = 3


# === Параметры ===
geojson_path = "images/xView_train.geojson"
train_dir = "images/train_images"
val_dir = "images/val_images"
out_base = "images/crops"

classes = [
    "Small Car", "Building", "Excavator", "Damaged Building",
    "Shipping Container Lot", 'Small Aircraft', "Tower",
    "Tugboat", "Ferry", "Aircraft Hangar", "Barge"
]

# === Создаём папки ===
for split in ("train", "val"):
    for cls in classes + ["negative"]:
        os.makedirs(f"{out_base}/{split}/{cls}", exist_ok=True)

# === Загрузка GeoJSON ===
gdf = gpd.read_file(geojson_path)


def process_split(split_dir, split_name):
    # Фильтруем аннотации по тому, есть ли файл в данной папке
    imgs = set(os.listdir(split_dir))
    split_gdf = gdf[gdf["image_id"].isin(imgs)]
    # Группируем аннотации по image_id
    grouped = split_gdf.groupby("image_id")
    print(grouped)
    for img_id, group in tqdm(grouped, desc=f"Processing {split_name}"):
        img_path = os.path.join(split_dir, img_id)
        with rasterio.open(img_path) as src:
            full_w, full_h = src.width, src.height

            # создаём объединённую геометрию «занятых» областей
            occupied = unary_union([row.geometry for _, row in group.iterrows()])

            # 1) Вырезаем каждую «позитивную» рамку по классам
            for idx, row in group.iterrows():
                if row["type_id"] not in id_to_name.keys():
                    continue
                if id_to_name[row["type_id"]] not in classes:
                    continue
                cls = id_to_name[row["type_id"]]

                xmin, ymin, xmax, ymax = map(int, row.bounds_imcoords.split(','))
                w, h = xmax - xmin, ymax - ymin
                if w <= 0 or h <= 0:
                    print('I AM HERE')
                    continue
                window = Window(xmin, ymin, w, h)
                patch = src.read(window=window)
                patch = np.transpose(patch, (1, 2, 0))
                try:
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                except:
                    print('FAIL')
                    continue
                save_p = os.path.join(out_base, split_name, cls, f"{img_id}_{idx}.png")
                cv2.imwrite(save_p, patch)

            # 2) Генерируем негативы
            neg_count = 0
            attempts = 0
            while neg_count < n_neg_per_img and attempts < n_neg_per_img * 10:
                attempts += 1
                x0 = random.randint(0, full_w - neg_size[0])
                y0 = random.randint(0, full_h - neg_size[1])
                candidate = box(x0, y0, x0 + neg_size[0], y0 + neg_size[1])
                # проверяем IoU < 0.1
                inter = occupied.intersection(candidate).area
                iou = inter / candidate.area
                if iou < 0.1:
                    window = Window(x0, y0, neg_size[0], neg_size[1])
                    patch = src.read(window=window)
                    patch = np.transpose(patch, (1, 2, 0))
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    save_p = os.path.join(out_base, split_name, "negative", f"{img_id}_neg{neg_count}.png")
                    cv2.imwrite(save_p, patch)
                    neg_count += 1


if __name__ == "__main__":
    process_split(train_dir, "train")

