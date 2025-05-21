
# -*- coding: utf-8 -*-
import shutil
import json
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
import fiona
import os
import configparser
from pyproj import Transformer
from shapely.ops import transform as shapely_transform
import time
import tempfile
import csv

def guess_land_type_key(properties):
    for key in properties.keys():
        if "地目" in key or "用途" in key or "種別" in key:
            return key
    return None

def safe_path(path):
    abspath = os.path.abspath(path)
    return abspath

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

tif_path = config["INPUT"]["NDVI_DIFF_TIF"]
gpkg_path = config["INPUT"]["GPKG"]
layer_name = config["INPUT"]["LAYER_NAME"]
bound_geojson = config["INPUT"]["BOUND_GEOJSON"]
output_dir = config["OUTPUT"]["PATCH_DIR"]
src_epsg = int(config["CRS"]["SOURCE_EPSG"])
dst_epsg = int(config["CRS"]["TARGET_EPSG"])

if os.path.exists(output_dir):
    print("既存のパッチディレクトリを削除しています...")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

with open(bound_geojson, encoding="utf-8") as f:
    bound_data = json.load(f)

bound_geom = [shape(feature["geometry"]) for feature in bound_data["features"]]
bound_union = bound_geom[0]
for geom in bound_geom[1:]:
    bound_union = bound_union.union(geom)

transformer_to_src = Transformer.from_crs(dst_epsg, src_epsg, always_xy=True)
bound_in_src = shapely_transform(lambda x, y: transformer_to_src.transform(x, y), bound_union)

print("地目分類中...")
with fiona.open(gpkg_path, layer=layer_name, encoding="cp932") as src_vec:
    first_feature = next(iter(src_vec), None)
    if not first_feature:
        print("GPKGにフィーチャが存在しません。")
        exit()
    land_key = guess_land_type_key(first_feature["properties"])
    if not land_key:
        print("地目に該当する属性が見つかりませんでした。")
        exit()

    for feature in src_vec:
        land_type = feature["properties"].get(land_key)
        if land_type:
            clean_land_type = str(land_type).strip().replace("　", "").replace("\n", "").replace("\r", "")
            for c in r'<>:"/\|?*':
                clean_land_type = clean_land_type.replace(c, '')
            if clean_land_type:
                os.makedirs(safe_path(os.path.join(output_dir, clean_land_type)), exist_ok=True)

start_time = time.time()
processed_count = 0
failed_patches = []

print("各筆を分類中...")

with fiona.open(gpkg_path, layer=layer_name, encoding="cp932") as src_vec:
    first_feature = next(iter(src_vec), None)
    land_key = guess_land_type_key(first_feature["properties"]) if first_feature else None
    if not land_key:
        print("再オープン時に地目キーが取得できませんでした。")
        exit()

    with rasterio.open(tif_path) as src_ras:
        for idx, feature in enumerate(src_vec):
            try:
                geom = shape(feature["geometry"])
                if not geom.intersects(bound_in_src):
                    continue
                geom_clip = geom.intersection(bound_in_src)
                if geom_clip.is_empty:
                    continue

                gid = feature["properties"].get("gid", idx)
                land_type = feature["properties"].get(land_key)
                if not land_type:
                    failed_patches.append({"gid": gid, "land_type": None, "reason": "land_type_none"})
                    continue

                clean_land_type = str(land_type).strip().replace("　", "").replace("\n", "").replace("\r", "")
                for c in r'<>:"/\|?*':
                    clean_land_type = clean_land_type.replace(c, '')
                if not clean_land_type:
                    failed_patches.append({"gid": gid, "land_type": land_type, "reason": "invalid_land_type"})
                    continue

                label_dir = os.path.join(output_dir, clean_land_type)
                os.makedirs(safe_path(label_dir), exist_ok=True)

                geom_json = [mapping(geom_clip)]
                out_image, out_transform = mask(src_ras, geom_json, crop=True)
                patch = out_image[0]
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    failed_patches.append({"gid": gid, "land_type": clean_land_type, "reason": "empty_patch"})
                    continue

                profile = src_ras.profile.copy()
                profile.update({
                    "height": patch.shape[0],
                    "width": patch.shape[1],
                    "transform": out_transform,
                    "count": 1
                })

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_out_path = os.path.join(tmp_dir, f"patch_{idx:04d}.tif")
                    with rasterio.open(tmp_out_path, "w", **profile) as dst:
                        dst.write(patch, 1)
                    final_out_path = os.path.join(label_dir, f"patch_{idx:04d}.tif")
                    shutil.move(tmp_out_path, final_out_path)

                processed_count += 1

            except Exception as e:
                failed_patches.append({
                    "gid": feature["properties"].get("gid", idx),
                    "land_type": feature["properties"].get(land_key),
                    "reason": f"exception: {e}"
                })

end_time = time.time()
elapsed = end_time - start_time
print(f"処理完了: 成功数={processed_count}, 処理時間={elapsed:.2f}秒")

with open("unclassified_features.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gid", "land_type", "reason"])
    writer.writeheader()
    writer.writerows(failed_patches)
