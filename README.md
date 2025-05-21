# nara-land-classifier
奈良市のオープンデータを用いた地目分類AIの教師データ作成プロジェクト
# 奈良市のオープンデータを使用して 自動地目分類器の教師データを作成

## 概要

本リポジトリは、奈良市が提供するオープンデータを利用して、地目分類AIの教師データを作成・学習するためのプロジェクトです。  
従来の [auto-geomoku-classifier](https://github.com/NaohikoMuramatsu2025/auto-geomoku-classifier) で使用されている `label_map.json` や `land_classifier_model.pth` とは別地域のデータを対象としています。奈良市データに合わせて適宜差し替えてご使用ください。

## 参考リンク

- 奈良市オープンデータ（筆界・地目等）:  
  [https://www.city.nara.lg.jp/soshiki/14/104605.html](https://www.city.nara.lg.jp/soshiki/14/104605.html)

## ディレクトリ構成
```text
├─Input # 入力データ（地図画像、GeoJSON等）
├─Output # 出力データ（AI分類結果など）
├─patches # 地目ごとに分類された小画像（パッチ画像）
│ ├─宅地
│ ├─公衆道路
│ ├─原野
│ ├─学校用地
│ └─… 他
├─start_nara.bat # 教師データ作成バッチ
├─start_train_ADD.bat # モデル学習開始バッチ
├─create_patches_count_fixed_CSV_ONLY.py # ステップ1用スクリプト
├─train_land_classifier_ADD.py # ステップ2用スクリプト
└─nara.qgz # QGIS プロジェクトファイル
```

## ステップ1：小さな画像をたくさん作る

`create_patches_count_fixed_CSV_ONLY.py` を実行して、地図や航空写真から地目別のパッチ画像を大量に生成します。

- **目的**：AIに学習させるための例題（画像＋地目ラベル）を準備する。  
- **出力先**：`patches/` ディレクトリ内に地目ごとのフォルダが作成され、画像が保存されます。

📝 この工程は、AIに「教科書」を渡す作業です。

## ステップ2：AIに土地の種類を学習させる

`train_land_classifier_ADD.py` を使用して、パッチ画像とラベル情報からAIモデルを学習させます。

- **使用技術**：PyTorch（深層学習）  
- **入力**：パッチ画像、ラベルCSV  
- **出力**：学習済みモデル（例：`land_classifier_model_nara.pth`）

🧠 AIが地目分類のパターンを自動的に学習します。
