# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-02-03 05:42:00 (ywatanabe)"
# # File: _download_images.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/web/_download_images.py"
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import requests
# from bs4 import BeautifulSoup
# import os
# import time
# 
# QUERY_VARIABLES = [
#     "エロ画像",
#     "女子大生",
#     "かわいい",
#     "水着",
#     "下着",
#     "セクシー",
#     "ヌード",
#     "おっぱい",
#     "巨乳",
#     "美人",
# ]
# 
# 
# def download_images(url, save_folder, n_images=20):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     images = soup.find_all('img')[:n_images]
# 
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
# 
#     for i, img in enumerate(images):
#         img_url = img.get('src')
#         if img_url:
#             try:
#                 img_data = requests.get(img_url).content
#                 timestamp = int(time.time() * 1000)
#                 with open(f'{save_folder}/image_{timestamp}.jpg', 'wb') as f:
#                     f.write(img_data)
#                 print(f'Downloaded image {i}')
#                 time.sleep(0.1)  # Add small delay to ensure unique timestamps
#             except Exception as e:
#                 print(f'Failed to download image {i}: {e}')
# 
# def main():
#     base_query = input("Enter base search query: ")
#     save_folder = f"/tmp/downloaded_images/{base_query.replace(' ', '+')}"
#     width = 400
#     n_images = 20
# 
#     for query_var in QUERY_VARIABLES:
#         query = f"{base_query} {query_var}"
#         url = f"https://www.google.com/search?q={query}&tbm=isch&tbs=isz:ex,iszw:{width}"
#         print(f"Downloading images for query: {query}")
#         download_images(url, save_folder, n_images)
# 
# # def download_images(url, save_folder):
# #     # ページのHTMLを取得
# #     response = requests.get(url)
# #     soup = BeautifulSoup(response.text, 'html.parser')
# 
# #     # imgタグを全て取得
# #     images = soup.find_all('img')
# 
# #     # 保存用フォルダの作成
# #     if not os.path.exists(save_folder):
# #         os.makedirs(save_folder)
# 
# #     # 画像のダウンロードと保存
# #     for i, img in enumerate(images):
# #         img_url = img.get('src')
# #         if img_url:
# #             try:
# #                 img_data = requests.get(img_url).content
# #                 with open(f'{save_folder}/image_{i}.jpg', 'wb') as f:
# #                     f.write(img_data)
# #                 print(f'画像{i}をダウンロードしました')
# #             except Exception as e:
# #                 print(f'画像{i}のダウンロードに失敗: {e}')
# 
# 
# # def main():
# #     query = input("Enter search query: ")
# #     width = 400
# #     url = f"https://www.google.com/search?q={query}&tbm=isch&tbs=isz:ex,iszw:{width}"
# #     save_folder = f"/tmp/downloaded_images/{query.replace(' ', '+')}"
# #     download_images(url, save_folder)
# 
# 
# if __name__ == "__main__":
#     main()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..web._download_images import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
