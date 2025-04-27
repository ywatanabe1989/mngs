# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/web/_summarize_url.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-29 21:43:30 (ywatanabe)"
# # ./src/mngs/web/_crawl.py
# 
# 
# import requests
# from bs4 import BeautifulSoup
# import urllib.parse
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import json
# from tqdm import tqdm
# import mngs
# from pprint import pprint
# from readability import Document
# 
# import re
# 
# 
# # def crawl_url(url, max_depth=1):
# #     print("\nCrawling...")
# #     visited = set()
# #     to_visit = [(url, 0)]
# #     contents = {}
# 
# #     while to_visit:
# #         current_url, depth = to_visit.pop(0)
# #         if current_url in visited or depth > max_depth:
# #             continue
# 
# #         try:
# #             response = requests.get(current_url)
# #             if response.status_code == 200:
# #                 visited.add(current_url)
# #                 contents[current_url] = response.text
# #                 soup = BeautifulSoup(response.text, "html.parser")
# 
# #                 for link in soup.find_all("a", href=True):
# #                     absolute_link = urllib.parse.urljoin(
# #                         current_url, link["href"]
# #                     )
# #                     if absolute_link not in visited:
# #                         to_visit.append((absolute_link, depth + 1))
# 
# #         except requests.RequestException:
# #             pass
# 
# #     return visited, contents
# 
# 
# def extract_main_content(html):
#     doc = Document(html)
#     content = doc.summary()
#     # Remove HTML tags
#     content = re.sub("<[^<]+?>", "", content)
#     # Remove extra whitespace
#     content = " ".join(content.split())
#     return content
# 
# 
# def crawl_url(url, max_depth=1):
#     print("\nCrawling...")
#     visited = set()
#     to_visit = [(url, 0)]
#     contents = {}
# 
#     while to_visit:
#         current_url, depth = to_visit.pop(0)
#         if current_url in visited or depth > max_depth:
#             continue
# 
#         try:
#             response = requests.get(current_url)
#             if response.status_code == 200:
#                 visited.add(current_url)
#                 main_content = extract_main_content(response.text)
#                 contents[current_url] = main_content
#                 soup = BeautifulSoup(response.text, "html.parser")
# 
#                 for link in soup.find_all("a", href=True):
#                     absolute_link = urllib.parse.urljoin(
#                         current_url, link["href"]
#                     )
#                     if absolute_link not in visited:
#                         to_visit.append((absolute_link, depth + 1))
# 
#         except requests.RequestException:
#             pass
# 
#     return visited, contents
# 
# 
# def crawl_to_json(start_url):
#     if not start_url.startswith("http"):
#         start_url = "https://" + start_url
#     crawled_urls, contents = crawl_url(start_url)
# 
#     print("\nSummalizing as json...")
# 
#     def process_url(url):
#         llm = mngs.ai.GenAI("gpt-4o-mini")
#         return {
#             "url": url,
#             "content": llm(
#                 f"Summarize this page in 1 line:\n\n{contents[url]}"
#             ),
#         }
# 
#     with ThreadPoolExecutor() as executor:
#         future_to_url = {
#             executor.submit(process_url, url): url for url in crawled_urls
#         }
#         crawled_pages = []
#         for future in tqdm(
#             as_completed(future_to_url),
#             total=len(crawled_urls),
#             desc="Processing URLs",
#         ):
#             crawled_pages.append(future.result())
# 
#     result = {"start_url": start_url, "crawled_pages": crawled_pages}
# 
#     return json.dumps(result, indent=2)
# 
# 
# def summarize_all(json_contents):
#     llm = mngs.ai.GenAI("gpt-4o-mini")
#     out = llm(
#         f"Summarize this json file with 5 bullet points:\n\n{json_contents}"
#     )
#     return out
# 
# 
# def summarize_url(start_url):
#     json_result = crawl_to_json(start_url)
#     ground_summary = summarize_all(json_result)
# 
#     pprint(ground_summary)
#     return ground_summary, json_result
# 
# 
# main = summarize_url
# 
# if __name__ == "__main__":
#     import argparse
#     import mngs
# 
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument("--url", "-u", type=str, help="(default: %(default)s)")
#     args = parser.parse_args()
#     mngs.gen.print_block(args, c="yellow")
# 
#     main(args.url)

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.web._summarize_url import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
