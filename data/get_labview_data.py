# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:28:17 2025

@author: 90829
"""

import requests
import os

# ========= 填你的个人信息 ==========
TOKEN = os.environ["GITHUB_TOKEN"]  # 从环境变量读取 GitHub personal access token
OWNER = "BE-HVACR"
REPO = "TAM-SCHT"
BRANCH = "main"
TARGET_PATH = "Data/Testing"  # 只递归这个目录

target_house = 'West'
year = 2025
# ==================================

headers = {"Authorization": f"token {TOKEN}"}

file_name = f"{target_house}_{year}_csv"
os.makedirs(f"{file_name}", exist_ok=True)

def fetch_files(path):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}"
    r = requests.get(url, headers=headers)
    items = r.json()

    if isinstance(items, dict) and items.get("message"):
        print("Error:", items)
        return

    for item in items:
        if item["type"] == "dir":
            fetch_files(item["path"])
        elif item["type"] == "file":
            name = item["name"]
            if target_house in name and year in name and name.endswith(".csv"):
                print("Downloading:", name)
                file_data = requests.get(item["download_url"], headers=headers).content
                with open(f"{file_name}/{name}", "wb") as f:
                    f.write(file_data)

print("Searching target files...")
fetch_files(TARGET_PATH)
print(f"Done! Files saved to ./{file_name}/")
