# -*- coding: utf-8 -*-
"""将论文插图目录下的 HTML 文件批量截图为 PNG"""

import os
import glob
from playwright.sync_api import sync_playwright

HTML_DIR = os.path.join(os.path.dirname(__file__), '论文插图')
OUT_DIR = os.path.join(os.path.dirname(__file__), '论文插图_png')
os.makedirs(OUT_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1200, 'height': 800})

    for html_file in sorted(glob.glob(os.path.join(HTML_DIR, '*.html'))):
        name = os.path.splitext(os.path.basename(html_file))[0]
        out_path = os.path.join(OUT_DIR, f'{name}.png')

        file_url = 'file:///' + os.path.abspath(html_file).replace('\\', '/')
        page.goto(file_url)
        page.wait_for_timeout(1500)  # 等渲染

        # 截取内容区域
        page.screenshot(path=out_path, full_page=True)
        print(f'OK: {name}.png')

    browser.close()

print(f'\n全部完成，输出目录: {OUT_DIR}')
