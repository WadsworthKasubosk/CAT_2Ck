# -*- coding: utf-8 -*-
"""用 playwright 为系统各主要页面截图, 对应论文 5.7 节 图5-5..5-10"""
import os, asyncio
from playwright.async_api import async_playwright

OUT = os.path.join(os.path.dirname(__file__), '论文插图_png')
os.makedirs(OUT, exist_ok=True)
BASE = 'http://127.0.0.1:5003'
PID = 8  # 王山, 已有 2 条记录

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await ctx.new_page()

        # ---- 图5-5 系统主界面 (患者列表首屏) ----
        await page.goto(BASE + '/', wait_until='networkidle', timeout=25000)
        await page.wait_for_timeout(3500)
        await page.screenshot(path=os.path.join(OUT, '图5-5_系统主界面.png'), full_page=False)
        print('saved 图5-5')

        # ---- 图5-10 患者管理界面 (全页) ----
        await page.screenshot(path=os.path.join(OUT, '图5-10_患者管理界面.png'), full_page=True)
        print('saved 图5-10')

        # ---- 图5-6/5-7 诊断页面 (点到历史图像 tab) ----
        await page.goto(BASE + f'/#/patient/{PID}', wait_until='networkidle', timeout=25000)
        await page.wait_for_timeout(4000)
        # 切换到 "历史图像对比" tab
        try:
            tab = page.get_by_text('历史图像对比', exact=False).first
            await tab.click()
            await page.wait_for_timeout(2500)
        except Exception as e:
            print('tab click err:', e)

        await ctx.new_page()  # no-op
        await page.set_viewport_size({'width': 1440, 'height': 1600})
        await page.wait_for_timeout(1000)
        await page.screenshot(path=os.path.join(OUT, '图5-6_CT上传与分割结果.png'), full_page=True)
        print('saved 图5-6')

        # ---- 图5-7 特征值表格 (先切到 "肿瘤区域特征值" tab) ----
        await page.set_viewport_size({'width': 1440, 'height': 900})
        try:
            tab = page.get_by_text('肿瘤区域特征值', exact=False).first
            await tab.click()
            await page.wait_for_timeout(1500)
        except Exception as e:
            print('feature tab err:', e)
        # 点第一条诊断记录以展示特征
        try:
            tab2 = page.get_by_text('诊断记录', exact=False).first
            await tab2.click()
            await page.wait_for_timeout(1500)
        except Exception as e:
            print('record tab err:', e)
        await page.set_viewport_size({'width': 1440, 'height': 1400})
        await page.wait_for_timeout(800)
        await page.screenshot(path=os.path.join(OUT, '图5-7_影像组学特征值表格.png'), full_page=True)
        print('saved 图5-7')

        # ---- 图5-8 趋势分析图表 ----
        await page.set_viewport_size({'width': 1440, 'height': 900})
        await page.goto(BASE + f'/#/patient/{PID}/trend', wait_until='networkidle', timeout=25000)
        await page.wait_for_timeout(4500)
        await page.screenshot(path=os.path.join(OUT, '图5-8_趋势分析图表.png'), full_page=False)
        print('saved 图5-8')

        # ---- 图5-9 LLM 辅助建议 (点"历史图像"/多指标tab先不切; 主要看AI辅助分析面板) ----
        # 这张图直接截趋势页底部 AI面板; 无 Key 也能看到 UI 形态
        await page.set_viewport_size({'width': 1440, 'height': 1400})
        await page.wait_for_timeout(800)
        await page.screenshot(path=os.path.join(OUT, '图5-9_LLM辅助建议.png'), full_page=True)
        print('saved 图5-9')

        await browser.close()

asyncio.run(run())
print('DONE')
