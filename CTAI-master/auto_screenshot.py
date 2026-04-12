"""
CTAI 重新截图 - 使用真实 DCM 文件
"""
import os, sys, time, requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
BASE_URL = 'http://localhost:8080'
API_URL = 'http://127.0.0.1:5003'

def save(driver, name, desc=""):
    path = os.path.join(SCREENSHOT_DIR, f'{name}.png')
    driver.save_screenshot(path)
    print(f'  [OK] {name}.png - {desc}')

def main():
    print('='*50)
    print('  CTAI 重新截图（使用真实CT图像）')
    print('='*50)

    # 确保有患者
    resp = requests.get(f'{API_URL}/api/patients')
    patients = resp.json().get('data', [])
    if not patients:
        requests.post(f'{API_URL}/api/patients', json={'name': '张三', 'gender': '男', 'age': 55, 'phone': '13800138001'})
        resp = requests.get(f'{API_URL}/api/patients')
        patients = resp.json().get('data', [])

    options = Options()
    options.add_argument('--window-size=1400,900')
    options.add_argument('--force-device-scale-factor=1')
    options.add_argument('--no-sandbox')
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except:
        driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)

    try:
        driver.get(BASE_URL)
        time.sleep(3)

        # 图1: 系统主页面
        print('[图1] 系统主页面...')
        save(driver, 'fig01_system_homepage', '系统主页面')

        # 关闭弹窗
        try:
            close_btn = driver.find_element(By.CSS_SELECTOR, '.el-dialog__headerbtn')
            close_btn.click()
            time.sleep(1)
            try:
                confirm = driver.find_element(By.CSS_SELECTOR, '.el-message-box__btns .el-button--primary')
                confirm.click()
                time.sleep(1)
            except: pass
        except: pass

        # 图2: 添加患者
        print('[图2] 添加患者...')
        try:
            add_btn = driver.find_element(By.XPATH, "//button[contains(.,'添加患者')]")
            add_btn.click()
            time.sleep(1)
            try:
                inputs = driver.find_elements(By.CSS_SELECTOR, '.el-dialog .el-form .el-input__inner')
                if inputs:
                    inputs[0].clear()
                    inputs[0].send_keys('王五')
                if len(inputs) > 1:
                    inputs[-1].clear()
                    inputs[-1].send_keys('13812345678')
            except: pass
            save(driver, 'fig02_add_patient', '添加患者')
            try:
                cancel = driver.find_element(By.XPATH, "//div[contains(@class,'el-dialog')]//button[contains(.,'取消')]")
                cancel.click()
                time.sleep(0.5)
            except: pass
        except Exception as e:
            print(f'  [WARN] {e}')

        # 选择患者
        print('[选择患者]...')
        try:
            sel = driver.find_element(By.CSS_SELECTOR, '.el-select .el-input__inner')
            sel.click()
            time.sleep(1)
            opts = driver.find_elements(By.CSS_SELECTOR, '.el-select-dropdown__item')
            if opts:
                opts[0].click()
                time.sleep(2)
        except: pass

        # ****** 用真实的 10013.dcm 上传 ******
        print('[图5-6] 使用真实 10013.dcm 上传...')
        real_dcm = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CTAI_flask', 'uploads', '10013.dcm'))
        print(f'  DCM 文件: {real_dcm} ({os.path.getsize(real_dcm)//1024} KB)')
        try:
            file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(real_dcm)
            time.sleep(2)
            save(driver, 'fig05_upload_dcm', '上传真实DCM文件')

            # 等待推理完成（模型推理可能需要较长时间）
            print('  等待模型推理...')
            for i in range(30):
                time.sleep(2)
                # 检查是否出现了"预测成功"提示
                try:
                    page = driver.page_source
                    if '预测成功' in page or '推理成功' in page:
                        print(f'  推理完成！(耗时约 {(i+1)*2} 秒)')
                        break
                except: pass
            time.sleep(2)

            # 滚动回顶部看CT图像
            driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)
            save(driver, 'fig06_tumor_result_top', '肿瘤识别结果-顶部（原始CT+标注图）')

            # 滚动到中间看特征值
            driver.execute_script("window.scrollTo(0, 500)")
            time.sleep(1)
            save(driver, 'fig06_tumor_result', '肿瘤识别结果（特征值）')

            # 全页面大图（窗口放大）
            driver.set_window_size(1400, 1600)
            driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)
            save(driver, 'fig06_full_page', '完整识别结果页面')
            driver.set_window_size(1400, 900)
            time.sleep(1)

        except Exception as e:
            print(f'  [WARN] 上传失败: {e}')

        # 图7: 面积对比
        print('[图7] 面积变化趋势...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '面积' in t.text:
                    t.click()
                    time.sleep(2)
                    save(driver, 'fig07_area_trend', '面积变化趋势')
                    break
        except: pass

        # 图8: 周长对比
        print('[图8] 周长变化趋势...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '周长' in t.text:
                    t.click()
                    time.sleep(2)
                    save(driver, 'fig08_perimeter_trend', '周长变化趋势')
                    break
        except: pass

        # 图9: 诊断记录
        print('[图9] 诊断记录...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '诊断记录' in t.text:
                    t.click()
                    time.sleep(2)
                    save(driver, 'fig09_diagnosis_records', '诊断记录')
                    break
        except: pass

        # 图10: 诊断详情
        print('[图10] 诊断详情...')
        try:
            btns = driver.find_elements(By.XPATH, "//button[contains(.,'查看详情')]")
            if btns:
                btns[-1].click()  # 点最新的记录
                time.sleep(2)
                # 滚动到详情区域
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)
                save(driver, 'fig10_diagnosis_detail', '诊断详情+AI模型选择')
        except: pass

        # 图11: 多项指标对比
        print('[图11] 多项指标对比...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '多项指标' in t.text:
                    t.click()
                    time.sleep(2)
                    save(driver, 'fig11_multi_compare', '多项指标对比')
                    break
        except: pass

        # 图12: 历史图像对比
        print('[图12] 历史图像对比...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '历史图像' in t.text:
                    t.click()
                    time.sleep(2)
                    save(driver, 'fig12_history_images', '历史图像对比')
                    break
        except: pass

        # 图13: 时序预测
        print('[图13] 时序预测...')
        try:
            for t in driver.find_elements(By.CSS_SELECTOR, '.el-tabs__item'):
                if '时序预测' in t.text:
                    t.click()
                    time.sleep(3)
                    save(driver, 'fig13_time_series', '时序预测')
                    break
        except: pass

        print()
        print('='*50)
        print(f'  截图完成！目录: {SCREENSHOT_DIR}')
        for f in sorted(os.listdir(SCREENSHOT_DIR)):
            if f.endswith('.png'):
                print(f'  {f} ({os.path.getsize(os.path.join(SCREENSHOT_DIR, f))//1024} KB)')

    finally:
        driver.quit()

if __name__ == '__main__':
    main()
