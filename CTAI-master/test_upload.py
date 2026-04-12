import sys, requests
sys.stdout.reconfigure(encoding='utf-8')

# 直接测试 upload API，带详细信息
resp = requests.get('http://127.0.0.1:5003/api/patients')
patients = resp.json().get('data', [])
pid = patients[0]['id'] if patients else None
print(f'患者ID: {pid}')

# 用不同文件名测试
fname = '10013.dcm'
filepath = f'CTAI_flask/uploads/{fname}'

print(f'File: {filepath}')
import os
print(f'Exists: {os.path.exists(filepath)}')
print(f'Size: {os.path.getsize(filepath)} bytes')

with open(filepath, 'rb') as f:
    content = f.read()
    print(f'Read {len(content)} bytes')

# 用 requests 上传
with open(filepath, 'rb') as f:
    resp = requests.post(
        'http://127.0.0.1:5003/upload',
        files={'file': (fname, f, 'application/octet-stream')},
        data={'patient_id': str(pid)}
    )
    
print(f'Response status code: {resp.status_code}')
print(f'Response body: {resp.text[:500]}')
