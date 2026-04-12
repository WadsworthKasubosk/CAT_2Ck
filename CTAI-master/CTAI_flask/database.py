"""
CTAI 数据库模块
- 使用 SQLite，无需额外安装
- 数据库文件：ctai.db（与 app.py 同目录）
- 包含 patients 表和 diagnosis_records 表
"""
import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ctai.db')


def _safe_serialize(obj):
    """JSON 序列化辅助：处理 numpy 等非标准类型"""
    if hasattr(obj, 'item'):  # numpy scalar (float64, int64 等)
        return obj.item()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 查询结果可按列名访问
    return conn


def init_db():
    """初始化数据库，创建表（如果不存在）"""
    conn = get_db()
    cursor = conn.cursor()

    # 患者信息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            gender     TEXT    NOT NULL,
            age        INTEGER,
            phone      TEXT,
            body_part  TEXT    DEFAULT '直肠',
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 诊断记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   INTEGER NOT NULL,
            dcm_filename TEXT,
            image_url    TEXT,
            draw_url     TEXT,
            mask_url     TEXT,
            area         REAL,
            perimeter    REAL,
            image_info   TEXT,
            status       TEXT    DEFAULT 'completed',
            doctor_note  TEXT,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')

    # 兼容已有数据库：如果表已存在但缺少新字段，用 ALTER TABLE 补上
    for col, col_def in [('mask_url', 'TEXT'), ('status', "TEXT DEFAULT 'completed'")]:
        try:
            cursor.execute(f'ALTER TABLE diagnosis_records ADD COLUMN {col} {col_def}')
        except Exception:
            pass  # 列已存在，忽略

    # LLM 辅助报告表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis_id  INTEGER NOT NULL,
            prompt        TEXT,
            advice        TEXT,
            model_name    TEXT,
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (diagnosis_id) REFERENCES diagnosis_records(id)
        )
    ''')

    conn.commit()
    conn.close()


# ============ 患者 CRUD ============

def add_patient(name, gender, age=None, phone=None, body_part='直肠'):
    """新增患者，返回新患者的 id"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO patients (name, gender, age, phone, body_part) VALUES (?, ?, ?, ?, ?)',
        (name, gender, age, phone, body_part)
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_patients():
    """查询所有患者，按创建时间倒序"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_patient(patient_id):
    """查询单个患者，返回 dict 或 None"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_patient(patient_id):
    """删除患者，返回是否成功"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============ 诊断记录 CRUD ============

def add_diagnosis_record(patient_id, dcm_filename=None, image_url=None,
                         draw_url=None, mask_url=None, area=None, perimeter=None,
                         image_info=None, status='completed', doctor_note=None):
    """新增诊断记录，返回新记录的 id"""
    # image_info 如果是 dict，序列化为 JSON 字符串
    # default=_safe_serialize 处理 numpy 等非标准类型
    if isinstance(image_info, dict):
        image_info = json.dumps(image_info, ensure_ascii=False, default=_safe_serialize)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO diagnosis_records
           (patient_id, dcm_filename, image_url, draw_url, mask_url, area, perimeter, image_info, status, doctor_note)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (patient_id, dcm_filename, image_url, draw_url, mask_url, area, perimeter, image_info, status, doctor_note)
    )
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_diagnosis_records_by_patient(patient_id):
    """查询某患者的所有诊断记录，按时间倒序"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM diagnosis_records WHERE patient_id = ? ORDER BY created_at DESC',
        (patient_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        # 将 image_info 从 JSON 字符串还原为 dict
        if d.get('image_info'):
            try:
                d['image_info'] = json.loads(d['image_info'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def get_diagnosis_record(record_id):
    """查询单条诊断记录，返回 dict 或 None"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM diagnosis_records WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if d.get('image_info'):
        try:
            d['image_info'] = json.loads(d['image_info'])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


# ============ LLM 报告 CRUD ============

def add_llm_report(diagnosis_id, prompt, advice, model_name):
    """保存 LLM 生成的报告，返回 report id"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO llm_reports (diagnosis_id, prompt, advice, model_name)
           VALUES (?, ?, ?, ?)''',
        (diagnosis_id, prompt, advice, model_name)
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id


def get_llm_reports_by_diagnosis(diagnosis_id):
    """查询某次诊断的所有 LLM 报告"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM llm_reports WHERE diagnosis_id = ? ORDER BY created_at DESC',
        (diagnosis_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_llm_report(report_id):
    """查询单条 LLM 报告"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM llm_reports WHERE id = ?', (report_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
