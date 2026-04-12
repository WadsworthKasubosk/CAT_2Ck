import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta

import torch
from flask import *

import core.main
import database
import llm_service
import core.net.unet as net
import core.time_series_predict as ts_predict

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['dcm'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


# Vue Router history 模式 fallback：前端路由交由 index.html 处理
@app.route('/patient/<path:subpath>')
def vue_router_fallback(subpath):
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        file_basename = file.filename.rsplit('.', 1)[0]  # 去掉 .dcm 后缀

        patient_id = request.form.get('patient_id')

        # 尝试模型推理
        try:
            pid, image_info = core.main.c_main(image_path, current_app.model)

            image_url = 'http://127.0.0.1:5003/tmp/image/' + pid
            draw_url = 'http://127.0.0.1:5003/tmp/draw/' + pid
            mask_url = 'http://127.0.0.1:5003/tmp/mask/' + file_basename + '_mask.png'

            result = {'status': 1,
                      'image_url': image_url,
                      'draw_url': draw_url,
                      'image_info': image_info}

            # 如果前端传了 patient_id，将诊断结果保存到数据库
            if patient_id:
                try:
                    patient_id = int(patient_id)
                    area_val = None
                    perimeter_val = None
                    if isinstance(image_info, dict):
                        area_item = image_info.get('area')
                        if isinstance(area_item, list) and len(area_item) > 1:
                            area_val = float(area_item[1])
                        perimeter_item = image_info.get('perimeter')
                        if isinstance(perimeter_item, list) and len(perimeter_item) > 1:
                            perimeter_val = float(perimeter_item[1])

                    record_id = database.add_diagnosis_record(
                        patient_id=patient_id,
                        dcm_filename=file.filename,
                        image_url=image_url,
                        draw_url=draw_url,
                        mask_url=mask_url,
                        area=area_val,
                        perimeter=perimeter_val,
                        image_info=image_info,
                        status='completed'
                    )
                    result['record_id'] = record_id
                except Exception as e:
                    result['db_error'] = str(e)

            return jsonify(result)

        except Exception as inference_err:
            # 推理失败：如果有 patient_id，仍然创建一条 failed 记录
            if patient_id:
                try:
                    record_id = database.add_diagnosis_record(
                        patient_id=int(patient_id),
                        dcm_filename=file.filename,
                        status='failed',
                        doctor_note=f'推理失败: {str(inference_err)}'
                    )
                    return jsonify({'status': 0,
                                    'msg': f'上传成功，但模型推理失败: {str(inference_err)}',
                                    'record_id': record_id,
                                    'inference_status': 'failed'})
                except Exception:
                    pass
            return jsonify({'status': 0,
                            'msg': f'上传成功，但模型推理失败: {str(inference_err)}',
                            'inference_status': 'failed'})

    return jsonify({'status': 0})


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        # URL 解码，处理中文文件名
        import urllib.parse
        file_decoded = urllib.parse.unquote(file)
        file_path = os.path.join('tmp', file_decoded)
        if not os.path.exists(file_path):
            return jsonify({'status': 0, 'msg': 'file not found'}), 404
        with open(file_path, 'rb') as f:
            image_data = f.read()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        pass


# ============ 患者管理接口 ============

@app.route('/api/patients', methods=['POST'])
def api_add_patient():
    data = request.get_json()
    if not data or not data.get('name') or not data.get('gender'):
        return jsonify({'status': 0, 'msg': '姓名和性别为必填项'}), 400

    patient_id = database.add_patient(
        name=data['name'],
        gender=data['gender'],
        age=data.get('age'),
        phone=data.get('phone'),
        body_part=data.get('body_part', '直肠')
    )
    patient = database.get_patient(patient_id)
    return jsonify({'status': 1, 'data': patient})


@app.route('/api/patients', methods=['GET'])
def api_list_patients():
    patients = database.get_patients()
    return jsonify({'status': 1, 'data': patients})


@app.route('/api/patients/<int:patient_id>', methods=['GET'])
def api_get_patient(patient_id):
    patient = database.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404
    return jsonify({'status': 1, 'data': patient})


@app.route('/api/patients/<int:patient_id>', methods=['DELETE'])
def api_delete_patient(patient_id):
    success = database.delete_patient(patient_id)
    if not success:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404
    return jsonify({'status': 1, 'msg': '删除成功'})


# ============ 诊断记录接口 ============

@app.route('/api/diagnosis', methods=['POST'])
def api_add_diagnosis():
    data = request.get_json()
    if not data or not data.get('patient_id'):
        return jsonify({'status': 0, 'msg': 'patient_id 为必填项'}), 400

    # 校验患者是否存在
    patient = database.get_patient(data['patient_id'])
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404

    record_id = database.add_diagnosis_record(
        patient_id=data['patient_id'],
        dcm_filename=data.get('dcm_filename'),
        image_url=data.get('image_url'),
        draw_url=data.get('draw_url'),
        mask_url=data.get('mask_url'),
        area=data.get('area'),
        perimeter=data.get('perimeter'),
        image_info=data.get('image_info'),
        status=data.get('status', 'completed'),
        doctor_note=data.get('doctor_note')
    )
    record = database.get_diagnosis_record(record_id)
    return jsonify({'status': 1, 'data': record})


@app.route('/api/patients/<int:patient_id>/records', methods=['GET'])
def api_list_diagnosis_records(patient_id):
    # 校验患者是否存在
    patient = database.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404

    records = database.get_diagnosis_records_by_patient(patient_id)
    return jsonify({'status': 1, 'data': records})


@app.route('/api/diagnosis/<int:record_id>', methods=['GET'])
def api_get_diagnosis(record_id):
    record = database.get_diagnosis_record(record_id)
    if not record:
        return jsonify({'status': 0, 'msg': '诊断记录不存在'}), 404
    return jsonify({'status': 1, 'data': record})


# ============ 特征值查询接口 ============

@app.route('/api/diagnosis/<int:record_id>/features', methods=['GET'])
def api_get_features(record_id):
    """查询某次诊断的完整特征值（从 image_info JSON 中提取）"""
    record = database.get_diagnosis_record(record_id)
    if not record:
        return jsonify({'status': 0, 'msg': '诊断记录不存在'}), 404

    image_info = record.get('image_info')
    if not image_info or not isinstance(image_info, dict):
        return jsonify({'status': 0, 'msg': '该诊断记录无特征数据'}), 404

    # 将 image_info 格式化为更直观的结构
    # 原格式: {"area": ["面积", 1086.0], ...}
    # 输出:   {"area": {"name": "面积", "value": 1086.0}, ...}
    features = {}
    for key, val in image_info.items():
        if isinstance(val, list) and len(val) >= 2:
            features[key] = {'name': val[0], 'value': val[1]}
        else:
            features[key] = {'name': key, 'value': val}

    return jsonify({
        'status': 1,
        'data': {
            'record_id': record['id'],
            'patient_id': record['patient_id'],
            'created_at': record['created_at'],
            'area': record.get('area'),
            'perimeter': record.get('perimeter'),
            'features': features
        }
    })


@app.route('/api/patients/<int:patient_id>/features', methods=['GET'])
def api_get_patient_features(patient_id):
    """查询某患者所有诊断的关键特征汇总（用于趋势分析）"""
    patient = database.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404

    records = database.get_diagnosis_records_by_patient(patient_id)

    summary = []
    for r in records:
        item = {
            'record_id': r['id'],
            'created_at': r['created_at'],
            'status': r.get('status', 'completed'),
            'area': r.get('area'),
            'perimeter': r.get('perimeter'),
        }
        # 从 image_info 中提取灰度均值等关键指标
        info = r.get('image_info')
        if isinstance(info, dict):
            for key in ['mean', 'std', 'focus_x', 'focus_y', 'ellipse']:
                val = info.get(key)
                if isinstance(val, list) and len(val) >= 2:
                    item[key] = val[1]
        summary.append(item)

    return jsonify({'status': 1, 'data': summary})


# ============ 趋势分析接口 ============

def _judge_trend(values):
    """简单趋势判断：比较首末值，变化超过 5% 判为增大/减小，否则稳定"""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return '数据不足'
    first, last = valid[0], valid[-1]
    if first == 0:
        return '增大' if last > 0 else '稳定'
    change_rate = (last - first) / abs(first)
    if change_rate > 0.05:
        return '增大'
    elif change_rate < -0.05:
        return '减小'
    else:
        return '稳定'


@app.route('/api/patients/<int:patient_id>/trend', methods=['GET'])
def api_get_trend(patient_id):
    """
    趋势分析接口：返回某患者各诊断指标的时间序列，可直接用于 ECharts 绑定。
    返回格式兼容前端 ECharts 的 xAxis.data + series[].data 格式。
    """
    patient = database.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404

    # 取所有已完成的诊断记录，按时间正序（画图需要从早到晚）
    records = database.get_diagnosis_records_by_patient(patient_id)
    completed = [r for r in records if r.get('status', 'completed') == 'completed']
    completed.reverse()  # 数据库返回倒序，反转为正序

    if not completed:
        return jsonify({'status': 1, 'data': {
            'count': 0,
            'dates': [],
            'area': {'values': [], 'trend': '数据不足'},
            'perimeter': {'values': [], 'trend': '数据不足'},
        }})

    # 提取时间序列和各指标序列
    dates = []
    area_values = []
    perimeter_values = []
    mean_values = []
    std_values = []
    ellipse_values = []

    for r in completed:
        # 日期格式化：取 "2026-03-13 02:30:00" 中的 "03-13"，图表更简洁
        date_str = r.get('created_at', '')
        if len(date_str) >= 10:
            dates.append(date_str[5:10])  # "03-13"
        else:
            dates.append(date_str)

        area_values.append(r.get('area'))
        perimeter_values.append(r.get('perimeter'))

        # 从 image_info 中提取可选指标
        info = r.get('image_info')
        if isinstance(info, dict):
            mean_item = info.get('mean')
            mean_values.append(mean_item[1] if isinstance(mean_item, list) and len(mean_item) > 1 else None)
            std_item = info.get('std')
            std_values.append(std_item[1] if isinstance(std_item, list) and len(std_item) > 1 else None)
            ellipse_item = info.get('ellipse')
            ellipse_values.append(ellipse_item[1] if isinstance(ellipse_item, list) and len(ellipse_item) > 1 else None)
        else:
            mean_values.append(None)
            std_values.append(None)
            ellipse_values.append(None)
    # ---- 新增：基于真实面积变化计算每条记录的病情状态 ----
    # 规则：对比当前面积与上一次面积的变化率
    #   缩小 > 5%  → 好转
    #   增大 > 5%  → 恶化
    #   变化 <= 5% → 稳定
    #   第一条记录 → 初诊
    #   面积为 None 或 0 → 待评估
    record_statuses = []
    for i, r in enumerate(completed):
        cur_area = r.get('area')
        if i == 0:
            record_statuses.append({
                'record_id': r.get('id'),
                'date': r.get('created_at', ''),
                'status': '初诊',
                'status_type': 'info',
                'detail': '首次诊断记录'
            })
        elif cur_area is None or cur_area == 0:
            record_statuses.append({
                'record_id': r.get('id'),
                'date': r.get('created_at', ''),
                'status': '待评估',
                'status_type': 'warning',
                'detail': '面积数据缺失，无法判断'
            })
        else:
            prev_area = completed[i - 1].get('area')
            if prev_area is None or prev_area == 0:
                record_statuses.append({
                    'record_id': r.get('id'),
                    'date': r.get('created_at', ''),
                    'status': '待评估',
                    'status_type': 'warning',
                    'detail': '上一次面积数据缺失'
                })
            else:
                change_rate = (cur_area - prev_area) / abs(prev_area)
                if change_rate < -0.05:
                    record_statuses.append({
                        'record_id': r.get('id'),
                        'date': r.get('created_at', ''),
                        'status': '好转',
                        'status_type': 'success',
                        'detail': '面积缩小 {:.1f}%'.format(abs(change_rate) * 100)
                    })
                elif change_rate > 0.05:
                    record_statuses.append({
                        'record_id': r.get('id'),
                        'date': r.get('created_at', ''),
                        'status': '恶化',
                        'status_type': 'danger',
                        'detail': '面积增大 {:.1f}%'.format(change_rate * 100)
                    })
                else:
                    record_statuses.append({
                        'record_id': r.get('id'),
                        'date': r.get('created_at', ''),
                        'status': '稳定',
                        'status_type': 'warning',
                        'detail': '面积变化 {:.1f}%'.format(change_rate * 100)
                    })

    # 综合病情判定（基于最近一次 vs 上一次，如果只有一条则为初诊）
    if len(record_statuses) <= 1:
        overall_status = '初诊'
        overall_type = 'info'
    else:
        overall_status = record_statuses[-1]['status']
        overall_type = record_statuses[-1]['status_type']

    return jsonify({
        'status': 1,
        'data': {
            'count': len(completed),
            'dates': dates,
            'area': {
                'name': '肿瘤面积',
                'values': area_values,
                'trend': _judge_trend(area_values),
            },
            'perimeter': {
                'name': '肿瘤周长',
                'values': perimeter_values,
                'trend': _judge_trend(perimeter_values),
            },
            'mean': {
                'name': '灰度均值',
                'values': mean_values,
                'trend': _judge_trend(mean_values),
            },
            'std': {
                'name': '灰度方差',
                'values': std_values,
                'trend': _judge_trend(std_values),
            },
            'ellipse': {
                'name': '似圆度',
                'values': ellipse_values,
                'trend': _judge_trend(ellipse_values),
            },
            # ---- 新增字段，不影响以上任何已有数据 ----
            'diagnosis_status': {
                'overall': overall_status,
                'overall_type': overall_type,
                'records': record_statuses,
            },
        }
    })


# ============ 时序预测接口 ============

@app.route('/api/patients/<int:patient_id>/forecast', methods=['GET'])
def api_forecast(patient_id):
    """
    时序预测接口：综合 LSTM + 线性回归，预测肿瘤指标未来趋势。
    可选参数: ?steps=3 (预测步数，默认 3)
    """
    patient = database.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 0, 'msg': '患者不存在'}), 404

    predict_steps = request.args.get('steps', 3, type=int)
    predict_steps = max(1, min(predict_steps, 10))  # 限制 1~10 步

    # 取所有已完成的诊断记录，按时间正序
    records = database.get_diagnosis_records_by_patient(patient_id)
    completed = [r for r in records if r.get('status', 'completed') == 'completed']
    completed.reverse()  # 数据库返回倒序，反转为正序

    if len(completed) < 2:
        return jsonify({
            'status': 1,
            'data': {
                'msg': '历史数据不足（至少需要 2 条诊断记录）',
                'lstm_success': False,
                'predict_steps': predict_steps,
                'features': {}
            }
        })

    # 提取各指标序列
    area_values = []
    perimeter_values = []
    mean_values = []
    std_values = []
    ellipse_values = []
    dates = []

    for r in completed:
        area_values.append(r.get('area'))
        perimeter_values.append(r.get('perimeter'))

        date_str = r.get('created_at', '')
        if len(date_str) >= 10:
            dates.append(date_str[5:10])
        else:
            dates.append(date_str)

        info = r.get('image_info')
        if isinstance(info, dict):
            mean_item = info.get('mean')
            mean_values.append(mean_item[1] if isinstance(mean_item, list) and len(mean_item) > 1 else None)
            std_item = info.get('std')
            std_values.append(std_item[1] if isinstance(std_item, list) and len(std_item) > 1 else None)
            ellipse_item = info.get('ellipse')
            ellipse_values.append(ellipse_item[1] if isinstance(ellipse_item, list) and len(ellipse_item) > 1 else None)
        else:
            mean_values.append(None)
            std_values.append(None)
            ellipse_values.append(None)

    # 调用综合预测
    result = ts_predict.combined_predict(
        area_values, perimeter_values, mean_values,
        std_values, ellipse_values, predict_steps
    )

    # 生成预测日期标签（T+1, T+2, ...）
    forecast_dates = [f'预测+{i+1}' for i in range(predict_steps)]

    result['history_dates'] = dates
    result['forecast_dates'] = forecast_dates

    # 附带各指标的历史值序列（前端绘图需要）
    result['history_values'] = {
        'area': area_values,
        'perimeter': perimeter_values,
        'mean': mean_values,
        'std': std_values,
        'ellipse': ellipse_values,
    }

    return jsonify({'status': 1, 'data': result})


# ============ LLM 辅助建议接口 ============

@app.route('/api/llm/providers', methods=['GET'])
def api_llm_providers():
    """返回可用的 LLM 模型提供商和模型列表"""
    providers = llm_service.get_available_providers()
    return jsonify({'status': 1, 'data': providers})


@app.route('/api/diagnosis/<int:record_id>/llm-advice', methods=['POST'])
def api_generate_llm_advice(record_id):
    """
    基于某次诊断的真实特征值 + 历史趋势数据，调用 LLM 生成辅助建议。
    可通过 POST body 指定 provider 和 model：
      {"provider": "deepseek", "model": "deepseek-chat"}
    结果保存到 llm_reports 表。
    """
    # 从请求体读取可选的 provider / model
    req_data = request.get_json(silent=True) or {}
    provider_id = req_data.get('provider')
    model_name = req_data.get('model')

    # 1. 查询诊断记录
    record = database.get_diagnosis_record(record_id)
    if not record:
        return jsonify({'status': 0, 'msg': '诊断记录不存在'}), 404

    features = record.get('image_info')
    if not features or not isinstance(features, dict):
        features = {}  # 无特征数据时传空字典，LLM 可生成基础建议

    # 2. 获取该患者的历史趋势数据（可选）
    trend_data = None
    patient_id = record.get('patient_id')
    if patient_id:
        records = database.get_diagnosis_records_by_patient(patient_id)
        completed = [r for r in records if r.get('status', 'completed') == 'completed']
        completed.reverse()
        if len(completed) >= 2:
            # 快速构建趋势数据
            area_vals = [r.get('area') for r in completed]
            peri_vals = [r.get('perimeter') for r in completed]
            trend_data = {
                'count': len(completed),
                'area': {'values': area_vals, 'trend': _judge_trend(area_vals)},
                'perimeter': {'values': peri_vals, 'trend': _judge_trend(peri_vals)},
            }

    # 3. 调用 LLM（支持指定 provider 和 model）
    llm_result = llm_service.generate_advice(features, trend_data, provider_id, model_name)

    # 4. 保存到数据库
    report_id = None
    if llm_result['success']:
        report_id = database.add_llm_report(
            diagnosis_id=record_id,
            prompt=llm_result['prompt'],
            advice=llm_result['advice'],
            model_name=llm_result['model_name']
        )

    return jsonify({
        'status': 1 if llm_result['success'] else 0,
        'data': {
            'report_id': report_id,
            'advice': llm_result['advice'],
            'model_name': llm_result['model_name'],
            'disclaimer': llm_result['disclaimer'],
            'provider': llm_result.get('provider'),
        },
        'msg': None if llm_result['success'] else llm_result['advice']
    })


@app.route('/api/diagnosis/<int:record_id>/llm-reports', methods=['GET'])
def api_get_llm_reports(record_id):
    """查询某次诊断的所有 LLM 报告"""
    record = database.get_diagnosis_record(record_id)
    if not record:
        return jsonify({'status': 0, 'msg': '诊断记录不存在'}), 404

    reports = database.get_llm_reports_by_diagnosis(record_id)
    # 给每份报告附加免责声明
    for r in reports:
        r['disclaimer'] = llm_service.DISCLAIMER

    return jsonify({'status': 1, 'data': reports})


@app.route('/api/llm/config', methods=['POST'])
def api_update_llm_config():
    """
    运行时更新 LLM 提供商配置（仅在内存中生效，重启后失效）
    POST body: {"provider": "deepseek", "api_key": "sk-xxx", "base_url": "可选", "default_model": "可选"}
    """
    data = request.get_json(force=True) if request.is_json else request.form
    provider_id = data.get('provider')
    api_key = data.get('api_key', '')
    base_url = data.get('base_url')
    default_model = data.get('default_model')

    if not provider_id:
        return jsonify({'status': 0, 'msg': '请指定 provider'}), 400
    if not api_key:
        return jsonify({'status': 0, 'msg': '请提供 api_key'}), 400

    success, msg = llm_service.update_provider_key(provider_id, api_key, base_url, default_model)
    if success:
        return jsonify({'status': 1, 'msg': msg})
    else:
        return jsonify({'status': 0, 'msg': msg}), 400

def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.Unet(1, 1).to(device)
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load("./core/net/model.pth", weights_only=False))
        else:
            model.load_state_dict(torch.load("./core/net/model.pth", map_location='cpu', weights_only=False))
        model.eval()
        print("[INFO] 模型加载成功")
    except Exception as e:
        print(f"[WARNING] 模型加载失败: {e}")
        print("[WARNING] /upload 推理功能不可用，但其他接口正常")
        model = None
    return model


if __name__ == '__main__':
    database.init_db()
    for d in ['uploads', 'tmp/ct', 'tmp/image', 'tmp/mask', 'tmp/draw']:
        os.makedirs(d, exist_ok=True)
    with app.app_context():
        current_app.model = init_model()
    app.run(host='127.0.0.1', port=5003, debug=False)
