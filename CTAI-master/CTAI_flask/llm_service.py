"""
CTAI LLM 辅助建议服务 — 多模型支持
支持模型:
  - 通义千问 (qwen-plus / qwen-turbo) — 阿里 DashScope
  - DeepSeek (deepseek-chat / deepseek-reasoner) — DeepSeek API
  - OpenAI (gpt-4o / gpt-3.5-turbo) — OpenAI API
  - 其他 OpenAI 兼容接口 (如 Ollama 本地模型)

配置方式：设置环境变量，或在 LLM_PROVIDERS 字典中直接填写 api_key
"""
import os
import json
import requests

# ============ 多模型配置 ============
# 每个 provider 的配置：api_key, base_url, default_model
# api_key 优先从环境变量读取，未设置则使用字典中的默认值（空字符串表示未配置）

LLM_PROVIDERS = {
    'qwen': {
        'name': '通义千问',
        'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'models': ['qwen-plus', 'qwen-turbo', 'qwen-max'],
        'default_model': 'qwen-plus',
    },
    'deepseek': {
        'name': 'DeepSeek',
        'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
        'base_url': 'https://api.deepseek.com/v1',
        'models': ['deepseek-chat', 'deepseek-reasoner'],
        'default_model': 'deepseek-chat',
    },
    'openai': {
        'name': 'OpenAI',
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'base_url': 'https://api.openai.com/v1',
        'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
        'default_model': 'gpt-4o-mini',
    },
    'ollama': {
        'name': 'Ollama (本地)',
        'api_key': 'ollama',  # Ollama 不需要真实 key
        'base_url': os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
        'models': ['llama3', 'qwen2', 'deepseek-r1:7b'],
        'default_model': 'llama3',
    },
}

# 默认使用的 provider 和 model（可通过环境变量覆盖）
DEFAULT_PROVIDER = os.environ.get('LLM_PROVIDER', 'qwen')
DEFAULT_MODEL = os.environ.get('LLM_MODEL', '')  # 空则使用 provider 的 default_model

# 免责声明
DISCLAIMER = '⚠️ 以上内容由 AI 生成，仅供辅助参考，不能替代专业医生的诊断和治疗方案。请遵医嘱。'


def get_available_providers():
    """
    返回所有已配置 API Key 的可用 provider 列表
    Returns: [{'id': 'qwen', 'name': '通义千问', 'models': [...], 'default_model': '...', 'available': True}, ...]
    """
    result = []
    for pid, cfg in LLM_PROVIDERS.items():
        available = bool(cfg.get('api_key'))
        result.append({
            'id': pid,
            'name': cfg['name'],
            'models': cfg['models'],
            'default_model': cfg['default_model'],
            'available': available,
        })
    return result


def build_prompt(features, trend_data=None):
    """
    根据诊断特征值和趋势数据构建 prompt

    Args:
        features: dict，单次诊断的特征值（image_info 格式）
        trend_data: dict 或 None，趋势分析数据（来自 /trend 接口）
    """
    # 提取关键特征
    feature_lines = []
    if isinstance(features, dict):
        key_features = ['area', 'perimeter', 'mean', 'std', 'ellipse',
                        'focus_x', 'focus_y', 'piandu', 'fengdu']
        for key in key_features:
            item = features.get(key)
            if isinstance(item, list) and len(item) >= 2:
                feature_lines.append(f"- {item[0]}: {item[1]}")

    feature_text = '\n'.join(feature_lines) if feature_lines else '暂无特征数据'

    # 构建趋势描述
    trend_text = ''
    if trend_data and isinstance(trend_data, dict):
        area_info = trend_data.get('area', {})
        perimeter_info = trend_data.get('perimeter', {})
        count = trend_data.get('count', 0)
        if count >= 2:
            trend_text = f"""
该患者已有 {count} 次诊断记录：
- 肿瘤面积变化趋势：{area_info.get('trend', '未知')}，历史值：{area_info.get('values', [])}
- 肿瘤周长变化趋势：{perimeter_info.get('trend', '未知')}，历史值：{perimeter_info.get('values', [])}
"""

    prompt = f"""你是一名医学影像分析助手，请根据以下直肠肿瘤 CT 图像的分析结果，给出辅助诊断建议。

## 本次诊断的肿瘤区域特征值
{feature_text}
{trend_text}
## 请给出以下内容
1. 对当前肿瘤特征的简要分析
2. 与前几次诊断对比的变化情况（如有历史数据）
3. 建议的后续检查或关注事项
4. 需要提醒医生注意的异常指标（如有）

注意：你的回答仅作为辅助参考信息，不能替代专业医生的诊断。请用简洁中文回答。"""

    return prompt


def call_llm(prompt, provider_id=None, model_name=None):
    """
    调用 LLM API（统一使用 OpenAI 兼容接口格式）

    Args:
        prompt: str, 用户 prompt
        provider_id: str, provider ID（如 'qwen', 'deepseek', 'openai', 'ollama'）
        model_name: str, 模型名（如 'deepseek-chat'）

    Returns:
        (success: bool, response_text: str, model_display: str)
    """
    # 确定 provider
    if not provider_id:
        provider_id = DEFAULT_PROVIDER
    provider = LLM_PROVIDERS.get(provider_id)
    if not provider:
        return False, f'未知的模型提供商: {provider_id}，可选: {list(LLM_PROVIDERS.keys())}', provider_id

    # 确定 model
    if not model_name:
        model_name = DEFAULT_MODEL if DEFAULT_MODEL else provider['default_model']

    # 检查 api_key
    api_key = provider.get('api_key', '')
    if not api_key:
        env_var_map = {
            'qwen': 'DASHSCOPE_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'openai': 'OPENAI_API_KEY',
        }
        env_hint = env_var_map.get(provider_id, f'{provider_id.upper()}_API_KEY')
        return False, f'{provider["name"]} 未配置 API Key，请设置环境变量 {env_hint}', f'{provider["name"]}/{model_name}'

    model_display = f'{provider["name"]}/{model_name}'
    base_url = provider['base_url'].rstrip('/')

    # 对通义千问特殊处理：优先使用 dashscope SDK
    if provider_id == 'qwen':
        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = api_key
            response = Generation.call(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                result_format='message'
            )

            if response.status_code == 200:
                text = response.output.choices[0].message.content
                return True, text, model_display
            else:
                # SDK 失败了，继续尝试 HTTP 方式
                pass
        except ImportError:
            pass  # dashscope 未安装，使用 HTTP 方式
        except Exception:
            pass  # SDK 调用异常，尝试 HTTP 方式

    # 统一的 OpenAI 兼容接口调用
    try:
        url = f'{base_url}/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2000,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            text = data['choices'][0]['message']['content']
            return True, text, model_display
        else:
            error_detail = resp.text[:500]
            return False, f'API 请求失败 ({resp.status_code}): {error_detail}', model_display

    except requests.exceptions.Timeout:
        return False, f'{provider["name"]} 请求超时，请稍后重试', model_display
    except requests.exceptions.ConnectionError:
        if provider_id == 'ollama':
            return False, f'无法连接 Ollama 服务，请确认 Ollama 已启动 ({base_url})', model_display
        return False, f'无法连接到 {provider["name"]} API，请检查网络', model_display
    except Exception as e:
        return False, f'LLM 调用异常: {str(e)}', model_display


def generate_advice(features, trend_data=None, provider_id=None, model_name=None):
    """
    完整流程：构建 prompt → 调用 LLM → 返回结果

    Args:
        features: dict, 诊断特征值
        trend_data: dict or None, 趋势数据
        provider_id: str or None, 使用哪个 provider
        model_name: str or None, 使用哪个模型

    Returns:
        dict: {success, prompt, advice, model_name, disclaimer, provider}
    """
    prompt = build_prompt(features, trend_data)
    success, advice, model_display = call_llm(prompt, provider_id, model_name)

    return {
        'success': success,
        'prompt': prompt,
        'advice': advice,
        'model_name': model_display,
        'disclaimer': DISCLAIMER,
        'provider': provider_id or DEFAULT_PROVIDER,
    }


# ============ 运行时 API Key 配置 ============
def update_provider_key(provider_id, api_key, base_url=None, default_model=None):
    """
    运行时更新某个 provider 的配置（不修改文件，仅在内存中生效）
    支持更新：api_key, base_url, default_model
    """
    if provider_id not in LLM_PROVIDERS:
        return False, f'未知的模型提供商: {provider_id}'
    LLM_PROVIDERS[provider_id]['api_key'] = api_key
    if base_url:
        LLM_PROVIDERS[provider_id]['base_url'] = base_url
    if default_model:
        LLM_PROVIDERS[provider_id]['default_model'] = default_model
    
    parts = []
    parts.append('API Key 已更新')
    if base_url:
        parts.append(f'接口地址: {base_url}')
    if default_model:
        parts.append(f'默认模型: {default_model}')
    
    return True, f'{LLM_PROVIDERS[provider_id]["name"]} ' + '，'.join(parts)
