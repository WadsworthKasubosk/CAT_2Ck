# -*- coding: utf-8 -*-
"""生成图4-2(YOLO标注示例) 与 图4-4(前端界面布局示意)"""
import os, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
from PIL import Image

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(__file__), '论文插图_png')
DS = r'C:\Users\da983\CAT_2Ck\CTAI-master\CTAI_model\datasets\rectal_tumor_seg'

# 找一个 label 非空的样本
def pick_sample():
    lbl_dir = os.path.join(DS, 'labels', 'train')
    img_dir = os.path.join(DS, 'images', 'train')
    for f in sorted(os.listdir(lbl_dir)):
        p = os.path.join(lbl_dir, f)
        if os.path.getsize(p) > 50:
            img = os.path.join(img_dir, f.replace('.txt', '.png'))
            if os.path.isfile(img):
                return img, p
    return None, None


def gen_fig4_2():
    img_path, lbl_path = pick_sample()
    if not img_path:
        print('无可用样本'); return
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(lbl_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    img_draw = img.copy()
    for line in lines:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        pts = np.array([[coords[i]*w, coords[i+1]*h] for i in range(0, len(coords), 2)], dtype=np.int32)
        cv2.polylines(img_draw, [pts], True, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    axes[0].set_title('CT 图像 + 肿瘤轮廓多边形', fontsize=13)
    axes[0].axis('off')

    # 右侧: 显示 label txt 前若干字符
    first_line = lines[0] if lines else ''
    parts = first_line.split()
    if len(parts) > 20:
        show_line = parts[0] + ' ' + ' '.join(parts[1:15]) + '\n             ... (共 {} 个坐标点) ...\n             '.format((len(parts)-1)//2) + ' '.join(parts[-10:])
    else:
        show_line = first_line

    axes[1].axis('off')
    axes[1].text(0.02, 0.95,
                 f'标注文件: {os.path.basename(lbl_path)}\n'
                 f'图像尺寸: {w}×{h}\n'
                 f'类别数: 1 (tumor)\n'
                 f'多边形数: {len(lines)}\n\n'
                 f'格式: <class> <x1/W> <y1/H> <x2/W> <y2/H> ...\n\n'
                 f'内容示例:\n{show_line}',
                 family='monospace', fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.6', fc='#f6f8fa', ec='#c9d1d9'))
    axes[1].set_title('对应的 YOLO 分割标注文件', fontsize=13)
    out = os.path.join(OUT, '图4-2_YOLO标注格式示例.png')
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print('saved', out)


def gen_fig4_4():
    """基于 5-5 系统主界面做功能区标注"""
    src = os.path.join(OUT, '图5-5_系统主界面.png')
    if not os.path.isfile(src):
        print('5-5 不存在'); return
    im = Image.open(src)
    w, h = im.size
    fig, ax = plt.subplots(figsize=(w/140, h/140))
    ax.imshow(im)
    ax.axis('off')

    annots = [
        ( 60,  30, 400,  90, '系统标题/导航'),
        ( 60, 110, 400,  80, '患者管理 + 搜索'),
        ( 60, 195, 1360, 80, '功能入口 (患者总数 / CT / 时序预测)'),
        ( 60, 290, 1360, 320, '患者列表 (详情 · 趋势 · 删除)'),
    ]
    for x, y, ww, hh, txt in annots:
        rect = patches.Rectangle((x, y), ww, hh, linewidth=2,
                                 edgecolor='#e67e22', facecolor='none',
                                 linestyle='--')
        ax.add_patch(rect)
        ax.annotate(txt, xy=(x + ww, y + hh/2),
                    xytext=(x + ww + 20, y + hh/2),
                    fontsize=11, color='#c0392b', va='center',
                    arrowprops=dict(arrowstyle='->', color='#c0392b'))

    ax.set_xlim(-260, w)
    out = os.path.join(OUT, '图4-4_前端界面布局.png')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    gen_fig4_2()
    gen_fig4_4()
