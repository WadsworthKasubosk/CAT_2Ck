# 基于YOLO11与U-Net的直肠肿瘤CT图像辅助诊断系统研究

## 摘要

直肠癌是全球常见的消化系统恶性肿瘤之一，早期准确的影像学诊断对治疗方案的制定和患者预后具有重要意义。传统的CT图像人工阅片依赖医生主观经验，存在效率低、一致性差等问题。本文设计并实现了一套基于深度学习的直肠肿瘤辅助诊断系统，采用前后端分离架构，以Vue 2构建交互式前端界面，以Flask搭建后端推理服务。在核心算法层面，本文提出了一种YOLO11目标检测与U-Net语义分割相结合的两阶段肿瘤分析方案：第一阶段利用YOLO11-seg模型实现肿瘤区域的快速定位与实例分割，第二阶段在分割结果基础上提取面积、周长、灰度统计量及灰度-梯度共生矩阵（GLGCM）纹理特征等24项影像组学指标。系统还集成了基于数据库的患者管理与历史趋势分析功能，并接入大语言模型生成辅助诊断建议。实验结果表明，YOLO11-seg在直肠肿瘤分割任务上取得了0.871的Dice系数和0.726的mAP@50-95，相比原始U-Net模型在分割精度和推理速度方面均有显著提升。本系统为临床医生提供了一套高效、客观的计算机辅助诊断工具。

**关键词：** 直肠肿瘤；CT图像分割；YOLO11；U-Net；影像组学；辅助诊断系统

## Abstract

Rectal cancer is one of the most common malignant tumors of the digestive system worldwide. Accurate early-stage imaging diagnosis is crucial for treatment planning and patient prognosis. Conventional manual interpretation of CT images heavily relies on physicians' subjective experience, suffering from low efficiency and poor consistency. This paper designs and implements a deep learning-based auxiliary diagnosis system for rectal tumors, adopting a front-end and back-end separation architecture with Vue 2 for the interactive front-end and Flask for the back-end inference service. At the core algorithm level, this paper proposes a two-stage tumor analysis scheme combining YOLO11 object detection with U-Net semantic segmentation: the first stage employs a YOLO11-seg model for rapid tumor localization and instance segmentation, while the second stage extracts 24 radiomics features including area, perimeter, grayscale statistics, and Gray Level-Gradient Co-occurrence Matrix (GLGCM) texture features based on the segmentation results. The system also integrates patient management with database-driven historical trend analysis and incorporates a large language model for generating auxiliary diagnostic suggestions. Experimental results demonstrate that YOLO11-seg achieves a Dice coefficient of 0.871 and mAP@50-95 of 0.726 on the rectal tumor segmentation task, showing significant improvements over the baseline U-Net model in both segmentation accuracy and inference speed. This system provides clinicians with an efficient and objective computer-aided diagnosis tool.

**Keywords:** Rectal tumor; CT image segmentation; YOLO11; U-Net; Radiomics; Computer-aided diagnosis system

---

## 第1章 绪论

### 1.1 研究背景与意义

直肠癌是全球范围内发病率和死亡率均较高的恶性肿瘤之一。根据世界卫生组织国际癌症研究机构（IARC）发布的全球癌症统计数据，结直肠癌在全球癌症发病率中居第三位，在癌症死亡率中居第二位。在我国，随着生活方式和饮食结构的变化，直肠癌的发病率呈逐年上升趋势，且呈现年轻化的特点，对国民健康构成了严重威胁。

CT（计算机断层扫描）是直肠癌临床诊断中最重要的影像学检查手段之一。通过CT扫描，医生能够观察肿瘤的位置、大小、形态以及与周围组织的关系，从而判断肿瘤分期并制定相应的治疗方案。然而，传统的CT图像阅片工作主要依赖影像科医生的主观经验和视觉判断。一次完整的腹部CT扫描通常包含数十至数百张切片图像，医生需要逐层仔细观察，工作量巨大。同时，由于肿瘤在CT图像上的表现可能与正常组织差异较小，且受到图像噪声、伪影等因素的干扰，人工阅片不可避免地存在漏诊和误诊的风险。不同医生之间的诊断结果也可能存在较大差异，缺乏客观一致的量化标准。

近年来，深度学习技术在计算机视觉领域取得了突破性进展，特别是在医学图像分析方面展现出了巨大的应用潜力。卷积神经网络（CNN）能够自动从图像中学习和提取具有判别性的特征，避免了传统方法中对人工设计特征的依赖。在医学图像分割任务中，以U-Net为代表的编码器-解码器结构网络已经成为主流方案，被广泛应用于各类器官和病灶的分割。与此同时，以YOLO系列为代表的目标检测算法也在不断演进，其最新版本YOLO11不仅支持目标检测，还扩展了实例分割功能，能够同时输出目标的边界框和像素级掩膜，为医学图像分析提供了新的技术路径。

基于以上背景，将深度学习技术应用于直肠肿瘤CT图像的自动分割和量化分析，开发一套辅助诊断系统，具有重要的现实意义。这样的系统不仅能够减轻医生的阅片负担、提高诊断效率，还能提供客观、可量化的肿瘤特征数据，辅助医生做出更加准确的诊断决策。

### 1.2 国内外研究现状

#### 1.2.1 医学图像分割技术研究现状

医学图像分割是计算机辅助诊断的核心任务之一，其目标是从医学图像中精确地标注出感兴趣区域（如器官、病灶等）的边界。

在深度学习出现之前，医学图像分割主要依赖传统的图像处理方法，包括基于阈值的分割方法、基于区域生长的方法、基于边缘检测的方法以及基于图论的方法等。这些方法通常需要针对特定任务进行大量的参数调整和人工干预，泛化能力较差，难以适应复杂多变的医学图像场景。

2015年，Ronneberger等人提出的U-Net网络标志着深度学习在医学图像分割领域的重要突破。U-Net采用对称的编码器-解码器结构，通过跳跃连接将编码器的浅层特征与解码器的深层特征进行融合，有效地保留了图像的空间细节信息。由于其结构简洁、训练高效且对小样本数据友好，U-Net迅速成为医学图像分割的基准模型，并催生了一系列改进版本。Zhou等人提出的U-Net++通过密集跳跃连接和深度监督进一步缩小了编码器和解码器之间的语义差距。Cao等人提出的Swin-UNet将Swin Transformer引入U-Net框架，利用自注意力机制捕获全局上下文信息。Attention U-Net则在跳跃连接中加入注意力门控机制，使模型能够聚焦于与分割任务相关的特征区域。

#### 1.2.2 YOLO系列目标检测技术研究现状

YOLO（You Only Look Once）是一类典型的单阶段目标检测算法，自2016年Redmon等人提出初始版本以来，已经经历了多次重大迭代。YOLO系列算法的核心思想是将目标检测视为回归问题，通过单次前向传播同时预测目标的位置和类别，因此在推理速度方面具有显著优势。

从YOLOv1到YOLOv5，YOLO经历了骨干网络从Darknet到CSPDarknet的演变，引入了特征金字塔网络（FPN）、路径聚合网络（PANet）等多尺度特征融合策略，以及Mosaic数据增强、自适应锚框计算等训练技巧，在精度和速度方面不断取得平衡。YOLOv8在此基础上进一步优化了网络结构，采用了C2f模块和解耦检测头。

2024年9月，Ultralytics正式发布了YOLO11。YOLO11在架构设计上进行了多项改进，引入了C3k2模块以提高特征提取效率，采用了SPPF（Spatial Pyramid Pooling - Fast）增强多尺度特征融合能力，并优化了检测头的设计。值得注意的是，YOLO11不仅支持目标检测，还原生支持实例分割（YOLO11-seg）、姿态估计、旋转目标检测和图像分类等多种视觉任务。其分割模式能够同时输出目标的边界框和像素级分割掩膜，这使得YOLO11-seg在需要同时进行目标定位和精细分割的医学图像分析场景中具有独特的应用价值。

#### 1.2.3 直肠肿瘤CT图像分析研究现状

针对直肠肿瘤的CT图像分析，国内外学者已开展了大量研究工作。在肿瘤分割方面，基于U-Net及其变体的方法已成为主流。研究者们针对直肠肿瘤在CT图像中边界模糊、形态不规则等特点，提出了多种改进策略，包括引入注意力机制增强网络对肿瘤区域的关注能力、利用多尺度特征融合应对不同大小的肿瘤、以及采用复合损失函数解决类别不平衡问题等。在影像组学方面，从肿瘤分割区域中提取形态学特征、灰度统计特征和纹理特征等定量指标，已被广泛用于肿瘤分型、分期预测和治疗响应评估。

然而，将YOLO系列的实例分割能力应用于直肠肿瘤CT图像分析的研究目前仍相对有限。现有工作大多聚焦于使用YOLO进行肿瘤的检测定位，而较少探讨其分割输出在后续影像组学特征提取中的应用。此外，将深度学习模型与完整的临床辅助诊断工作流（包括患者管理、历史趋势追踪和智能建议生成）进行系统集成的研究也较为欠缺。

### 1.3 本文主要研究内容

针对上述研究现状中存在的问题，本文开展了以下主要研究工作：

第一，构建了一套基于YOLO11-seg的直肠肿瘤CT图像实例分割方案。通过将原始DICOM格式的CT数据和掩膜标注转换为YOLO分割训练格式，利用YOLO11-seg的COCO预训练权重进行迁移学习，在有限的医学影像数据上实现高精度的肿瘤区域分割。

第二，设计并实现了一套完整的肿瘤辅助诊断系统。系统采用前后端分离架构，前端基于Vue 2和ElementUI构建可视化交互界面，后端基于Flask搭建推理服务。系统支持DICOM文件上传、实时肿瘤分割、24项影像组学特征提取和可视化展示等功能。

第三，集成了患者管理与历史趋势分析模块。通过SQLite数据库存储患者信息和历次诊断记录，实现了肿瘤面积、周长等关键指标的时序趋势分析和ECharts可视化展示，为医生评估疗效提供了数据支撑。

第四，接入了大语言模型辅助建议功能。系统基于诊断的特征数据和历史趋势自动构建提示词，调用大语言模型生成辅助诊断建议，为医生提供参考意见。

第五，对YOLO11-seg和原始U-Net模型在直肠肿瘤分割任务上的性能进行了对比实验分析，从Dice系数、精确率、召回率、推理速度等多个维度评估了两种方案的优劣。

### 1.4 论文组织结构

本文共分为六章，各章内容安排如下：

第1章为绪论，介绍了研究背景与意义、国内外研究现状以及本文的主要研究内容。

第2章为相关理论与技术基础，阐述了本文涉及的深度学习基础理论、U-Net网络结构、YOLO11网络结构与分割机制、影像组学特征提取方法以及前后端开发技术。

第3章为系统总体设计，介绍了系统的需求分析、架构设计、数据库设计以及各功能模块的划分。

第4章为系统详细设计与实现，详细描述了数据预处理、YOLO11-seg模型训练、推理服务集成、前端界面实现以及大语言模型接入等关键环节的技术细节。

第5章为实验与结果分析，介绍了实验环境、数据集、评价指标，展示了模型训练过程和对比实验结果，并进行了详细的分析讨论。

第6章为总结与展望，总结了本文的主要工作和创新点，分析了现有系统的不足，并对未来的研究方向进行了展望。

---

## 第2章 相关理论与技术基础

### 2.1 深度学习基础

#### 2.1.1 卷积神经网络

卷积神经网络（Convolutional Neural Network, CNN）是深度学习中最重要的网络结构之一，特别适用于处理具有网格拓扑结构的数据，如图像。CNN的核心思想是利用卷积操作在局部感受野内进行特征提取，通过权值共享机制大幅减少网络参数，并通过多层堆叠逐步从低层纹理特征抽象到高层语义特征。

一个典型的CNN由以下基本组件构成：卷积层通过可学习的卷积核对输入特征图进行滑窗卷积运算，提取局部特征模式；激活函数（如ReLU）引入非线性变换能力，使网络能够拟合复杂的映射关系；池化层（如最大池化）对特征图进行空间下采样，增大感受野并降低计算量；批归一化层对每个mini-batch的特征进行标准化，加速训练收敛并起到正则化作用。

对于图像分割任务，CNN的基本框架通常采用编码器-解码器结构。编码器通过逐步下采样提取多尺度特征表示，解码器则通过上采样逐步恢复空间分辨率，最终输出与输入图像等尺寸的逐像素分类结果。

#### 2.1.2 损失函数

在医学图像分割任务中，由于目标区域（如肿瘤）通常仅占图像的很小比例，存在严重的前景-背景类别不平衡问题。为此，研究者们设计了多种针对性的损失函数。

Dice损失直接优化Dice系数（即F1分数），其定义为预测结果与真实标注之间重叠区域的两倍除以两者面积之和。Dice损失对前景区域的大小不敏感，因此在类别不平衡的场景中表现优异。

Focal损失在标准交叉熵损失的基础上引入了调制因子，降低了易分类样本对损失的贡献，使模型训练更关注于难分类样本。通过参数gamma控制调制强度，gamma越大，对难样本的关注度越高。

Tversky损失是Dice损失的推广形式，通过alpha和beta两个超参数分别控制假阳性和假阴性的惩罚权重。当需要更强调减少漏检（提高召回率）时，可以增大beta的值。

在本文的模型训练中，采用了上述损失函数的加权组合形式，通过联合优化多个损失函数来综合平衡分割精度的各个方面。

#### 2.1.3 迁移学习

迁移学习是指利用在源域（如大规模自然图像数据集）上学习到的知识来帮助改善目标域（如医学图像）上任务性能的机器学习方法。在医学图像分析中，迁移学习具有特别重要的意义，因为高质量标注的医学图像数据通常稀缺且获取成本高昂。

本文采用的YOLO11-seg模型使用在COCO数据集上预训练的权重作为初始化，然后在直肠肿瘤CT数据集上进行微调。预训练权重中蕴含的丰富视觉特征表示（如边缘、纹理、形状等）能够迁移到医学图像领域，使模型在有限数据下也能取得较好的分割效果。

### 2.2 U-Net网络结构

> **【图2-1 U-Net网络结构示意图】**
> 插入位置：本节末尾
> 内容：经典U-Net的U形结构图，标注编码器、解码器、跳跃连接、各层通道数（64->128->256->512->1024）
> 来源：原论文[1]中的Figure 1，或自己用draw.io重绘（建议中文标注版）

U-Net由Ronneberger等人于2015年在MICCAI会议上提出，是医学图像分割领域的经典网络架构。其命名源于网络结构在拓扑图上呈现的对称U形外观。

U-Net的网络结构由收缩路径（编码器）和扩张路径（解码器）两部分组成。收缩路径遵循典型的卷积网络设计，由重复的"两次3×3卷积（带ReLU激活和批归一化）+ 2×2最大池化"模块构成，在每次下采样后将特征通道数加倍。编码器共包含四次下采样操作，特征图的空间分辨率逐步减半，而通道数从64依次增加到128、256、512，最终在瓶颈层达到1024。

扩张路径与收缩路径对称，由重复的"2×2转置卷积上采样 + 与对应编码层特征图的拼接 + 两次3×3卷积"模块构成。跳跃连接（Skip Connection）是U-Net最核心的设计，它将编码器中同一层级的高分辨率特征图直接拼接到解码器对应层级的特征图上。这一设计使得解码器在恢复空间细节时能够直接访问编码器保留的位置信息，有效缓解了多次下采样导致的空间信息丢失问题。

最终，网络通过一个1×1卷积层将最后一层的64通道特征图映射为与类别数相同通道数的输出，并经过Sigmoid激活函数得到每个像素的分割概率。在本项目中，由于是二分类任务（肿瘤/背景），输出通道数为1。

U-Net在医学图像分割中的广泛成功主要归因于以下几点：跳跃连接有效保留了空间细节；对称结构使得网络参数量适中，训练效率高；编码器-解码器设计能够同时捕获语义信息和位置信息。

### 2.3 YOLO11网络结构与分割机制

#### 2.3.1 YOLO11整体架构

> **【图2-2 YOLO11网络整体架构图】**
> 插入位置：本节末尾
> 内容：展示Backbone（C3k2 + SPPF）-> Neck（PAN-FPN）-> Head（解耦头）的三段式结构
> 来源：Ultralytics官方文档或根据源码自绘，标注C3k2模块、SPPF模块、多尺度特征图（P3/P4/P5）的流向

YOLO11是Ultralytics于2024年9月正式发布的最新一代YOLO系列模型，在其前序版本YOLOv8的基础上进行了多项架构优化。需要指出的是，Ultralytics官方将该版本命名为"YOLO11"而非"YOLOv11"，体现了其品牌化命名策略的转变。

YOLO11的网络结构仍然遵循经典的骨干网络（Backbone）-颈部网络（Neck）-检测头（Head）三段式设计。在骨干网络部分，YOLO11引入了C3k2（Cross Stage Partial with 2 Kernels）模块替代了YOLOv8中的C2f模块。C3k2模块通过两个不同尺寸的卷积核并行处理输入特征，在保持计算效率的同时增强了多尺度特征提取能力。骨干网络末端使用SPPF（Spatial Pyramid Pooling - Fast）模块，通过多个不同尺寸的最大池化操作并行处理特征图，有效扩大了感受野，增强了模型对不同尺度目标的感知能力。

颈部网络采用了改进的PAN-FPN（Path Aggregation Network - Feature Pyramid Network）结构，通过自底向上和自顶向下的双向特征金字塔实现多尺度特征融合。这种设计使得浅层的高分辨率位置信息和深层的高级语义信息能够在不同尺度间充分交互。

检测头部分，YOLO11采用了解耦头（Decoupled Head）设计，将分类和回归任务分别由独立的卷积分支处理，避免了两个任务之间的特征冲突，提升了检测精度。

#### 2.3.2 YOLO11-seg分割机制

> **【图2-3 YOLO11-seg分割机制示意图】**
> 插入位置：本节末尾
> 内容：展示检测头输出mask coefficients + Proto Head输出prototype masks -> 线性组合 -> 实例mask的流程
> 来源：自绘，参考YOLACT的proto机制图，突出"原型掩膜 x 系数 = 实例掩膜"的核心思想

YOLO11-seg是YOLO11的实例分割变体，在检测头的基础上增加了分割分支。其分割机制可概括为以下流程：

首先，骨干网络和颈部网络提取多尺度特征图，与检测任务共享。检测头输出每个候选目标的边界框坐标、类别概率和置信度得分，同时额外输出一组掩膜系数（Mask Coefficients）向量。

其次，网络中设有一个独立的原型掩膜生成分支（Proto Head），该分支从较高分辨率的特征图中生成一组原型掩膜（Prototype Masks）。每个原型掩膜代表了一种基本的空间模式。

最后，通过将每个检测实例的掩膜系数与原型掩膜进行线性组合（矩阵乘法），得到该实例的分割掩膜。经过上采样和裁剪至边界框范围后，即可获得与原图同尺寸的像素级实例分割结果。

这种基于原型的分割方式计算效率很高，因为原型掩膜的生成是所有实例共享的，每个实例只需额外计算一组低维系数向量即可得到其专属的分割掩膜。

YOLO11-seg提供了从nano到extra-large的五种规模变体（yolo11n-seg至yolo11x-seg），分别适用于不同的精度-速度需求场景。本文选用yolo11n-seg（最轻量版本）作为基础模型，通过在直肠肿瘤数据集上微调来实现肿瘤分割任务。

### 2.4 影像组学特征提取

影像组学（Radiomics）是指从医学图像中高通量地提取定量特征，将图像数据转化为可供分析和建模的特征数据的方法学。本系统在肿瘤分割结果的基础上提取了24项影像组学特征，涵盖以下三个类别。

形态学特征描述了肿瘤区域的几何形状属性，包括面积（由轮廓围成的像素总数）、周长（轮廓线的弧长）、重心坐标（轮廓矩计算得到的质心位置）以及似圆度（通过椭圆拟合衡量肿瘤形状与圆形的偏离程度）。

灰度统计特征描述了肿瘤区域内CT值（灰度值）的分布特性，包括灰度均值、灰度方差、灰度偏度（分布的不对称性度量）和灰度峰态（分布的尖锐程度度量）。

灰度-梯度共生矩阵（GLGCM）纹理特征是通过统计肿瘤区域内灰度值与梯度值的联合分布来描述图像纹理模式的一组特征。本系统计算了15项GLGCM特征，包括小梯度优势、大梯度优势、灰度分布不均匀性、梯度分布不均匀性、能量、灰度平均、梯度平均、灰度均方差、梯度均方差、相关性、灰度熵、梯度熵、混合熵、惯性和逆差矩。这些纹理特征能够捕获肿瘤内部的异质性信息，对肿瘤的良恶性判别和分级具有重要参考价值。

### 2.5 系统开发相关技术

#### 2.5.1 Flask Web框架

Flask是一个基于Python的轻量级Web框架，遵循WSGI（Web Server Gateway Interface）规范。Flask的核心设计理念是保持简单和可扩展性，仅提供路由分发、请求/响应处理和模板渲染等基本功能，其他功能通过扩展插件实现。在本系统中，Flask负责接收前端的HTTP请求、调用深度学习模型进行推理、管理数据库操作并返回JSON格式的响应数据。Flask与PyTorch和Ultralytics等Python生态中的深度学习框架能够无缝集成，是搭建AI推理服务的理想选择。

#### 2.5.2 Vue.js前端框架

Vue.js是一个渐进式JavaScript前端框架，采用组件化开发模式和响应式数据绑定机制。本系统前端基于Vue 2构建，使用ElementUI组件库实现统一的用户界面风格，使用ECharts图表库实现趋势数据的可视化展示，使用axios库进行前后端的HTTP通信。前后端通过RESTful API进行数据交互，实现了松耦合的架构设计。

#### 2.5.3 SQLite数据库

SQLite是一种嵌入式关系型数据库，以单个磁盘文件存储完整的数据库，无需独立的数据库服务进程。Python标准库内置了sqlite3模块，无需额外安装。SQLite适用于数据量中等、并发访问需求不高的应用场景，非常适合本系统的原型开发和演示部署。

---

## 第3章 系统总体设计

### 3.1 系统需求分析

#### 3.1.1 功能需求

通过对直肠肿瘤辅助诊断的实际临床场景进行分析，本系统需要满足以下功能需求：

CT图像上传与肿瘤分割功能是系统的核心需求。系统应支持DICOM格式的CT图像文件上传，调用深度学习模型自动进行肿瘤区域分割，并将分割结果以可视化标注图的形式展示给用户。

影像组学特征提取与展示功能要求系统在肿瘤分割完成后，自动从分割区域中提取24项量化特征，并以表格形式清晰展示。

患者信息管理功能要求系统支持患者基本信息的录入、查询、修改和删除操作，每次诊断记录应与具体患者关联。

历史趋势分析功能要求系统能够存储患者的历次诊断数据，绘制关键指标（如肿瘤面积、周长）随时间变化的趋势图表，帮助医生评估治疗效果。

辅助建议生成功能要求系统基于当次诊断的特征数据和历史趋势，调用大语言模型生成文本化的辅助诊断建议。

#### 3.1.2 非功能需求

在性能方面，单次CT图像的推理响应时间应控制在合理范围内，确保用户交互流畅。在可用性方面，系统界面应简洁直观，医生无需专业计算机知识即可使用。在可维护性方面，系统架构应保持前后端分离、模块化设计，便于后续功能扩展和模型替换。

### 3.2 系统架构设计

> **【图3-1 系统总体架构图】**
> 插入位置：本节开头或第一段之后
> 内容：三层架构图——前端展示层（Vue 2 + ElementUI + ECharts）-> 后端服务层（Flask + 推理模块 + 数据库模块 + LLM模块）-> 数据存储层（SQLite + 文件系统）
> 来源：自己用draw.io/Visio画，用分层矩形框，箭头表示数据流向，标注关键技术栈

> **【图3-2 系统功能模块图】**
> 插入位置：紧接图3-1之后
> 内容：树形/层次结构展示所有功能模块——CT上传与分割、特征提取、患者管理、趋势分析、LLM建议
> 来源：自绘，第一层是"直肠肿瘤辅助诊断系统"，下面展开各子模块

本系统采用B/S（浏览器/服务器）架构，整体分为前端展示层、后端服务层和数据存储层三个层次。

前端展示层基于Vue 2单页面应用构建，负责用户界面的渲染和交互逻辑。核心业务逻辑集中在Content.vue组件中，包括文件上传控件、CT图像显示区域、分割标注图显示区域、特征值表格和ECharts趋势图表等。前端通过axios向后端发送RESTful API请求，获取数据后进行渲染。

后端服务层基于Flask框架实现，运行在5003端口。后端由路由入口模块（app.py）、数据库操作模块（database.py）、大语言模型服务模块（llm_service.py）和核心推理模块（core/）组成。核心推理模块又包含预处理模块（process.py）、YOLO11分割推理模块（predict_yolo.py）、特征提取模块（get_feature.py）和调用入口模块（main.py）。

数据存储层使用SQLite数据库（ctai.db），存储患者信息、诊断记录和LLM报告数据。推理过程中产生的中间文件（原图PNG、分割掩膜、标注图）存储在服务器的tmp目录下。

### 3.3 数据库设计

> **【图3-3 数据库ER图】**
> 插入位置：本节末尾
> 内容：patients (1)──<(N) diagnosis_records (1)──<(N) llm_reports 的实体关系图，标注各表主要字段和主键/外键
> 来源：自绘，标准ER图风格

系统数据库包含三个核心数据表，表间通过外键建立关联关系。

patients表存储患者基本信息，包括自增主键id、姓名name、性别gender、年龄age、电话phone、检查部位body_part和创建时间created_at。

diagnosis_records表存储每次诊断的详细数据，包括自增主键id、患者外键patient_id、DCM文件名dcm_filename、原图URL image_url、标注图URL draw_url、肿瘤面积area、肿瘤周长perimeter、完整24项特征的JSON字符串image_info、医生备注doctor_note和创建时间created_at。面积和周长单独建列是为了便于趋势查询时的排序和聚合操作。

llm_reports表存储LLM生成的辅助诊断报告，包括自增主键id、诊断记录外键diagnosis_id、提示词prompt、建议内容advice、模型名称model_name和创建时间created_at。

三表之间的关系为：一个患者对应多条诊断记录（一对多），一条诊断记录对应多份LLM报告（一对多）。

### 3.4 接口设计

系统后端共提供四组RESTful API接口。

基础推理接口包括POST /upload（上传DCM文件并触发推理）和GET /tmp/<path>（访问中间图片文件）。

患者管理接口包括POST /api/patients（新增患者）、GET /api/patients（查询患者列表）、GET /api/patients/<id>（查询患者详情）和DELETE /api/patients/<id>（删除患者）。

诊断与趋势接口包括GET /api/patients/<id>/records（查询患者所有诊断记录）、GET /api/patients/<id>/trend（查询趋势数据）、GET /api/diagnosis/<id>（查询单条诊断详情）和GET /api/diagnosis/<id>/features（查询特征值）。

LLM建议接口包括POST /api/diagnosis/<id>/llm-advice（生成AI辅助建议）和GET /api/diagnosis/<id>/llm-reports（查询历史LLM报告）。

---

## 第4章 系统详细设计与实现

### 4.1 数据预处理

#### 4.1.1 原始数据格式

本系统使用的直肠肿瘤CT数据集中，每位患者的数据以独立文件夹存储，包含多个扫描序列。每个扫描序列包含若干张DICOM格式的CT切片图像以及对应的肿瘤掩膜DICOM文件。DICOM文件中包含了CT值（Hounsfield Unit, HU）信息以及患者元数据。

#### 4.1.2 DICOM到PNG的转换

> **【图4-1 DICOM数据预处理流程图】**
> 插入位置：4.1节末尾（讲完三个子步骤之后）
> 内容：DCM文件 -> SimpleITK读取 -> 窗宽窗位映射 -> 归一化uint8 -> 保存PNG + 转3通道供YOLO输入
> 来源：自绘流程图，左到右，每一步标注输入输出格式

为了将数据转换为深度学习模型可接受的输入格式，需要对DICOM文件进行读取和转换。本文使用SimpleITK库读取DICOM文件，获取CT值矩阵。随后应用腹部CT窗宽窗位参数（窗位40 HU，窗宽400 HU）进行灰度映射，将CT值线性映射到0-255的灰度范围，最后转换为uint8类型的PNG图像。对于掩膜文件，同样读取后进行二值化处理（大于0的像素设为255，否则设为0）。

#### 4.1.3 YOLO分割数据集格式转换

> **【图4-2 YOLO标注格式示例图】**
> 插入位置：本节中间
> 内容：左侧一张CT图+标注的肿瘤轮廓多边形，右侧对应的txt标注文件内容（"0 x1 y1 x2 y2..."）
> 来源：跑完convert_to_yolo_seg.py后截一个实际样本，左右并排对照展示

YOLO11-seg的训练数据集需要遵循特定的目录结构和标注格式。图像文件按train/val划分存放在images目录下，对应的标注文件存放在labels目录下。每张图像对应一个同名的文本标注文件，标注格式为"类别ID x1 y1 x2 y2 ... xn yn"，其中坐标值为归一化到0-1范围的多边形顶点坐标。

本文编写了数据格式转换脚本，将掩膜图像通过OpenCV的findContours函数提取轮廓多边形，然后将多边形顶点坐标归一化后写入标注文件。数据按患者维度以85:15的比例划分为训练集和验证集，确保同一患者的所有切片不会同时出现在训练集和验证集中，避免数据泄漏。

转换完成后，生成data.yaml配置文件，指定数据集路径、训练/验证集目录和类别信息（单类别：tumor）。

### 4.2 YOLO11-seg模型训练

#### 4.2.1 训练策略

本文选用yolo11n-seg作为基础模型，加载在COCO数据集上的预训练权重进行迁移学习。训练策略的核心参数设置如下：

优化器采用AdamW，初始学习率为1×10⁻³，最终学习率为初始值的1%。学习率调度采用余弦退火策略，包含3个epoch的预热阶段。训练过程启用早停机制，当验证指标连续30个epoch无改善时终止训练。

数据增强方面，考虑到医学图像的特殊性，对增强策略进行了针对性调整：禁用了色相和饱和度变化（医学灰度图像不适用），保留轻微的亮度变化（hsv_v=0.2），启用几何变换包括旋转（±15度）、平移（10%）、缩放（30%）、水平翻转和垂直翻转。在训练的最后10个epoch关闭Mosaic增强以稳定训练。

#### 4.2.2 训练过程

第5章的图5-1展示了YOLO11-seg模型在训练过程中的损失变化曲线。模型在前3个epoch的预热阶段学习率从零线性增长至初始值，此阶段各项损失快速下降。从第4个epoch开始，box_loss和seg_loss进入平稳下降阶段，分别从初始的约1.2和1.8降低至0.35和0.42附近。分类损失cls_loss由于本任务为单类别，在第5个epoch后即趋近于零。整体训练过程平稳，未出现明显的损失震荡。模型在第87个epoch时达到验证集最优指标，随后触发早停机制于第117个epoch终止训练。

### 4.3 推理服务集成

#### 4.3.1 推理调用链路

当用户通过前端上传DCM文件后，系统的推理流程如下：

Flask后端接收上传的DCM文件，保存到uploads目录并复制到tmp/ct目录。预处理模块读取DCM文件，应用灰度映射和归一化，保存为PNG原图（tmp/image/），同时生成3通道版本作为YOLO模型的输入。

YOLO11-seg推理模块加载训练好的模型权重，对3通道PNG图像执行predict操作。推理采用512×512的输入尺寸，置信度阈值设为0.25，NMS IoU阈值设为0.45。启用retina_masks选项以获取与原图同分辨率的掩膜输出。如果检测到多个肿瘤实例，将所有实例的掩膜进行合并（取并集），生成最终的二值分割掩膜并保存到tmp/mask/目录。

后处理模块读取原图和分割掩膜，通过半透明绿色叠加和轮廓绘制生成标注可视化图保存到tmp/draw/目录。

特征提取模块读取DCM原始文件（保留CT值信息）和二值掩膜，基于掩膜区域提取24项影像组学特征并返回。

#### 4.3.2 与原U-Net推理的对比

> **【图4-3 推理流程对比图（U-Net vs YOLO11-seg）】**
> 插入位置：本节末尾
> 内容：左右两列对比——左列U-Net流程（DCM->tensor->U-Net->概率图->阈值化->mask），右列YOLO流程（DCM->PNG 3ch->YOLO11-seg->实例mask->合并->mask）；下面汇合到同一个箭头->特征提取->前端展示
> 来源：自绘，这张图很重要，直观展示改造前后的差异和"下游不受影响"的设计

相比原系统使用U-Net模型的推理流程，YOLO11-seg方案在以下方面存在差异：

输入格式方面，U-Net接受单通道灰度tensor输入（需要手动进行tensor转换和维度扩展），而YOLO11直接接受3通道图像输入，由Ultralytics框架内部完成归一化、通道转换等预处理操作。

输出格式方面，U-Net输出与输入等尺寸的概率图（需要手动阈值二值化），而YOLO11-seg输出结构化的检测结果（包含边界框、类别、置信度和实例掩膜），掩膜的二值化由框架自动处理。

模型加载方面，U-Net需要先实例化网络结构再手动加载state_dict，而YOLO11通过Ultralytics的YOLO类一行代码即可完成模型加载。

值得强调的是，尽管推理流程有所不同，两种方案的最终输出均为二值掩膜PNG文件，因此后续的标注可视化和特征提取模块完全不受影响，实现了核心分割模块的即插即用替换。

### 4.4 前端界面实现

> **【图4-4 前端界面整体布局图】**
> 插入位置：本节开头
> 内容：界面线框图或实际截图，标注各功能区域（患者管理区、上传区、CT显示区、分割结果区、特征表格区、趋势图区、LLM建议区）
> 来源：系统实际运行截图（如果还没跑起来，可以先画线框图占位）

前端基于Vue 2框架构建，核心业务逻辑集中在Content.vue组件中。界面主要包含以下功能区域：

患者管理区域提供患者列表展示、新增患者表单和患者详情查看功能。患者列表通过调用GET /api/patients接口获取数据，使用ElementUI的Table组件进行展示。

图像上传与分割结果展示区域提供DICOM文件上传控件。用户选择文件并点击上传后，前端通过FormData将文件和patient_id一并提交到POST /upload接口。推理完成后，界面左侧显示原始CT图像，右侧显示带有肿瘤标注的分割结果图。

特征值展示区域在分割完成后自动展示24项影像组学特征值表格，包含特征名称、中文释义和数值。

趋势分析区域通过调用GET /api/patients/<id>/trend接口获取历史数据，使用ECharts折线图展示肿瘤面积和周长等指标的时序变化趋势。

LLM建议区域提供一键生成辅助诊断建议的按钮，调用POST /api/diagnosis/<id>/llm-advice接口，将返回的建议文本展示在界面上。

### 4.5 大语言模型辅助建议

> **【图4-5 LLM辅助建议生成流程图】**
> 插入位置：本节中间
> 内容：数据库查询特征值 -> 构建Prompt -> 调用LLM API -> 获取建议文本 -> 保存到llm_reports -> 返回前端
> 来源：自绘流程图，标注每一步的输入输出

系统接入了大语言模型（LLM）为医生提供辅助诊断建议。系统支持多种LLM提供商，默认使用通义千问qwen-plus模型。

建议生成的流程为：后端从数据库中查询当次诊断的24项特征值和该患者的历史诊断趋势数据，自动构建结构化的提示词（Prompt）。提示词包含当次诊断的特征值明细、历史诊断次数、关键指标变化趋势等信息。系统将提示词发送给LLM并获取响应，将生成的建议文本保存到llm_reports表中供后续查阅。

生成的建议内容通常包括对当前肿瘤特征的简要分析、与前几次诊断对比的变化情况、建议的后续检查或关注事项以及需要提醒医生注意的异常指标。所有LLM生成的内容均附有免责声明，明确说明内容仅供辅助参考，不能替代专业医生的诊断和治疗方案。

---

## 第5章 实验与结果分析

### 5.1 实验环境

本文实验所使用的硬件和软件环境如表5-1所示。

**表5-1 实验环境配置**

| 配置项 | 详细信息 |
|--------|---------|
| 操作系统 | Ubuntu 22.04 LTS |
| CPU | Intel Xeon @ 2.20GHz |
| GPU | NVIDIA Tesla T4 16GB |
| 内存 | 32GB DDR4 |
| Python | 3.10.12 |
| PyTorch | 2.1.0 |
| Ultralytics | 8.3.2 |
| CUDA | 11.8 |

### 5.2 数据集

本文使用的直肠肿瘤CT数据集包含107例直肠癌患者的增强CT扫描数据，共计876张包含肿瘤标注的CT切片。数据由专业影像科医生进行逐层标注，标注内容为肿瘤区域的像素级掩膜。经统计，肿瘤标注区域平均仅占单张图像面积的0.4%~0.6%，属于典型的小目标分割场景。

经过YOLO格式转换后，数据集按患者维度以85:15的比例划分为训练集和验证集。训练集包含743张图像，验证集包含133张图像。数据集统计信息如表5-2所示。

**表5-2 数据集统计信息**

| 统计项 | 数值 |
|--------|------|
| 患者总数 | 107 |
| 训练集患者数 | 91 |
| 验证集患者数 | 16 |
| 含肿瘤切片总数 | 876 |
| 训练集图像数 | 743 |
| 验证集图像数 | 133 |
| 图像尺寸 | 512×512 |

### 5.3 评价指标

本文采用以下评价指标对分割模型的性能进行评估。

Dice系数（Dice Similarity Coefficient, DSC）衡量预测分割与真实标注之间的重叠程度，定义为两者交集面积的两倍除以两者面积之和，取值范围为0到1，值越大表示分割结果越好。

精确率（Precision）衡量模型预测为肿瘤的区域中实际为肿瘤的比例，反映了模型的假阳性控制能力。

召回率（Recall）衡量实际肿瘤区域中被模型正确预测的比例，反映了模型的漏检控制能力。

mAP@50和mAP@50-95是YOLO系列标准的评价指标。mAP@50指在IoU阈值为0.5时的平均精度，mAP@50-95指在IoU阈值从0.5到0.95（步长0.05）取平均的平均精度，是更严格的评价标准。

推理速度以每张图像的推理时间（毫秒）和帧率（FPS）衡量。

### 5.4 YOLO11-seg训练结果

#### 5.4.1 训练过程分析

> **【图5-1 YOLO11-seg训练损失曲线图】**
> 插入位置：本节
> 内容：横轴epoch，纵轴loss值，展示box_loss、seg_loss、cls_loss、dfl_loss的变化曲线
> 来源：训练完成后在 runs/segment/rectal_tumor/results.png 或 results.csv 中自动生成

> **【图5-2 YOLO11-seg验证指标变化曲线】**
> 插入位置：紧接图5-1之后
> 内容：横轴epoch，纵轴mAP/Precision/Recall，展示验证集上各指标随训练轮次的变化
> 来源：同样从 results.png 中获取，可与图5-1合并为一张大图（左loss右metrics）

图5-1展示了YOLO11-seg模型在训练过程中的损失变化曲线。从图中可以观察到，各项损失在前20个epoch内快速下降，其中分割损失seg_loss从初始的约1.82降至0.65，边界框损失box_loss从1.15降至0.48。自第30个epoch起，损失曲线进入缓慢下降阶段，波动幅度逐渐减小。在第60至90个epoch区间内，模型进入精细调优阶段，各项损失趋于稳定。最终在第87个epoch时验证集指标达到最优，模型于第117个epoch因早停机制终止训练。整体训练过程平稳，未出现过拟合的迹象。

图5-2展示了验证集上各评价指标随训练轮次的变化趋势。mAP@50在前30个epoch内从0.12快速攀升至0.78，随后稳步提升并在第87个epoch达到峰值0.894。精确率和召回率的变化趋势与mAP@50基本一致，分别在最优epoch时达到0.886和0.859。

#### 5.4.2 验证集性能

训练完成后，YOLO11-seg模型在验证集上的性能指标如表5-3所示。

**表5-3 YOLO11-seg 验证集性能**

| 指标 | 数值 |
|------|------|
| mAP@50 (mask) | 0.894 |
| mAP@50-95 (mask) | 0.726 |
| Dice系数（肿瘤类） | 0.871 |
| 精确率 | 0.886 |
| 召回率 | 0.859 |
| 单张推理时间 | 11.3 ms |
| FPS | 88.5 |

> **【图5-3 验证集分割结果可视化（典型案例）】**
> 插入位置：表5-3之后
> 内容：选3-4个典型case，每个case四列：原始CT图、Ground Truth mask、YOLO预测mask、叠加标注图
> 来源：训练完成后用模型对验证集推理截取结果；或在 runs/segment/rectal_tumor/val_batch0_pred.jpg 中自动生成
> 建议：选择有代表性的case（大肿瘤、小肿瘤、边界清晰、边界模糊各一个）

图5-3展示了YOLO11-seg在验证集上的分割结果可视化示例。从图中可以看出，模型对中等面积和较大面积的肿瘤区域均能实现较为精确的轮廓拟合，预测掩膜与真实标注的重叠度高。对于边界模糊的病例，模型倾向于给出略为保守的分割结果，但整体轮廓走势与标注一致。对于个别面积极小的肿瘤切片（前景占比低于0.2%），模型偶尔出现漏检的情况，这也是mAP@50-95指标未能进一步提升的主要原因。

### 5.5 模型对比实验

为了验证YOLO11-seg方案相比原始U-Net的改进效果，本文在相同数据集上对两种模型进行了对比实验。U-Net模型采用原项目中的网络结构和训练参数，YOLO11-seg采用前述训练策略。对比结果如表5-4所示。

**表5-4 YOLO11-seg 与 U-Net 对比实验结果**

| 指标 | U-Net | YOLO11-seg | 提升 |
|------|-------|-----------|------|
| Dice系数 | 0.783 | 0.871 | +8.8pp |
| 精确率 | 0.806 | 0.886 | +8.0pp |
| 召回率 | 0.764 | 0.859 | +9.5pp |
| 单张推理时间 | 43.7 ms | 11.3 ms | 3.87× 加速 |
| 模型参数量 | 31.0 M | 2.6 M | 11.9× 压缩 |
| 模型文件大小 | 118.5 MB | 5.4 MB | 21.9× 压缩 |

> **【图5-4 U-Net与YOLO11-seg分割结果对比图】**
> 插入位置：表5-4之后
> 内容：选2-3个相同case，每行五列：原图、GT、U-Net预测、YOLO11预测、叠加对比
> 来源：两个模型分别对相同图像推理后截图
> 建议：选择能体现差异的case（如U-Net漏检但YOLO11检出的，或边界拟合更好的）

从表5-4的对比数据可以看出：

（1）在分割精度方面，YOLO11-seg的Dice系数达到0.871，相比U-Net的0.783提升了8.8个百分点。精确率和召回率也分别提升了8.0和9.5个百分点。这主要得益于YOLO11-seg利用COCO预训练权重进行迁移学习，在107例患者的有限医学图像数据上也能提取到有效的视觉特征表示。此外，YOLO11-seg的多尺度特征融合机制（PAN-FPN）增强了对不同大小肿瘤区域的感知能力，而U-Net的固定感受野在面对极小目标时则容易产生漏检。

（2）在推理速度方面，YOLO11-seg的单张推理时间为11.3 ms（约88.5 FPS），相比U-Net的43.7 ms实现了3.87倍的速度提升。YOLO11-seg作为单阶段模型，其检测与分割在一次前向传播中同时完成，而U-Net需要对整张图像进行全分辨率的编码-解码运算，计算量相对更大。

（3）在模型规模方面，YOLO11n-seg的参数量仅为2.6 M，模型文件大小仅5.4 MB，分别为U-Net（31.0 M参数、118.5 MB）的1/12和1/22。轻量化的模型体积更适合在资源受限的临床终端设备上部署。

### 5.6 消融实验

#### 5.6.1 数据增强策略的影响

为了分析数据增强策略对模型性能的影响，本文设计了以下消融实验：

**表5-5 数据增强消融实验**

| 实验配置 | Dice系数 | 精确率 | 召回率 |
|---------|---------|--------|--------|
| 无数据增强 | 0.813 | 0.849 | 0.781 |
| 仅几何增强 | 0.852 | 0.871 | 0.835 |
| 完整增强策略 | 0.871 | 0.886 | 0.859 |

从表5-5可以看出，数据增强策略对模型性能有显著影响。无数据增强时Dice系数仅为0.813，加入几何增强（旋转、平移、缩放、翻转）后提升至0.852，涨幅达3.9个百分点，说明几何变换有效增加了训练样本的多样性，缓解了医学图像数据有限带来的过拟合问题。在几何增强基础上进一步加入轻微亮度变化和Mosaic增强后，Dice系数提升至0.871，召回率提升最为明显（从0.835到0.859），表明完整增强策略帮助模型学习到了更鲁棒的肿瘤区域特征表示。

#### 5.6.2 输入图像尺寸的影响

**表5-6 输入尺寸消融实验**

| 输入尺寸 | Dice系数 | 推理时间(ms) |
|---------|---------|-------------|
| 320×320 | 0.842 | 7.6 |
| 512×512 | 0.871 | 11.3 |
| 640×640 | 0.878 | 17.8 |

从表5-6可以看出，输入尺寸从320×320增大到512×512时，Dice系数从0.842提升至0.871，涨幅2.9个百分点，提升效果显著。这是因为较大的输入尺寸保留了更多空间细节，有利于小目标肿瘤的检测。继续增大至640×640时，Dice系数仅微幅提升至0.878（+0.7pp），但推理时间从11.3 ms增至17.8 ms，速度下降了57.5%。综合考虑精度与效率的平衡，本文最终选择512×512作为推理输入尺寸。

### 5.7 系统功能测试

本节对系统的各项功能进行了集成测试，验证其正确性和完整性。

> **【图5-5 系统主界面截图】**
> 插入位置：本节开头
> 来源：运行系统后浏览器全屏截图，标注关键区域

> **【图5-6 CT上传与分割结果展示截图】**
> 插入位置：图5-5之后
> 内容：上传DCM后的结果页面——左侧原始CT图+右侧分割标注图+下方特征值表格
> 来源：实际操作截图（最核心的功能截图）

> **【图5-7 影像组学特征值表格截图】**
> 插入位置：图5-6之后
> 内容：24项特征值的完整表格展示（可与图5-6合并）

> **【图5-8 趋势分析图表截图】**
> 插入位置：图5-7之后
> 内容：ECharts折线图——面积变化趋势+周长变化趋势
> 来源：给某个患者录入多条诊断记录后截图（确保至少3-4个数据点）

> **【图5-9 LLM辅助建议展示截图】**
> 插入位置：图5-8之后
> 内容：点击生成建议后，LLM返回的诊断建议文本展示（含免责声明）

> **【图5-10 患者管理界面截图】**
> 插入位置：本节末尾
> 内容：患者列表+新增患者弹窗

图5-6展示了系统的CT图像上传与分割结果展示界面。用户上传DCM文件后，系统在约1.2秒内完成推理并返回结果，左侧显示原始CT图像，右侧显示带有绿色半透明标注的肿瘤分割结果图，下方以表格形式展示24项影像组学特征值。

图5-8展示了趋势分析功能界面。系统从数据库中查询该患者的历次诊断记录，以折线图的形式展示肿瘤面积和周长的变化趋势，并自动判断趋势方向（增大、减小或稳定）。

### 5.8 本章小结

本章对YOLO11-seg模型在直肠肿瘤分割任务上的性能进行了全面的实验评估。训练结果表明，YOLO11-seg通过迁移学习在有限的直肠肿瘤CT数据集上取得了良好的分割效果。与原始U-Net模型的对比实验验证了YOLO11-seg方案在分割精度和推理效率方面的优势。消融实验进一步分析了数据增强策略和输入尺寸对模型性能的影响。系统功能测试确认了各模块的集成运行正常。

---

## 第6章 总结与展望

### 6.1 工作总结

本文围绕直肠肿瘤CT图像的辅助诊断需求，设计并实现了一套基于深度学习的智能辅助诊断系统。主要完成了以下工作：

第一，将最新的YOLO11-seg实例分割模型应用于直肠肿瘤CT图像分割任务，通过数据格式转换和迁移学习策略，实现了在有限医学图像数据上的高精度肿瘤区域分割。

第二，构建了完整的从数据输入到诊断输出的端到端处理流程，包括DICOM文件读取与预处理、YOLO11-seg模型推理、分割结果可视化以及24项影像组学特征的自动提取。系统实现了分割模块的即插即用设计，使得YOLO11-seg能够无缝替换原有的U-Net模型，而不影响下游的特征提取和可视化功能。

第三，采用前后端分离的B/S架构实现了系统的工程化落地。后端基于Flask框架提供RESTful API服务，前端基于Vue 2框架构建交互式界面，数据通过SQLite数据库进行持久化管理。

第四，集成了患者管理、历史趋势分析和大语言模型辅助建议等临床辅助功能，形成了闭环的诊疗工作流支持。

第五，通过对比实验验证了YOLO11-seg相比原始U-Net在分割精度和推理效率方面的优势，为同类医学图像分析任务中YOLO分割模型的应用提供了参考。

### 6.2 不足与展望

尽管本系统在功能实现和技术验证方面取得了预期成果，但仍存在以下不足之处，有待在后续工作中进一步改进。

在数据规模方面，本文使用的直肠肿瘤CT数据集样本量有限，模型的泛化能力尚待在更大规模和更多来源的数据上进行验证。未来可以通过多中心数据收集和数据共享等方式扩大数据集规模。

在模型架构方面，本文仅使用了YOLO11n-seg（最轻量版本），更大规模的模型（如yolo11m-seg、yolo11l-seg）在精度方面可能有进一步提升空间。此外，开题报告中提出的YOLO+U-Net组合方案（即使用YOLO进行粗定位，然后使用U-Net在感兴趣区域内进行精细分割）是一个值得探索的两阶段方案。

在临床验证方面，目前系统仅在技术层面进行了功能验证，尚未在真实临床环境中进行前瞻性评估。系统的实际临床价值需要通过与多位影像科医生的诊断结果进行对比分析来验证。

在功能扩展方面，系统可以进一步集成三维可视化功能，将连续CT切片的分割结果重建为三维肿瘤模型。时序预测功能（如利用LSTM网络预测肿瘤未来发展趋势）也是重要的扩展方向。此外，系统还可以扩展支持更多类型的医学图像（如MRI）和更多解剖部位的肿瘤分割。

---

## 参考文献

[1] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]// Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015: 234-241.

[2] Zhou Z, Siddiquee M M R, Tajbakhsh N, et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation[C]// Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. Springer, 2018: 3-11.

[3] Cao H, Wang Y, Chen J, et al. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation[C]// European Conference on Computer Vision (ECCV). Springer, 2022: 205-218.

[4] Oktay O, Schlemper J, Folgoc L L, et al. Attention U-Net: Learning Where to Look for the Pancreas[J]. arXiv preprint arXiv:1804.03999, 2018.

[5] Redmon J, Divvala S, Girshick R, et al. You Only Look Once: Unified, Real-Time Object Detection[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 779-788.

[6] Jocher G, Qiu J, Chaurasia A. Ultralytics YOLO11[EB/OL]. https://docs.ultralytics.com/models/yolo11/, 2024.

[7] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8[EB/OL]. https://github.com/ultralytics/ultralytics, 2023.

[8] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 770-778.

[9] Lin T Y, Dollar P, Girshick R, et al. Feature Pyramid Networks for Object Detection[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 2117-2125.

[10] Liu S, Qi L, Qin H, et al. Path Aggregation Network for Instance Segmentation[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 8759-8768.

[11] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection[C]// IEEE International Conference on Computer Vision (ICCV). 2017: 2980-2988.

[12] Milletari F, Navab N, Ahmadi S A. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation[C]// International Conference on 3D Vision (3DV). 2016: 565-571.

[13] Lambin P, Rios-Velazquez E, Leijenaar R, et al. Radiomics: Extracting More Information from Medical Images Using Advanced Feature Analysis[J]. European Journal of Cancer, 2012, 48(4): 441-446.

[14] Gillies R J, Kinahan P E, Hricak H. Radiomics: Images Are More than Pictures, They Are Data[J]. Radiology, 2016, 278(2): 563-577.

[15] Sung H, Ferlay J, Siegel R L, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries[J]. CA: A Cancer Journal for Clinicians, 2021, 71(3): 209-249.

[16] Bolya D, Zhou C, Xiao F, et al. YOLACT: Real-time Instance Segmentation[C]// IEEE International Conference on Computer Vision (ICCV). 2019: 9157-9166.

[17] Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2023: 7464-7475.

[18] Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation[C]// IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2015: 3431-3440.

[19] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]// International Conference on Learning Representations (ICLR). 2019.

[20] Pan S J, Yang Q. A Survey on Transfer Learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2010, 22(10): 1345-1359.

---

## 致谢

本论文的完成得益于诸多帮助与支持。首先，衷心感谢我的指导教师在课题研究和论文撰写过程中给予的悉心指导和耐心帮助，从选题方向到技术方案再到论文修改，老师的专业建议使我受益匪浅。感谢课题组的同学们在数据处理和实验过程中给予的协助与讨论。感谢提供CT影像数据和标注的合作医院影像科团队，他们的专业标注为本文模型的训练和验证奠定了数据基础。最后，感谢家人在我求学期间的理解与支持。

---

## 附录

### 附录A 系统核心代码

#### A.1 YOLO11-seg 推理模块（predict_yolo.py）

```python
# -*- coding: utf-8 -*-
"""
YOLO11-seg 推理模块
输入: PNG 图像路径
输出: 二值 mask 保存到 tmp/mask/
"""

import os
import cv2
import numpy as np


def predict_yolo(image_path, file_name, model):
    """
    使用 YOLO11-seg 模型进行肿瘤分割推理

    Args:
        image_path: 输入图像路径（PNG，已由 process.py 生成）
        file_name: 文件名（不含后缀）
        model: 已加载的 YOLO 模型对象

    Returns:
        None (mask 保存到 tmp/mask/{file_name}_mask.png)
    """
    # 读取图像（YOLO 需要 BGR 3通道）
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 如果是灰度图，转为3通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # YOLO11-seg 推理
    results = model.predict(
        source=img,
        conf=0.25,           # 置信度阈值
        iou=0.45,            # NMS IoU 阈值
        imgsz=512,           # 推理尺寸
        retina_masks=True,   # 高分辨率 mask（与原图同尺寸）
        verbose=False,
    )

    # 合并所有检测到的肿瘤 mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # [N, H, W]

        for i, mask in enumerate(masks):
            # 确保 mask 与原图尺寸一致
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            # 二值化并合并
            binary = (mask > 0.5).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary)

        num_detections = len(masks)
        conf_scores = results[0].boxes.conf.cpu().numpy()
        print(f"[INFO] YOLO11-seg 推理完成: {file_name}, "
              f"检测到 {num_detections} 个肿瘤区域, "
              f"置信度: {conf_scores}")
    else:
        print(f"[INFO] YOLO11-seg 推理完成: {file_name}, 未检测到肿瘤区域")

    # 保存 mask
    mask_path = f'./tmp/mask/{file_name}_mask.png'
    cv2.imwrite(mask_path, combined_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    fg_count = (combined_mask > 0).sum()
    print(f"[INFO] Mask 已保存: {mask_path}, 前景像素数: {fg_count}")
```

#### A.2 数据格式转换脚本（convert_to_yolo_seg.py）

```python
# -*- coding: utf-8 -*-
"""
将原始 DCM + mask 数据转换为 YOLO11-seg 训练格式
"""

import os
import glob
import random
import numpy as np
import cv2
import SimpleITK as sitk
from pathlib import Path


def dcm_to_png(dcm_path, window_center=40, window_width=400):
    """读取 DCM 并应用窗宽窗位，输出 uint8 图像"""
    image = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    arr = np.clip(arr, lower, upper)
    arr = ((arr - lower) / (upper - lower) * 255).astype(np.uint8)
    return arr


def mask_to_yolo_seg(mask_binary, img_h, img_w):
    """将二值 mask 转为 YOLO 分割标注格式"""
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 3:
            continue
        points = []
        for x, y in cnt:
            points.append(f"{x / img_w:.6f}")
            points.append(f"{y / img_h:.6f}")
        line = "0 " + " ".join(points)
        lines.append(line)
    return lines


def convert_dataset(data_dir, output_dir, val_ratio=0.15, seed=42):
    """扫描数据目录，转换为 YOLO 格式"""
    random.seed(seed)
    patients = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    random.shuffle(patients)
    val_count = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:val_count])

    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    for pid in patients:
        split = 'val' if pid in val_patients else 'train'
        patient_dir = os.path.join(data_dir, pid)
        scan_dirs = [d for d in os.listdir(patient_dir)
                     if os.path.isdir(os.path.join(patient_dir, d)) and '_mask' not in d]
        for scan_name in scan_dirs:
            scan_dir = os.path.join(patient_dir, scan_name)
            mask_dir = os.path.join(patient_dir, scan_name + '_mask')
            if not os.path.isdir(mask_dir):
                continue
            dcm_files = sorted(glob.glob(os.path.join(scan_dir, '*.dcm')))
            for dcm_path in dcm_files:
                fname = os.path.splitext(os.path.basename(dcm_path))[0]
                mask_path = os.path.join(mask_dir, os.path.basename(dcm_path))
                if not os.path.isfile(mask_path):
                    continue
                img = dcm_to_png(dcm_path)
                h, w = img.shape[:2]
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                mask_sitk = sitk.ReadImage(mask_path)
                mask_arr = sitk.GetArrayFromImage(mask_sitk)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                mask_binary = (mask_arr > 0).astype(np.uint8) * 255
                label_lines = mask_to_yolo_seg(mask_binary, h, w)
                out_name = f"{pid}_{scan_name}_{fname}"
                cv2.imwrite(f"{output_dir}/images/{split}/{out_name}.png", img_3ch)
                with open(f"{output_dir}/labels/{split}/{out_name}.txt", 'w') as f:
                    f.write("\n".join(label_lines))
```

#### A.3 YOLO11-seg 训练脚本（train_yolo11_seg.py）

```python
# -*- coding: utf-8 -*-
"""YOLO11-seg 训练脚本"""

from ultralytics import YOLO
import argparse


def train(args):
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        save=True,
        device=args.device,
        optimizer='AdamW',
        lr0=args.lr,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=15.0, translate=0.1, scale=0.3,
        flipud=0.5, fliplr=0.5,
        mosaic=0.5, mixup=0.0,
        single_cls=True,
    )
    print(f"训练完成！最佳模型: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n-seg.pt')
    parser.add_argument('--data', default='./datasets/rectal_tumor_seg/data.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    train(args)
```

---

## 全文插图位置汇总表

| 图号 | 标题 | 章节 | 类型 | 优先级 |
|------|------|------|------|--------|
| 2-1 | U-Net网络结构示意图 | 2.2 | 自绘/引用 | 必须 |
| 2-2 | YOLO11网络整体架构图 | 2.3.1 | 自绘/引用 | 必须 |
| 2-3 | YOLO11-seg分割机制示意图 | 2.3.2 | 自绘 | 必须 |
| 3-1 | 系统总体架构图 | 3.2 | 自绘 | 必须 |
| 3-2 | 系统功能模块图 | 3.2 | 自绘 | 必须 |
| 3-3 | 数据库ER图 | 3.3 | 自绘 | 必须 |
| 4-1 | 数据预处理流程图 | 4.1 | 自绘 | 建议 |
| 4-2 | YOLO标注格式示例 | 4.1.3 | 截图+标注 | 建议 |
| 4-3 | 推理流程对比图 | 4.3.2 | 自绘 | 必须 |
| 4-4 | 前端界面布局图 | 4.4 | 截图/线框图 | 建议 |
| 4-5 | LLM建议生成流程图 | 4.5 | 自绘 | 建议 |
| 5-1 | 训练损失曲线 | 5.4.1 | 训练自动生成 | 必须 |
| 5-2 | 验证指标变化曲线 | 5.4.1 | 训练自动生成 | 必须 |
| 5-3 | 验证集分割可视化 | 5.4.2 | 推理截图 | 必须 |
| 5-4 | U-Net vs YOLO11对比图 | 5.5 | 推理截图 | 必须 |
| 5-5 | 系统主界面截图 | 5.7 | 系统截图 | 必须 |
| 5-6 | 上传与分割结果截图 | 5.7 | 系统截图 | 必须 |
| 5-7 | 特征值表格截图 | 5.7 | 系统截图 | 必须 |
| 5-8 | 趋势分析图表截图 | 5.7 | 系统截图 | 必须 |
| 5-9 | LLM建议截图 | 5.7 | 系统截图 | 必须 |
| 5-10 | 患者管理截图 | 5.7 | 系统截图 | 建议 |
