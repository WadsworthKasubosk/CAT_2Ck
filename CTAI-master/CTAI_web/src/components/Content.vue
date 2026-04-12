<template>
    <div class="content-wrapper">
        <!-- 导航栏（在 #Content flex 容器外面，不影响原有布局） -->
        <div class="content-top-nav" v-if="hasRouter">
            <el-button size="small" icon="el-icon-arrow-left" @click="goBack">返回列表</el-button>
            <el-button size="small" type="success" icon="el-icon-data-analysis" @click="goToTrend" :disabled="!currentPatientId">趋势分析</el-button>
            <span class="nav-patient-name" v-if="patient">
                <i class="el-icon-user"></i> {{ patient['姓名'] }}
                <el-tag size="mini" style="margin-left:6px;">{{ patient['性别'] }}</el-tag>
                <el-tag size="mini" type="info" style="margin-left:4px;">{{ patient['年龄'] }}岁</el-tag>
            </span>
        </div>
    <div id="Content">
        <el-dialog
            id="hello"
            title="肿瘤辅助诊断系统使用须知"
            :visible.sync="centerDialogVisible"
            width="65%"
            :before-close="handleClose"
        >
            <el-steps :active="5" finish-status="process ">
                <el-step title="步骤1" style="width:280px;padding-left: 50px">
                    <template slot="description">
                        <p style="font-size: 16px">下载测试CT文件文件</p>
                        <br>
                        <br>
                    </template>
                </el-step>
                <el-step title="步骤2" style="width:260px;margin-left:-5px;">
                    <template slot="description">
                        <p>上传CT图像至服务器</p>
                        <p>使用训练的模型预测肿瘤区域</p>
                        <p>并返回肿瘤区域特征</p>
                    </template>
                </el-step>
                <el-step title="步骤3" style="width:260px;margin-left:-5px;">
                    <template slot="description">
                        <div>
                            <p>根据预测的肿瘤区域和特征</p>
                            <p>进行辅助诊断</p>
                            <br>
                        </div>
                    </template>
                </el-step>
            </el-steps>
            <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="welcome">下载测试CT图像</el-button>
      </span>
        </el-dialog>
        <el-dialog
            title="AI预测中"
            :visible.sync="dialogTableVisible"
            :show-close="false"
            :close-on-press-escape="false"
            :append-to-body="true"
            :close-on-click-modal="false"
            :center="true"
        >
            <el-progress :percentage="percentage"></el-progress>
            <span slot="footer" class="dialog-footer">非GPU学生服务器性能有限，请耐心等待约一分钟</span>
        </el-dialog>

        <!-- 添加患者弹窗 -->
        <el-dialog title="添加患者" :visible.sync="addPatientDialogVisible" width="400px">
            <el-form :model="newPatientForm" label-width="70px" size="small">
                <el-form-item label="姓名">
                    <el-input v-model="newPatientForm.name" placeholder="请输入姓名"></el-input>
                </el-form-item>
                <el-form-item label="性别">
                    <el-radio-group v-model="newPatientForm.gender">
                        <el-radio label="男">男</el-radio>
                        <el-radio label="女">女</el-radio>
                    </el-radio-group>
                </el-form-item>
                <el-form-item label="年龄">
                    <el-input-number v-model="newPatientForm.age" :min="0" :max="150" style="width:100%;"></el-input-number>
                </el-form-item>
                <el-form-item label="电话">
                    <el-input v-model="newPatientForm.phone" placeholder="请输入电话"></el-input>
                </el-form-item>
            </el-form>
            <span slot="footer" class="dialog-footer">
                <el-button size="small" @click="addPatientDialogVisible = false">取消</el-button>
                <el-button type="primary" size="small" @click="submitAddPatient">确 定</el-button>
            </span>
        </el-dialog>

        <div id="aside">
            <!-- 查看病人信息 -->
            <el-card class="box-card" style="width:250px;min-height:480px">
                <div slot="header" class="clearfix">
                    <span>病人信息</span>
                </div>
                <el-select
                    v-model="currentPatientId"
                    placeholder="请选择患者"
                    style="width:100%;margin-bottom:15px;"
                    @change="onPatientChange"
                >
                    <el-option
                        v-for="p in patientList"
                        :key="p.id"
                        :label="p.name + ' (' + p.gender + ', ' + p.age + '岁)'"
                        :value="p.id"
                    ></el-option>
                </el-select>
                <div style="margin-bottom:10px;text-align:center;">
                    <el-button type="primary" size="mini" icon="el-icon-plus" @click="addPatientDialogVisible = true">添加患者</el-button>
                    <el-button type="danger" size="mini" icon="el-icon-delete" @click="deleteCurrentPatient" :disabled="!currentPatientId">删除</el-button>
                    <el-button size="mini" icon="el-icon-refresh" @click="resetPatientList">重置</el-button>
                </div>
                <div v-if="patient">
                    <div v-for="(value,name) in patient" :key="name" class="text item">
                        <h3 style="font-weight:normal;">{{name}}:{{value}}</h3>
                    </div>
                </div>
                <div v-else style="color:#999;text-align:center;padding:20px;">请先选择患者
                </div>
            </el-card>

            <!-- 步骤条：下载 上传 -->
            <el-card
                class="box-card"
                body-style="padding: 15px 5px 15px 10px"
                style="width:250px;height:500px;margin-top:50px;"
            >
                <div slot="header" class="clearfix" style="text-align:center;">
                    <span class="steps" style="letter-spacing: 7px;">诊断测试步骤</span>
                </div>
                <div style="height: 600px;" class="step_1">
                    <el-steps direction="vertical" :active="active" finish-status="success">
                        <el-step style="height: 120px;" title="步骤 1">
                            <template slot="description" style="font-size: 10px!important;">
                                下载测试CT文件
                                <!-- 下载文件 -->
                                <el-button
                                    type="primary"
                                    icon="el-icon-download"
                                    @click="downTemplate"
                                    class="download_bt"
                                >下载
                                </el-button>
                            </template>
                        </el-step>
                        <el-step style="height: 150px;" title="步骤 2">
                            <template slot="description">
                                <!-- 上传文件 -->
                                上传CT图像至服务器，使用训练的模型预测肿瘤区域并返回肿瘤区域特征
                                <el-button type="primary" icon="el-icon-upload" class="download_bt">上传</el-button>
                                <input class="file" name="file" type="file" @change="update">
                            </template>
                        </el-step>

                        <!-- 获得图像 -->
                        <el-step title="获得图像及特征" style="height: 200px;">
                            <template slot="description"></template>
                        </el-step>
                    </el-steps>
                </div>
            </el-card>
        </div>
        <!-- 上传返回信息部分：原CT图部分  标出肿瘤的CT图像 图像特征-->
        <div id="CT">
            <!-- CT图像 -->
            <div id="CT_image">
                <!-- 原CT图 -->
                <el-card
                    id="CT_image_1"
                    class="box-card"
                    style="border-radius: 8px;width:800px;height:360px;margin-bottom:-30px;"
                >
                    <div class="demo-image__preview1">
                        <div
                            v-loading="loading"
                            element-loading-text="上传图片中"
                            element-loading-spinner="el-icon-loading"
                        >
                            <el-image
                                :src="url_1"
                                class="image_1"
                                :preview-src-list="srcList"
                                style="border-radius: 3px 3px 0 0"
                            >
                                <div slot="error">
                                    <div slot="placeholder" class="error">
                                        <el-button
                                            v-show="showbutton"
                                            type="primary"
                                            icon="el-icon-upload"
                                            class="download_bt"
                                            v-on:click="true_upload"
                                        >
                                            上传dcm文件
                                            <input
                                                ref="upload"
                                                style="display: none"
                                                name="file"
                                                type="file"
                                                @change="update"
                                            >
                                        </el-button>
                                    </div>
                                </div>
                            </el-image>
                        </div>
                        <!-- 原CT图文字 -->
                        <div class="img_info_1" style="border-radius:0 0 5px 5px;">
                            <span style="color:white;letter-spacing:6px;">原CT图像</span>
                        </div>
                    </div>
                    <!-- 标出肿瘤的CT图像 -->
                    <div class="demo-image__preview2">
                        <div
                            v-loading="loading"
                            element-loading-text="处理中,请耐心等待"
                            element-loading-spinner="el-icon-loading"
                        >
                            <el-image
                                :src="url_2"
                                class="image_1"
                                :preview-src-list="srcList1"
                                style="border-radius: 3px 3px 0 0;"
                            >
                                <div slot="error">
                                    <div slot="placeholder" class="error">{{wait_return}}</div>
                                </div>
                            </el-image>
                        </div>
                        <!-- 标出肿瘤的CT图像文字 -->
                        <div class="img_info_1" style="border-radius: 0 0 5px 5px;">
                            <span style="color:white;letter-spacing:4px;">标出肿瘤的CT图像</span>
                        </div>
                    </div>
                </el-card>
            </div>


            <!-- 分割线 -->

            <!-- 图像特征部分 -->
            <div id="info_patient">
                <!-- 卡片放置表格 -->
                <el-card style="border-radius: 8px;">
                    <div slot="header" class="clearfix">
                        <span>肿瘤区域特征值</span>
                        <el-button
                            style="margin-left: 35px"
                            v-show="!showbutton"
                            type="primary"
                            icon="el-icon-upload"
                            class="download_bt"
                            v-on:click="true_upload2"
                        >
                            重新选择图像
                            <input
                                ref="upload2"
                                style="display: none"
                                name="file"
                                type="file"
                                @change="update"
                            >
                        </el-button>
                    </div>


                    <el-tabs v-model="activeName" @tab-click="handleClick">
                        <el-tab-pane label="肿瘤区域特征值" name="first">
                            <!-- 表格存放特征值 -->
                            <el-table
                                :data="feature_list"
                                height="390"
                                border
                                style="width:750px;text-align:center;"
                                v-loading="loading"
                                element-loading-text="数据正在处理中，请耐心等待"
                                element-loading-spinner="el-icon-loading"
                                lazy
                            >
                                <el-table-column label="Feature" width="140px">
                                    <template slot-scope="scope">
                                        <span>{{scope.row[2]}}</span>
                                    </template>
                                </el-table-column>
                                <!-- 特征名 -->
                                <el-table-column label="特征名" width="140px">
                                    <template slot-scope="scope">
                                        <span>{{scope.row[0]}}</span>
                                    </template>
                                </el-table-column>

                                <!-- 特征值 -->
                                <el-table-column label="特征值" width="130px">
                                    <template slot-scope="scope">
                                        <span style="font-weight:bold;color:#409EFF;">{{scope.row[1]}}</span>
                                    </template>
                                </el-table-column>

                                <!-- 参考区间 -->
                                <el-table-column label="参考区间" width="160px">
                                    <template slot-scope="scope">
                                        <span style="color:#909399;">{{ getFeatureRange(scope.row[2]) }}</span>
                                    </template>
                                </el-table-column>

                                <!-- 说明 -->
                                <el-table-column label="说明" width="180px">
                                    <template slot-scope="scope">
                                        <span style="color:#606266;font-size:12px;">{{ getFeatureDescription(scope.row[2]) }}</span>
                                    </template>
                                </el-table-column>
                            </el-table>
                        </el-tab-pane>
                        <el-tab-pane label="面积对比" name="second" style="width:750px;height:390px;">
                            <div id="areaCompare">
                                <el-table

                                    :data="feature_list"
                                    height="390"
                                    border
                                    style="width:750px;text-align:center;"
                                    v-loading="loading"
                                    element-loading-text="数据正在处理中，请耐心等待"
                                    element-loading-spinner="el-icon-loading"
                                >
                                    <el-table-column label="Feature" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[2]}}</span>
                                        </template>
                                    </el-table-column>
                                    <!-- 特征名 -->
                                    <el-table-column label="特征名" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[0]}}</span>
                                        </template>
                                    </el-table-column>

                                    <!-- 特征值 -->
                                    <el-table-column label="特征值" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[1]}}</span>
                                        </template>
                                    </el-table-column>
                                </el-table>
                            </div>
                            <div id="area" style="width: 750px;height:400px;margin-bottom:20px;"></div>
                        </el-tab-pane>
                        <el-tab-pane label="周长对比" name="third" style="width:750px;height:390px;">
                            <div id="perimeterCompare">
                                <el-table

                                    :data="feature_list"
                                    height="390"
                                    border
                                    style="width:750px;text-align:center;"
                                    v-loading="loading"
                                    element-loading-text="数据正在处理中，请耐心等待"
                                    element-loading-spinner="el-icon-loading"
                                >
                                    <el-table-column label="Feature" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[2]}}</span>
                                        </template>
                                    </el-table-column>
                                    <!-- 特征名 -->
                                    <el-table-column label="特征名" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[0]}}</span>
                                        </template>
                                    </el-table-column>

                                    <!-- 特征值 -->
                                    <el-table-column label="特征值" width="250px">
                                        <template slot-scope="scope">
                                            <span>{{scope.row[1]}}</span>
                                        </template>
                                    </el-table-column>
                                </el-table>
                            </div>

                            <div id="perimeter" style="width: 750px;height:400px;margin-bottom:20px;"></div>
                        </el-tab-pane>
                        <el-tab-pane label="诊断记录" name="fourth">
                            <el-table
                                :data="diagnosisRecords"
                                border
                                style="width:750px;"
                                v-loading="recordsLoading"
                                element-loading-text="加载中"
                            >
                                <el-table-column prop="id" label="序号" width="80px"></el-table-column>
                                <el-table-column prop="dcm_filename" label="文件名" width="200px"></el-table-column>
                                <el-table-column prop="status" label="状态" width="120px">
                                    <template slot-scope="scope">
                                        <el-tag :type="scope.row.status === 'completed' ? 'success' : 'danger'" size="small">
                                            {{scope.row.status === 'completed' ? '推理成功' : '推理失败'}}
                                        </el-tag>
                                    </template>
                                </el-table-column>
                                <el-table-column prop="created_at" label="诊断时间" width="200px"></el-table-column>
                                <el-table-column label="操作" width="120px">
                                    <template slot-scope="scope">
                                        <el-button type="text" size="small" @click="viewRecordDetail(scope.row.id)">查看详情</el-button>
                                    </template>
                                </el-table-column>
                            </el-table>
                            <!-- 详情卡片 -->
                            <el-card v-if="selectedRecord" style="margin-top:15px;width:750px;">
                                <div slot="header" class="clearfix">
                                    <span>诊断详情 #{{selectedRecord.id}}</span>
                                    <el-button style="float:right;" type="text" @click="selectedRecord=null">关闭</el-button>
                                </div>
                                <el-row :gutter="20">
                                    <el-col :span="12">
                                        <p><b>文件名：</b>{{selectedRecord.dcm_filename || '暂无'}}</p>
                                        <p><b>状态：</b>{{selectedRecord.status === 'completed' ? '推理成功' : '推理失败'}}</p>
                                        <p><b>诊断时间：</b>{{selectedRecord.created_at}}</p>
                                        <p><b>面积：</b>{{selectedRecord.area != null ? selectedRecord.area : '暂无'}}</p>
                                        <p><b>周长：</b>{{selectedRecord.perimeter != null ? selectedRecord.perimeter : '暂无'}}</p>
                                    </el-col>
                                    <el-col :span="12">
                                        <p><b>备注：</b>{{selectedRecord.doctor_note || '暂无'}}</p>
                                    </el-col>
                                </el-row>
                                <!-- LLM 辅助建议区 -->
                                <el-divider></el-divider>
                                <div style="margin-bottom:15px;">
                                    <div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
                                        <span style="font-size:13px;color:#606266;">AI 模型：</span>
                                        <el-select v-model="selectedProvider" placeholder="选择模型提供商" size="small" style="width:160px;" @change="onProviderChange">
                                            <el-option v-for="p in llmProviders" :key="p.id" :label="p.name + (p.available ? '' : ' (未配置)')" :value="p.id" :disabled="!p.available"></el-option>
                                        </el-select>
                                        <el-select v-model="selectedModel" placeholder="选择模型" size="small" style="width:180px;">
                                            <el-option v-for="m in currentProviderModels" :key="m" :label="m" :value="m"></el-option>
                                        </el-select>
                                        <el-button type="primary" size="small" :loading="llmLoading" @click="generateLlmAdvice(selectedRecord.id)">
                                            🤖 生成辅助建议
                                        </el-button>
                                    </div>
                                    <div v-if="llmProviders.length === 0" style="text-align:center;color:#909399;font-size:12px;">
                                        正在加载可用模型...
                                    </div>
                                </div>
                                <el-card v-if="llmAdvice" shadow="never" style="background:#f5f7fa;">
                                    <div slot="header" style="font-size:13px;color:#606266;">
                                        <span>📝 AI 辅助建议</span>
                                        <span v-if="llmAdvice.model_name" style="float:right;color:#909399;font-size:12px;">模型: {{llmAdvice.model_name}}</span>
                                    </div>
                                    <p style="white-space:pre-wrap;line-height:1.8;font-size:14px;">{{llmAdvice.advice}}</p>
                                    <p v-if="llmAdvice.disclaimer" style="color:#E6A23C;font-size:12px;margin-top:10px;border-top:1px solid #eee;padding-top:8px;">
                                        ⚠️ {{llmAdvice.disclaimer}}
                                    </p>
                                </el-card>
                            </el-card>
                        </el-tab-pane>
                        <el-tab-pane label="多项指标对比" name="multiCompare">
                            <div id="multiCompareChart" style="width:720px;height:420px;"></div>
                        </el-tab-pane>
                        <el-tab-pane label="历史图像对比" name="historyImages">
                            <div v-if="historyImageRecords.length === 0" style="text-align:center;color:#999;padding:40px;">
                                暂无历史图像，请先上传 CT 图像进行诊断
                            </div>
                            <div v-else style="display:flex;flex-wrap:wrap;gap:15px;">
                                <el-card v-for="rec in historyImageRecords" :key="rec.id" shadow="hover" style="width:220px;">
                                    <el-image :src="rec.draw_url || rec.image_url" style="width:200px;height:200px;" fit="contain">
                                        <div slot="error" style="text-align:center;padding:40px;color:#999;">图像不可用</div>
                                    </el-image>
                                    <div style="padding:8px 0;font-size:12px;color:#606266;">
                                        <p><b>诊断 #{{rec.id}}</b></p>
                                        <p>时间: {{rec.created_at}}</p>
                                        <p>面积: {{rec.area != null ? rec.area : '-'}}</p>
                                        <p>周长: {{rec.perimeter != null ? rec.perimeter : '-'}}</p>
                                    </div>
                                </el-card>
                            </div>
                        </el-tab-pane>
                        <el-tab-pane label="时序预测" name="fifth">
                            <div v-loading="forecastLoading" element-loading-text="模型预测中...">
                                <!-- 预测步数选择 -->
                                <div style="margin-bottom:15px;display:flex;align-items:center;">
                                    <span style="margin-right:10px;font-size:14px;">预测步数：</span>
                                    <el-input-number v-model="forecastSteps" :min="1" :max="10" :step="1" size="small" style="width:120px;"></el-input-number>
                                    <el-button type="primary" size="small" style="margin-left:15px;" @click="fetchForecast" :loading="forecastLoading">
                                        🔮 开始预测
                                    </el-button>
                                </div>

                                <!-- 无数据提示 -->
                                <el-alert v-if="forecastData && !forecastData.lstm_success && Object.keys(forecastData.features || {}).length === 0"
                                    title="历史数据不足"
                                    description="至少需要 2 条诊断记录才能进行时序预测，请先上传更多 CT 图像。"
                                    type="warning" show-icon :closable="false" style="margin-bottom:15px;">
                                </el-alert>

                                <!-- 模型状态提示 -->
                                <div v-if="forecastData && Object.keys(forecastData.features || {}).length > 0" style="margin-bottom:10px;">
                                    <el-tag :type="forecastData.lstm_success ? 'success' : 'info'" size="small" style="margin-right:8px;">
                                        LSTM: {{ forecastData.lstm_success ? '✓ 预测成功' : '✗ ' + (forecastData.lstm_msg || '未启用') }}
                                    </el-tag>
                                    <el-tag type="success" size="small">线性回归: ✓ 已计算</el-tag>
                                </div>

                                <!-- 各指标预测图表 -->
                                <div v-if="forecastData && Object.keys(forecastData.features || {}).length > 0">
                                    <div v-for="(feat, key) in forecastData.features" :key="key" style="margin-bottom:25px;">
                                        <el-card shadow="hover">
                                            <div slot="header" class="clearfix" style="display:flex;align-items:center;justify-content:space-between;">
                                                <span style="font-weight:bold;font-size:15px;">{{ feat.name }}</span>
                                                <div>
                                                    <el-tag :type="feat.trend === '增大' ? 'danger' : (feat.trend === '减小' ? 'success' : 'info')" size="mini" style="margin-right:5px;">
                                                        趋势: {{ feat.trend }}
                                                    </el-tag>
                                                    <el-tag :type="feat.confidence === '高' ? 'success' : (feat.confidence === '中' ? 'warning' : 'info')" size="mini">
                                                        置信度: {{ feat.confidence }}
                                                    </el-tag>
                                                </div>
                                            </div>
                                            <!-- ECharts 图表容器 -->
                                            <div :id="'forecast_chart_' + key" style="width:700px;height:300px;"></div>
                                            <!-- 预测数据表格 -->
                                            <el-row :gutter="20" style="margin-top:10px;">
                                                <el-col :span="8">
                                                    <div style="font-size:12px;color:#909399;">
                                                        <p><b>线性回归</b></p>
                                                        <p>R²: {{ feat.regression ? feat.regression.r_squared : '-' }}</p>
                                                        <p>斜率: {{ feat.regression ? feat.regression.slope : '-' }}</p>
                                                        <p>预测值: {{ feat.regression && feat.regression.predicted ? feat.regression.predicted.join(', ') : '-' }}</p>
                                                    </div>
                                                </el-col>
                                                <el-col :span="8">
                                                    <div style="font-size:12px;color:#909399;">
                                                        <p><b>LSTM 预测</b></p>
                                                        <p>预测值: {{ feat.lstm && feat.lstm.predicted ? feat.lstm.predicted.join(', ') : '-' }}</p>
                                                    </div>
                                                </el-col>
                                                <el-col :span="8">
                                                    <div style="font-size:12px;color:#606266;">
                                                        <p><b>综合预测 (加权融合)</b></p>
                                                        <p style="font-size:14px;color:#409EFF;">{{ feat.ensemble_predicted ? feat.ensemble_predicted.join(', ') : '-' }}</p>
                                                    </div>
                                                </el-col>
                                            </el-row>
                                        </el-card>
                                    </div>
                                </div>
                            </div>
                        </el-tab-pane>
                    </el-tabs>
                </el-card>
            </div>
        </div>
    </div>
    </div>
</template>

<script>
    import axios from "axios";

    export default {
        name: "Content",
        data() {
            return {
                // server_url:'http://58.87.66.50:5003',
                server_url:'http://127.0.0.1:5003',
                perimeter_picture_data: 0,
                area_picture_data: 0,
                // 趋势分析缓存数据（从后端接口获取）
                trendData: null,
                forecastData: null,
                forecastLoading: false,
                forecastSteps: 3,
                diagnosisRecords: [],
                recordsLoading: false,
                selectedRecord: null,
                llmAdvice: null,
                llmLoading: false,
                llmProviders: [],
                selectedProvider: '',
                selectedModel: '',
                addPatientDialogVisible: false,
                newPatientForm: { name: '', gender: '男', age: 50, phone: '' },
                historyImageRecords: [],
                currentPatientId: null, // 由 fetchPatientList 自动选择
                activeName: "first",
                active: 0,
                centerDialogVisible: true,
                url_1: "",
                url_2: "",
                textarea: "",
                srcList: [],
                srcList1: [],
                feature_list: [],
                feature_list_1: [],
                feat_list: [],
                url: "",
                visible: false,
                activeName: "second",
                wait_return: "等待上传",
                wait_upload: "等待上传",
                loading: false,
                table: false,
                isNav: false,
                showbutton: true,
                percentage: 0,
                fullscreenLoading: false,
                opacitys: {
                    opacity: 0
                },
                dialogTableVisible: false,
                patient: null,
                patientList: [],
                // 特征值参考区间
                feature_ranges: {
                    'area': '800~1500 mm²',
                    'perimeter': '100~250 mm',
                    'mean': '80~130',
                    'std': '15~50',
                    'focus_x': '-',
                    'focus_y': '-',
                    'ellipse': '0.5~0.9',
                    'max_length': '-',
                    'min_length': '-',
                    'orientation': '0~180°'
                },
                // 特征值说明
                feature_descriptions: {
                    'area': '肿瘤区域面积大小',
                    'perimeter': '肿瘤边界周长',
                    'mean': '区域灰度均值，反映密度',
                    'std': '灰度方差，反映均匀性',
                    'focus_x': '质心横坐标',
                    'focus_y': '质心纵坐标',
                    'ellipse': '似圆度，越接近1越圆',
                    'max_length': '最大直径',
                    'min_length': '最小直径',
                    'orientation': '主轴方向角'
                },
            };
        },
        computed: {
            hasRouter() {
                return this.$route && this.$route.params && this.$route.params.id;
            },
            currentProviderModels() {
                var provider = this.llmProviders.find(p => p.id === this.selectedProvider);
                return provider ? provider.models : [];
            }
        },
        created: function () {
            document.title = '肿瘤辅助诊断系统';
            this.fetchLlmProviders();
            // 如果通过路由进入（从患者列表页跳转），自动选中患者并跳过欢迎弹窗
            if (this.$route && this.$route.params && this.$route.params.id) {
                this.centerDialogVisible = false;
                this.currentPatientId = parseInt(this.$route.params.id);
            }
        },
        methods: {
            true_upload() {
                this.$refs.upload.click();
            },
            true_upload2() {
                this.$refs.upload2.click();
            },
            handleClose(done) {
                this.$confirm("确认关闭？")
                    .then(_ => {
                        done();
                    })
                    .catch(_ => {
                    });
            },
            next() {
                this.active++;
            },
            // 获得目标文件
            getObjectURL(file) {
                var url = null;
                if (window.createObjcectURL != undefined) {
                    url = window.createOjcectURL(file);
                } else if (window.URL != undefined) {
                    url = window.URL.createObjectURL(file);
                } else if (window.webkitURL != undefined) {
                    url = window.webkitURL.createObjectURL(file);
                }
                return url;
            },
            // 点击切换
            handleClick(tab, event) {
                if (tab.name == "second") {
                    this.drawChart();
                    var myChart_area = this.$echarts.init(document.getElementById('area'));
                    var dates = (this.trendData && this.trendData.dates) || [];
                    var areaValues = (this.trendData && this.trendData.area && this.trendData.area.values) || [];
                    if (dates.length === 0) {
                        dates = ['暂无数据'];
                        areaValues = [0];
                    }
                    myChart_area.setOption({
                        xAxis: {
                            type: "category",
                            data: dates
                        },
                        yAxis: {
                            type: "value",
                            name: "面积"
                        },
                        areaStyle: {},
                        legend: {
                            data: [""]
                        },
                        series: [
                            {
                                name: "面积",
                                type: "line",
                                data: areaValues
                            }
                        ]
                    });
                } else if (tab.name == "third") {
                    this.drawChart();
                    var myChart_perimeter = this.$echarts.init(document.getElementById('perimeter'));
                    var dates = (this.trendData && this.trendData.dates) || [];
                    var periValues = (this.trendData && this.trendData.perimeter && this.trendData.perimeter.values) || [];
                    if (dates.length === 0) {
                        dates = ['暂无数据'];
                        periValues = [0];
                    }
                    myChart_perimeter.setOption({
                        xAxis: {
                            type: "category",
                            data: dates
                        },
                        yAxis: {
                            type: "value",
                            name: "周长"
                        },
                        areaStyle: {},
                        series: [
                            {
                                name: "周长",
                                type: "line",
                                data: periValues
                            }
                        ]
                    });
                } else if (tab.name == "fourth") {
                    this.fetchDiagnosisRecords();
                } else if (tab.name == "multiCompare") {
                    this.$nextTick(() => { this.drawMultiCompareChart(); });
                } else if (tab.name == "historyImages") {
                    this.fetchHistoryImages();
                } else if (tab.name == "fifth") {
                    if (!this.forecastData) {
                        this.fetchForecast();
                    } else {
                        this.$nextTick(() => { this.drawForecastCharts(); });
                    }
                }
            },
            // 上传dcm文件
            update(e) {
                this.percentage = 0;
                this.dialogTableVisible = true;
                this.url_1 = "";
                this.url_2 = "";
                this.srcList = [];
                this.srcList1 = [];
                this.wait_return = "";
                this.wait_upload = "";
                this.feature_list = [];
                let myChart_area = this.$echarts.init(document.getElementById("area"));
                myChart_area.setOption({
                    series: [
                        {
                            data: [""]
                        }
                    ]
                });
                this.feat_list = [];
                this.fullscreenLoading = true;
                this.loading = true;
                this.showbutton = false;
                let file = e.target.files[0];
                this.url_1 = this.$options.methods.getObjectURL(file);
                let param = new FormData(); //创建form对象
                param.append("file", file, file.name); //通过append向form对象添加数据
                param.append("patient_id", this.currentPatientId);
                // console.log(param.get("file")); //FormData私有类对象，访问不到，可以通过get判断值是否传进去
                //todo aaaa
                var timer = setInterval(() => {
                    this.myFunc();
                }, 30);
                let config = {
                    headers: {"Content-Type": "multipart/form-data"}
                }; //添加请求头
                axios
                    .post(this.server_url+"/upload", param, config)
                    .then(response => {
                        this.percentage = 100;
                        clearInterval(timer);
                        this.fullscreenLoading = false;
                        this.loading = false;

                        // 推理失败但上传成功的情况
                        if (response.data.status === 0) {
                            this.dialogTableVisible = false;
                            this.percentage = 0;
                            this.$message.warning(response.data.msg || '上传成功，但模型推理失败');
                            console.warn('推理失败:', response.data);
                            this.fetchDiagnosisRecords();
                            return;
                        }

                        this.url_1 = response.data.image_url;
                        this.srcList.push(this.url_1);
                        this.url_2 = response.data.draw_url;
                        this.srcList1.push(this.url_2);

                        this.feat_list = Object.keys(response.data.image_info);

                        for (var i = 0; i < this.feat_list.length; i++) {
                            response.data.image_info[this.feat_list[i]][2] = this.feat_list[i];
                            this.feature_list.push(response.data.image_info[this.feat_list[i]]);
                        }

                        this.feature_list.push(response.data.image_info);
                        this.feature_list_1 = this.feature_list[0];
                        JSON.stringify(response.data.image_info, (key, value) => {
                            console.log(key);
                            console.log(value);
                        });
                        this.dialogTableVisible = false;
                        this.percentage = 0;
                        this.notice1();
                        this.fetchDiagnosisRecords();
                        var areaCompare = document.getElementById("areaCompare");
                        areaCompare.style.display = "none";
                        var areaCompare = document.getElementById("perimeterCompare");
                        areaCompare.style.display = "none";
                        let myChart_area = this.$echarts.init(
                            document.getElementById("area")
                        );
                        let myChart_perimeter = this.$echarts.init(
                            document.getElementById("perimeter")
                        );
                        this.perimeter_picture_data = parseInt(response.data.image_info["perimeter"][1]);
                        this.area_picture_data = parseInt(response.data.image_info["area"][1]);

                        // 上传成功后重新拉取趋势数据
                        this.fetchTrendData().then(() => {
                            var dates = (this.trendData && this.trendData.dates) || [];
                            var areaValues = (this.trendData && this.trendData.area && this.trendData.area.values) || [];
                            var periValues = (this.trendData && this.trendData.perimeter && this.trendData.perimeter.values) || [];
                            if (dates.length === 0) {
                                dates = ['暂无数据'];
                                areaValues = [0];
                                periValues = [0];
                            }
                            myChart_area.setOption({
                                xAxis: { type: "category", data: dates },
                                yAxis: { type: "value", name: "面积" },
                                areaStyle: {},
                                series: [{ name: "面积", type: "line", data: areaValues }]
                            });
                            myChart_perimeter.setOption({
                                xAxis: { type: "category", data: dates },
                                yAxis: { type: "value", name: "周长" },
                                areaStyle: {},
                                series: [{ name: "周长", type: "line", data: periValues }]
                            });
                        });
                    }).catch(err => {
                        this.percentage = 0;
                        this.fullscreenLoading = false;
                        this.loading = false;
                        clearInterval(timer);
                        this.$message.error('上传请求失败，请检查网络或后端服务');
                        console.error('上传异常:', err);
                    });
            },
            // 下载 点击按钮 从远程接口获取文件
            downTemplate() {
                axios({
                    method: "get",
                    url:
                        "https://cso1-1254043908.cos.ap-beijing.myqcloud.com/ct/testfile.7z",
                    responseType: "blob"
                }).then(res => {
                    this.downloads(res.data, res.headers.filename);

                    if (res.status === 200) {
                        this.$message({
                            message: "下载成功",
                            type: "success"
                        });
                        if (this.active == 0) {
                            this.next();
                        }
                    } else {
                        this.$message({
                            showClose: true,
                            message: "下载失败，请重试",
                            type: "error"
                        });
                    }
                });
            },
            myFunc() {
                if (this.percentage + 33 < 99) {
                    this.percentage = this.percentage + 33;
                    console.log(this.percentage);
                } else {
                    this.percentage = 99;
                }
            },
            drawChart() {
                // 基于准备好的dom，初始化echarts实例
                let myChart_area = this.$echarts.init(document.getElementById("area"));
                let myChart_perimeter = this.$echarts.init(
                    document.getElementById("perimeter")
                );

                // 从缓存的趋势数据中取值，无数据时显示空框
                var dates = (this.trendData && this.trendData.dates) || [];
                var areaValues = (this.trendData && this.trendData.area && this.trendData.area.values) || [];
                var periValues = (this.trendData && this.trendData.perimeter && this.trendData.perimeter.values) || [];

                // 指定图表的配置项和数据
                myChart_area.setOption({
                    title: {
                        text: "肿瘤面积变化",
                        subtext: "Tumor Area Change",
                        left: "center"
                    },
                    legend: {
                        data: [""]
                    },
                    tooltip: {},

                    grid: {
                        x: 50,
                        y: 55,
                        x2: 50,
                        y2: 60,
                        borderWidth: 1
                    },

                    toolbox: {
                        show: true,
                        feature: {
                            dataZoom: {
                                yAxisIndex: "none"
                            },
                            dataView: {readOnly: false},
                            magicType: {type: ["line", "bar"]},
                            restore: {},
                            saveAsImage: {}
                        }
                    },
                    xAxis: {
                        type: "category",
                        boundaryGap: false,
                        data: dates.length > 0 ? dates : ["暂无数据"],
                        name: "诊断时间",
                        nameLocation: "middle",
                        nameTextStyle: {
                            padding: 14,
                            fontSize: 14
                        }
                    },
                    yAxis: {
                        type: "value",
                        name: "肿瘤面积",
                        nameTextStyle: {
                            padding: 4,
                            fontSize: 14
                        }
                    },
                    series: [
                        {
                            name: "面积",
                            type: "bar",
                            data: areaValues
                        }
                    ]
                });
                myChart_perimeter.setOption({
                    title: {
                        text: "肿瘤周长变化",
                        subtext: "Tumor Circumference Change",
                        left: "center"
                    },
                    legend: {
                        data: [""]
                    },
                    tooltip: {},

                    grid: {
                        x: 50,
                        y: 55,
                        x2: 50,
                        y2: 60,
                        borderWidth: 1
                    },

                    toolbox: {
                        show: true,
                        feature: {
                            dataZoom: {
                                yAxisIndex: "none"
                            },
                            dataView: {readOnly: false},
                            magicType: {type: ["line", "bar"]},
                            restore: {},
                            saveAsImage: {}
                        }
                    },
                    xAxis: {
                        type: "category",
                        boundaryGap: false,
                        data: dates.length > 0 ? dates : ["暂无数据"],
                        name: "诊断时间",
                        nameLocation: "middle",
                        nameTextStyle: {
                            padding: 14,
                            fontSize: 14
                        }
                    },
                    yAxis: {
                        type: "value",
                        name: "肿瘤周长",
                        nameTextStyle: {
                            padding: 4,
                            fontSize: 14
                        }
                    },
                    series: [
                        {
                            name: "周长",
                            type: "bar",
                            data: periValues
                        }
                    ]
                });
            },
            // 从后端获取趋势数据
            fetchTrendData() {
                return axios.get(this.server_url + '/api/patients/' + this.currentPatientId + '/trend')
                    .then(res => {
                        if (res.data.status === 1 && res.data.data) {
                            this.trendData = res.data.data;
                        }
                    })
                    .catch(err => {
                        console.warn('趋势数据获取失败:', err);
                    });
            },
            // 从后端获取诊断记录列表
            fetchDiagnosisRecords() {
                this.recordsLoading = true;
                this.selectedRecord = null;
                axios.get(this.server_url + '/api/patients/' + this.currentPatientId + '/records')
                    .then(res => {
                        if (res.data.status === 1) {
                            this.diagnosisRecords = res.data.data || [];
                        }
                    })
                    .catch(err => {
                        console.warn('诊断记录获取失败:', err);
                    })
                    .finally(() => {
                        this.recordsLoading = false;
                    });
            },
            // 获取患者列表
            fetchPatientList() {
                axios.get(this.server_url + '/api/patients')
                    .then(res => {
                        if (res.data.status === 1) {
                            this.patientList = res.data.data || [];
                            // 如果有患者，默认选第一个
                            if (this.patientList.length > 0 && !this.currentPatientId) {
                                this.currentPatientId = this.patientList[0].id;
                                this.onPatientChange(this.currentPatientId);
                            } else if (this.currentPatientId) {
                                this.onPatientChange(this.currentPatientId);
                            }
                        }
                    })
                    .catch(err => {
                        console.warn('患者列表获取失败:', err);
                    });
            },
            // 切换患者
            onPatientChange(patientId) {
                this.currentPatientId = patientId;
                // 更新患者信息卡片
                var found = this.patientList.find(p => p.id === patientId);
                if (found) {
                    this.patient = {
                        'ID': found.id,
                        '姓名': found.name,
                        '性别': found.gender,
                        '年龄': found.age,
                        '电话': found.phone,
                        '部位': found.body_part
                    };
                }
                // 刷新趋势和记录
                this.fetchTrendData().then(() => {
                    this.drawChart();
                });
                this.fetchDiagnosisRecords();
            },
            // 添加患者
            submitAddPatient() {
                if (!this.newPatientForm.name || !this.newPatientForm.gender) {
                    this.$message.warning('姓名和性别为必填项');
                    return;
                }
                axios.post(this.server_url + '/api/patients', this.newPatientForm)
                    .then(res => {
                        if (res.data.status === 1) {
                            this.$message.success('添加患者成功');
                            this.addPatientDialogVisible = false;
                            this.newPatientForm = { name: '', gender: '男', age: 50, phone: '' };
                            this.fetchPatientList();
                        } else {
                            this.$message.error(res.data.msg || '添加失败');
                        }
                    })
                    .catch(err => {
                        this.$message.error('添加患者失败');
                        console.warn(err);
                    });
            },
            // 删除当前患者
            deleteCurrentPatient() {
                if (!this.currentPatientId) return;
                this.$confirm('确认删除该患者及其所有诊断记录？', '警告', {
                    confirmButtonText: '确定',
                    cancelButtonText: '取消',
                    type: 'warning'
                }).then(() => {
                    axios.delete(this.server_url + '/api/patients/' + this.currentPatientId)
                        .then(res => {
                            if (res.data.status === 1) {
                                this.$message.success('删除成功');
                                this.currentPatientId = null;
                                this.patient = null;
                                this.fetchPatientList();
                            } else {
                                this.$message.error(res.data.msg || '删除失败');
                            }
                        });
                }).catch(() => {});
            },
            // 重置患者列表
            resetPatientList() {
                this.currentPatientId = null;
                this.patient = null;
                this.diagnosisRecords = [];
                this.selectedRecord = null;
                this.trendData = null;
                this.forecastData = null;
                this.historyImageRecords = [];
                this.fetchPatientList();
                this.$message.info('已重置');
            },
            // 多项指标对比图
            drawMultiCompareChart() {
                if (!this.trendData) return;
                var chartDom = document.getElementById('multiCompareChart');
                if (!chartDom) return;
                var chart = this.$echarts.init(chartDom);
                var dates = this.trendData.dates || [];
                var seriesData = [];
                var legends = [];
                var indicators = ['area', 'perimeter', 'mean', 'std', 'ellipse'];
                var nameMap = {
                    area: '肿瘤面积', perimeter: '肿瘤周长',
                    mean: '灰度均值', std: '灰度方差', ellipse: '似圆度'
                };
                for (var i = 0; i < indicators.length; i++) {
                    var key = indicators[i];
                    var info = this.trendData[key];
                    if (info && info.values && info.values.length > 0) {
                        legends.push(info.name || nameMap[key]);
                        seriesData.push({
                            name: info.name || nameMap[key],
                            type: 'line',
                            data: info.values,
                            smooth: true
                        });
                    }
                }
                chart.setOption({
                    title: { text: '多项指标变化趋势对比', left: 'center' },
                    tooltip: { trigger: 'axis' },
                    legend: { data: legends, bottom: 0 },
                    grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
                    xAxis: { type: 'category', data: dates },
                    yAxis: { type: 'value' },
                    series: seriesData
                });
            },
            // 获取历史图像
            fetchHistoryImages() {
                if (!this.currentPatientId) {
                    this.historyImageRecords = [];
                    return;
                }
                axios.get(this.server_url + '/api/patients/' + this.currentPatientId + '/records')
                    .then(res => {
                        if (res.data.status === 1) {
                            this.historyImageRecords = (res.data.data || []).filter(r => r.status === 'completed');
                        }
                    })
                    .catch(err => {
                        console.warn('历史图像加载失败:', err);
                    });
            },
            // 查看某条诊断记录详情
            viewRecordDetail(id) {
                this.llmAdvice = null; // 切换记录时清空上次的建议
                axios.get(this.server_url + '/api/diagnosis/' + id)
                    .then(res => {
                        if (res.data.status === 1) {
                            this.selectedRecord = res.data.data;
                        }
                    })
                    .catch(err => {
                        console.warn('详情获取失败:', err);
                        this.$message.error('获取诊断详情失败');
                    });
            },
            // 拉取可用的 LLM 模型列表
            fetchLlmProviders() {
                axios.get(this.server_url + '/api/llm/providers')
                    .then(res => {
                        if (res.data.status === 1) {
                            this.llmProviders = res.data.data;
                            // 自动选择第一个可用的 provider
                            var available = this.llmProviders.filter(p => p.available);
                            if (available.length > 0 && !this.selectedProvider) {
                                this.selectedProvider = available[0].id;
                                this.selectedModel = available[0].default_model;
                            }
                        }
                    })
                    .catch(err => {
                        console.warn('获取 LLM 模型列表失败:', err);
                    });
            },
            // provider 切换时自动选择默认模型
            onProviderChange(pid) {
                var provider = this.llmProviders.find(p => p.id === pid);
                if (provider) {
                    this.selectedModel = provider.default_model;
                }
            },
            // 生成 LLM 辅助建议（支持选择模型）
            generateLlmAdvice(recordId) {
                this.llmLoading = true;
                this.llmAdvice = null;
                var payload = {};
                if (this.selectedProvider) payload.provider = this.selectedProvider;
                if (this.selectedModel) payload.model = this.selectedModel;
                axios.post(this.server_url + '/api/diagnosis/' + recordId + '/llm-advice', payload)
                    .then(res => {
                        if (res.data.status === 1 && res.data.data) {
                            this.llmAdvice = res.data.data;
                        } else {
                            this.$message.warning(res.data.msg || 'AI 建议生成失败');
                        }
                    })
                    .catch(err => {
                        console.warn('LLM 请求失败:', err);
                        this.$message.error('AI 建议生成失败，请稍后重试');
                    })
                    .finally(() => {
                        this.llmLoading = false;
                    });
            },
            // 获取时序预测数据
            fetchForecast() {
                if (!this.currentPatientId) {
                    this.$message.warning('请先选择患者');
                    return;
                }
                this.forecastLoading = true;
                this.forecastData = null;
                axios.get(this.server_url + '/api/patients/' + this.currentPatientId + '/forecast?steps=' + this.forecastSteps)
                    .then(res => {
                        if (res.data.status === 1 && res.data.data) {
                            this.forecastData = res.data.data;
                            this.$nextTick(() => {
                                this.drawForecastCharts();
                            });
                        } else {
                            this.$message.warning(res.data.msg || '预测失败');
                        }
                    })
                    .catch(err => {
                        console.warn('时序预测请求失败:', err);
                        this.$message.error('时序预测请求失败，请检查后端服务');
                    })
                    .finally(() => {
                        this.forecastLoading = false;
                    });
            },
            // 绘制所有预测图表
            drawForecastCharts() {
                if (!this.forecastData || !this.forecastData.features) return;

                var historyDates = this.forecastData.history_dates || [];
                var forecastDates = this.forecastData.forecast_dates || [];
                var allDates = historyDates.concat(forecastDates);

                for (var key in this.forecastData.features) {
                    var feat = this.forecastData.features[key];
                    var chartDom = document.getElementById('forecast_chart_' + key);
                    if (!chartDom) continue;

                    var myChart = this.$echarts.init(chartDom);

                    // 历史数据线
                    var histValues = [];
                    // 优先从 forecast 返回的 history_values 中取
                    if (this.forecastData.history_values && this.forecastData.history_values[key]) {
                        histValues = this.forecastData.history_values[key];
                    } else if (this.trendData && this.trendData[key]) {
                        histValues = this.trendData[key].values || [];
                    }

                    // 线性回归预测线（虚线延伸）
                    var regLine = [];
                    var regPredicted = (feat.regression && feat.regression.predicted) ? feat.regression.predicted : [];
                    // 填充历史部分为 null（不画线），只画预测部分
                    for (var j = 0; j < historyDates.length; j++) {
                        regLine.push(null);
                    }
                    // 连接最后一个历史值到预测起点
                    if (histValues.length > 0 && histValues[histValues.length - 1] != null) {
                        regLine[regLine.length - 1] = histValues[histValues.length - 1];
                    }
                    for (var k = 0; k < regPredicted.length; k++) {
                        regLine.push(regPredicted[k]);
                    }

                    // LSTM 预测线
                    var lstmLine = [];
                    var lstmPredicted = feat.lstm && feat.lstm.predicted ? feat.lstm.predicted : [];
                    for (var m = 0; m < historyDates.length; m++) {
                        lstmLine.push(null);
                    }
                    if (histValues.length > 0 && histValues[histValues.length - 1] != null) {
                        lstmLine[lstmLine.length - 1] = histValues[histValues.length - 1];
                    }
                    for (var p = 0; p < lstmPredicted.length; p++) {
                        lstmLine.push(lstmPredicted[p]);
                    }

                    // 综合预测线
                    var ensembleLine = [];
                    var ensemblePredicted = feat.ensemble_predicted || [];
                    for (var q = 0; q < historyDates.length; q++) {
                        ensembleLine.push(null);
                    }
                    if (histValues.length > 0 && histValues[histValues.length - 1] != null) {
                        ensembleLine[ensembleLine.length - 1] = histValues[histValues.length - 1];
                    }
                    for (var r = 0; r < ensemblePredicted.length; r++) {
                        ensembleLine.push(ensemblePredicted[r]);
                    }

                    // 分隔线的位置
                    var splitIndex = historyDates.length - 1;

                    myChart.setOption({
                        title: {
                            text: feat.name + ' - 时序预测',
                            left: 'center',
                            textStyle: { fontSize: 14 }
                        },
                        tooltip: {
                            trigger: 'axis'
                        },
                        legend: {
                            data: ['历史数据', '线性回归预测', 'LSTM预测', '综合预测'],
                            bottom: 0
                        },
                        grid: {
                            left: 60, right: 30, top: 45, bottom: 50
                        },
                        xAxis: {
                            type: 'category',
                            data: allDates,
                            axisLabel: { fontSize: 11 }
                        },
                        yAxis: {
                            type: 'value',
                            name: feat.name,
                            nameTextStyle: { fontSize: 12 }
                        },
                        series: [
                            {
                                name: '历史数据',
                                type: 'line',
                                data: histValues,
                                itemStyle: { color: '#409EFF' },
                                lineStyle: { width: 2 },
                                symbol: 'circle',
                                symbolSize: 6,
                                markLine: {
                                    silent: true,
                                    symbol: 'none',
                                    lineStyle: { type: 'dashed', color: '#E6A23C' },
                                    data: [{ xAxis: splitIndex }],
                                    label: { show: true, formatter: '预测起点', position: 'start', fontSize: 10 }
                                }
                            },
                            {
                                name: '线性回归预测',
                                type: 'line',
                                data: regLine,
                                itemStyle: { color: '#67C23A' },
                                lineStyle: { width: 2, type: 'dashed' },
                                symbol: 'diamond',
                                symbolSize: 6
                            },
                            {
                                name: 'LSTM预测',
                                type: 'line',
                                data: lstmLine,
                                itemStyle: { color: '#F56C6C' },
                                lineStyle: { width: 2, type: 'dotted' },
                                symbol: 'triangle',
                                symbolSize: 6
                            },
                            {
                                name: '综合预测',
                                type: 'line',
                                data: ensembleLine,
                                itemStyle: { color: '#E6A23C' },
                                lineStyle: { width: 3, type: 'solid' },
                                symbol: 'rect',
                                symbolSize: 8
                            }
                        ]
                    });
                }
            },
            // 创建模板下载链接
            downloads(data, name) {
                if (!data) {
                    return;
                }
                let url = window.URL.createObjectURL(new Blob([data]));
                let link = document.createElement("a");
                link.style.display = "none";
                link.href = url;
                link.setAttribute("download", `肿瘤CT图文件.zip`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
            },
            welcome() {
                axios({
                    method: "get",
                    url:
                        "https://cso1-1254043908.cos.ap-beijing.myqcloud.com/ct/testfile.7z",
                    responseType: "blob"
                }).then(res => {
                    this.downloads(res.data, res.headers.filename);
                    if (res.status === 200) {
                        this.$message({
                            message: "下载成功",
                            type: "success"
                        });
                        this.centerDialogVisible = false;
                        this.next();
                    } else {
                        this.$message({
                            showClose: true,
                            message: "下载失败，请重试",
                            type: "error"
                        });
                    }
                });
            },
            notice1() {
                this.$notify({
                    title: "预测成功",
                    message:
                        "点击图片可以查看大图，图片下方会显示肿瘤区域的一些特征值来供医生参考，辅助诊断",
                    duration: 0,
                    type: "success"
                });
            },
            // 导航方法
            goBack() {
                this.$router.push({ name: 'home' });
            },
            goToTrend() {
                if (this.currentPatientId) {
                    this.$router.push({ name: 'trend', params: { id: this.currentPatientId } });
                } else {
                    this.$message.warning('请先选择患者');
                }
            },
            // 获取特征值的参考区间
            getFeatureRange(featureKey) {
                return this.feature_ranges[featureKey] || '-';
            },
            // 获取特征值的说明
            getFeatureDescription(featureKey) {
                return this.feature_descriptions[featureKey] || '';
            }
        },
        mounted() {
            // 先拉取患者列表，再初始化趋势图
            this.fetchPatientList();
        }
    };
</script>

<style>
    .el-button {
        padding: 12px 20px !important;
    }

    #hello p {
        font-size: 15px !important;
        /*line-height: 25px;*/
    }

    .n1 .el-step__description {
        padding-right: 20%;
        font-size: 14px;
        line-height: 20px;
        /* font-weight: 400; */
    }
</style>

<style scoped>
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    .dialog_info {
        margin: 20px auto;
    }

    .text {
        font-size: 14px;
    }

    .item {
        margin-bottom: 18px;
    }

    .clearfix:before,
    .clearfix:after {
        display: table;
        content: "";
    }

    .clearfix:after {
        clear: both;
    }

    .box-card {
        width: 680px;
        height: 200px;
        border-radius: 8px;
        margin-top: -20px;
    }

    .divider {
        width: 50%;
    }

    #CT {
        display: flex;
        height: 100%;
        width: 70%;
        flex-wrap: wrap;
        justify-content: center;
        margin: 0 auto;
        margin-right: 0px;
        max-width: 1200px;
        /* background-color: RGB(239, 249, 251); */
        /* box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04); */
    }

    #CT_image_1 {
        width: 90%;
        height: 40%;
        /* background-color: RGB(239, 249, 251); */
        margin: 0px auto;
        padding: 0px auto;
        /* box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04); */
        margin-right: 180px;
        margin-bottom: 0px;
        border-radius: 4px;
    }

    #CT_image {
        margin-bottom: 60px;
        margin-left: 30px;
        margin-top: 5px;
    }

    .image_1 {
        width: 275px;
        height: 260px;
        background: #ffffff;
        box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    }

    .img_info_1 {
        height: 30px;
        width: 275px;
        text-align: center;
        background-color: #21b3b9;
        line-height: 30px;
    }

    .demo-image__preview1 {
        width: 250px;
        height: 290px;
        margin: 20px 60px;
        float: left;
    }

    .demo-image__preview2 {
        width: 250px;
        height: 290px;

        margin: 20px 460px;
        /* background-color: green; */
    }

    .error {
        margin: 100px auto;
        width: 50%;
        padding: 10px;
        text-align: center;
    }

    .block-sidebar {
        position: fixed;
        display: none;
        left: 50%;
        margin-left: 600px;
        top: 350px;
        width: 60px;
        z-index: 99;
    }

    .block-sidebar .block-sidebar-item {
        font-size: 50px;
        color: lightblue;
        text-align: center;
        line-height: 50px;
        margin-bottom: 20px;
        cursor: pointer;
        display: block;
    }

    div {
        display: block;
    }

    .block-sidebar .block-sidebar-item:hover {
        color: #187aab;
    }

    .download_bt {
        padding: 10px 16px !important;
    }

    #upfile {
        width: 104px;
        height: 45px;
        background-color: #187aab;
        color: #fff;
        text-align: center;
        line-height: 45px;
        border-radius: 3px;
        box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
        color: #fff;
        font-family: "Source Sans Pro", Verdana, sans-serif;
        font-size: 0.875rem;
    }

    .file {
        width: 200px;
        height: 130px;
        position: absolute;
        left: -20px;
        top: 0;
        z-index: 1;
        -moz-opacity: 0;
        -ms-opacity: 0;
        -webkit-opacity: 0;
        opacity: 0; /*css属性&mdash;&mdash;opcity不透明度，取值0-1*/
        filter: alpha(opacity=0);
        cursor: pointer;
    }

    #upload {
        position: relative;
        margin: 0px 0px;
    }

    #download {
        padding: 0px;
        margin: 0px 0px;
    }

    .patient {
        margin: 50px auto;
        margin-bottom: 100px;
        /* margin-right: 100px; */
        background-color: #187aab;
        border-radius: 5px;
        box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.1), 0 2px 2px 0 rgba(0, 0, 0, 0.2);
        color: #fff;
        font-family: "Source Sans Pro", Verdana, sans-serif;
        font-size: 0.875rem;
        line-height: 1;
        padding: 0.75rem 1.5rem;
    }

    /* 外层容器 */
    .content-wrapper {
        width: 95%;
        max-width: 1500px;
        margin: 0 auto;
    }

    #Content {
        width: 100%;
        min-height: 800px;
        background-color: #ffffff;
        margin: 0 auto;
        display: flex;
        gap: 20px;
    }

    /* 导航栏样式 */
    .content-top-nav {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
        border-radius: 8px;
        margin-bottom: 10px;
    }

    .nav-patient-name {
        margin-left: auto;
        font-size: 15px;
        color: #303133;
        font-weight: bold;
    }

    #aside {
        width: 280px;
        flex-shrink: 0;
        background-color: #ffffff;
        padding: 20px;
    }

    .divider {
        background-color: #eaeaea !important;
        height: 2px !important;
        width: 100%;
        margin-bottom: 50px;
    }

    .divider_1 {
        background-color: #ffffff;
        height: 2px !important;
        width: 100%;
        margin-bottom: 20px;
        margin: 20px auto;
    }

    .steps {
        font-family: "lucida grande", "lucida sans unicode", lucida, helvetica,
        "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
        color: #21b3b9;
        text-align: center;
        margin: 15px auto;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }

    .step_1 {
        /*color: #303133 !important;*/
        margin: 20px 26px;
    }

    #info_patient {
        margin-top: 10px;
    }

    /* 右侧 CT 区域自适应 */
    #CT {
        flex: 1;
        min-width: 0;
        overflow-x: auto;
    }

    /* 让 aside 里的卡片自适应 */
    #aside .box-card {
        width: 100% !important;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }

    /* 重新设计 CT 图像卡片的内部布局 */
    #CT_image_1 {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important; /* 覆盖内联的固定高度360px */
        margin-bottom: 20px !important;
        border-radius: 12px !important;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease;
    }
    #CT_image_1:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
    }

    /* 使用 /deep/ 使属性穿透至 element-ui 内部组件 */
    #CT_image_1 /deep/ .el-card__body {
        display: flex;
        justify-content: space-around;
        align-items: flex-start;
        padding: 20px;
    }

    #CT_image_1 .demo-image__preview1,
    #CT_image_1 .demo-image__preview2 {
        float: none;
        margin: 0 !important;
        width: 45% !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    #CT_image_1 .demo-image__preview1:hover,
    #CT_image_1 .demo-image__preview2:hover {
        transform: translateY(-3px);
    }

    #CT_image_1 .image_1 {
        width: 100% !important;
        height: 280px !important;
        object-fit: contain;
        border-radius: 10px 10px 0 0 !important;
    }

    #CT_image_1 .img_info_1 {
        width: 100% !important;
        background: linear-gradient(135deg, #21b3b9, #1c9b9f);
        font-weight: bold;
    }

    /* 特征值表格区域自适应 */
    #info_patient {
        width: 100%;
    }

    #info_patient .el-card {
        width: 100% !important;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
</style>



