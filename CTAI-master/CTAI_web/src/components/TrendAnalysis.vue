<template>
    <div class="trend-analysis-page">
        <!-- 顶部导航栏 -->
        <div class="top-nav">
            <el-button size="small" icon="el-icon-arrow-left" @click="goBack">返回列表</el-button>
            <el-button size="small" type="primary" icon="el-icon-monitor" @click="goToDiagnosis">进入诊断</el-button>
            <span class="nav-title" v-if="patient">
                <i class="el-icon-user"></i> {{ patient.name }}
                <el-tag size="mini" style="margin-left:8px;">{{ patient.gender }}</el-tag>
                <el-tag size="mini" type="info" style="margin-left:4px;">{{ patient.age }}岁</el-tag>
                <el-tag v-if="diagnosisStatus" :type="diagnosisStatus.overall_type" size="small" style="margin-left:10px;font-weight:bold;">
                    {{ diagnosisStatus.overall }}
                </el-tag>
            </span>
        </div>

        <div class="trend-content-wrapper">
            <!-- 左侧：患者信息 + 诊断历史表格 -->
            <div class="trend-sidebar">
                <!-- 患者信息卡 -->
                <el-card shadow="hover" class="patient-card">
                    <div slot="header" class="clearfix">
                        <span style="font-weight:bold;color:#21b3b9;">
                            <i class="el-icon-user-solid"></i> 患者信息
                        </span>
                    </div>
                    <div v-if="patient" class="patient-info">
                        <div class="info-row"><span class="label">ID：</span><span>{{ patient.id }}</span></div>
                        <div class="info-row"><span class="label">姓名：</span><span>{{ patient.name }}</span></div>
                        <div class="info-row"><span class="label">性别：</span><span>{{ patient.gender }}</span></div>
                        <div class="info-row"><span class="label">年龄：</span><span>{{ patient.age }}</span></div>
                        <div class="info-row"><span class="label">电话：</span><span>{{ patient.phone || '未填写' }}</span></div>
                        <div class="info-row"><span class="label">部位：</span><span>{{ patient.body_part || '直肠' }}</span></div>
                        <div class="info-row"><span class="label">建档：</span><span>{{ patient.created_at }}</span></div>
                    </div>
                    <div v-else style="text-align:center;color:#999;padding:20px;">加载中...</div>
                </el-card>

                <!-- 诊断记录表格 -->
                <el-card shadow="hover" class="records-card">
                    <div slot="header" class="clearfix">
                        <span style="font-weight:bold;color:#21b3b9;">
                            <i class="el-icon-document"></i> 诊断记录
                        </span>
                        <el-tag size="mini" type="info" style="float:right;margin-top:2px;">
                            共 {{ diagnosisRecords.length }} 条
                        </el-tag>
                    </div>
                    <el-table
                        :data="diagnosisRecords"
                        v-loading="recordsLoading"
                        size="mini"
                        border
                        style="width:100%;"
                        max-height="350"
                    >
                        <el-table-column prop="id" label="#" width="45"></el-table-column>
                        <el-table-column label="时间" width="105">
                            <template slot-scope="scope">
                                {{ formatDate(scope.row.created_at) }}
                            </template>
                        </el-table-column>
                        <el-table-column label="面积" width="75">
                            <template slot-scope="scope">
                                {{ scope.row.area != null ? scope.row.area.toFixed(1) : '-' }}
                            </template>
                        </el-table-column>
                        <el-table-column label="周长" width="75">
                            <template slot-scope="scope">
                                {{ scope.row.perimeter != null ? scope.row.perimeter.toFixed(1) : '-' }}
                            </template>
                        </el-table-column>
                        <el-table-column label="状态" width="80">
                            <template slot-scope="scope">
                                <el-tag :type="getRecordStatusType(scope.row.id)" size="mini">
                                    {{ getRecordStatusText(scope.row.id) }}
                                </el-tag>
                            </template>
                        </el-table-column>
                    </el-table>
                </el-card>
            </div>

            <!-- 右侧：图表区域 -->
            <div class="trend-main">
                <el-card shadow="hover" class="chart-card">
                    <el-tabs v-model="activeTab" @tab-click="handleTabClick">
                        <!-- 面积趋势 -->
                        <el-tab-pane label="面积趋势" name="area-trend">
                            <div class="chart-header" v-if="trendData && trendData.area">
                                <el-tag :type="trendTagType(trendData.area.trend)" size="small">
                                    趋势：{{ trendData.area.trend }}
                                </el-tag>
                            </div>
                            <div id="trend-area-chart" style="width:100%;height:380px;"></div>
                        </el-tab-pane>

                        <!-- 周长趋势 -->
                        <el-tab-pane label="周长趋势" name="perimeter-trend">
                            <div class="chart-header" v-if="trendData && trendData.perimeter">
                                <el-tag :type="trendTagType(trendData.perimeter.trend)" size="small">
                                    趋势：{{ trendData.perimeter.trend }}
                                </el-tag>
                            </div>
                            <div id="trend-perimeter-chart" style="width:100%;height:380px;"></div>
                        </el-tab-pane>

                        <!-- 多指标雷达图 -->
                        <el-tab-pane label="多指标对比" name="radar">
                            <div id="trend-radar-chart" style="width:100%;height:400px;"></div>
                        </el-tab-pane>

                        <!-- 历史图像对比 -->
                        <el-tab-pane label="历史图像" name="history-images">
                            <div v-if="historyRecords.length === 0" style="text-align:center;color:#999;padding:40px;">
                                暂无历史图像，请先上传 CT 图像进行诊断
                            </div>
                            <div v-else class="image-grid">
                                <el-card v-for="rec in historyRecords" :key="rec.id" shadow="hover" class="image-item">
                                    <el-image :src="rec.draw_url || rec.image_url" style="width:100%;height:180px;" fit="contain">
                                        <div slot="error" style="text-align:center;padding:40px;color:#999;">图像不可用</div>
                                    </el-image>
                                    <div style="padding:8px 0;font-size:12px;color:#606266;">
                                        <p><b>诊断 #{{ rec.id }}</b></p>
                                        <p>时间: {{ formatDate(rec.created_at) }}</p>
                                        <p>面积: {{ rec.area != null ? rec.area.toFixed(1) : '-' }}</p>
                                        <p>周长: {{ rec.perimeter != null ? rec.perimeter.toFixed(1) : '-' }}</p>
                                    </div>
                                </el-card>
                            </div>
                        </el-tab-pane>
                    </el-tabs>
                </el-card>
            </div>
        </div>

        <!-- AI 辅助分析 —— 独立区域，放在页面底部 -->
        <el-card shadow="hover" class="ai-analysis-section">
            <div slot="header" class="clearfix">
                <span style="font-weight:bold;color:#21b3b9;font-size:16px;">
                    <i class="el-icon-cpu"></i> AI 辅助分析
                </span>
                <el-tag v-if="diagnosisStatus" :type="diagnosisStatus.overall_type" size="small" style="margin-left:12px;">
                    当前综合状态：{{ diagnosisStatus.overall }}
                </el-tag>
            </div>
            <div style="margin-bottom:15px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="font-size:13px;color:#606266;">AI 模型：</span>
                <el-select v-model="selectedProvider" placeholder="选择模型" size="small" style="width:160px;" @change="onProviderChange">
                    <el-option v-for="p in llmProviders" :key="p.id" :label="p.name + (p.available ? '' : ' (未配置)')" :value="p.id" :disabled="!p.available"></el-option>
                </el-select>
                <el-select v-model="selectedModel" placeholder="选择模型" size="small" style="width:180px;">
                    <el-option v-for="m in currentProviderModels" :key="m" :label="m" :value="m"></el-option>
                </el-select>
                <el-select v-model="selectedRecordId" placeholder="选择诊断记录" size="small" style="width:160px;">
                    <el-option v-for="r in completedRecords" :key="r.id" :label="'#' + r.id + ' ' + formatDate(r.created_at)" :value="r.id"></el-option>
                </el-select>
                <el-button type="primary" size="small" :loading="llmLoading" @click="generateAdvice" :disabled="!selectedRecordId">
                    🤖 生成分析建议
                </el-button>
            </div>

            <!-- 病情状态时间线 -->
            <div v-if="diagnosisStatus && diagnosisStatus.records && diagnosisStatus.records.length > 0" style="margin-bottom:15px;">
                <p style="font-size:13px;color:#606266;margin-bottom:8px;font-weight:bold;">📊 各次诊断病情变化（基于肿瘤面积变化率，阈值 ±5%）：</p>
                <div style="display:flex;flex-wrap:wrap;gap:8px;">
                    <el-tag v-for="(s, idx) in diagnosisStatus.records" :key="idx" :type="s.status_type" size="small" effect="light">
                        {{ s.date ? s.date.substring(5, 16) : '' }} · {{ s.status }} · {{ s.detail }}
                    </el-tag>
                </div>
            </div>

            <el-card v-if="llmAdvice" shadow="never" style="background:#f5f7fa;">
                <div slot="header" style="font-size:13px;color:#606266;">
                    <span>📝 AI 辅助建议</span>
                    <span v-if="llmAdvice.model_name" style="float:right;color:#909399;font-size:12px;">模型: {{ llmAdvice.model_name }}</span>
                </div>
                <p style="white-space:pre-wrap;line-height:1.8;font-size:14px;">{{ llmAdvice.advice }}</p>
                <p v-if="llmAdvice.disclaimer" style="color:#E6A23C;font-size:12px;margin-top:10px;border-top:1px solid #eee;padding-top:8px;">
                    ⚠️ {{ llmAdvice.disclaimer }}
                </p>
            </el-card>
            <div v-else-if="!llmLoading" style="text-align:center;color:#999;padding:30px;">
                请选择一条诊断记录，然后点击"生成分析建议"
            </div>
        </el-card>
    </div>
</template>

<script>
import axios from "axios";

export default {
    name: "TrendAnalysis",
    data() {
        return {
            server_url: "http://127.0.0.1:5003",
            patientId: null,
            patient: null,
            trendData: null,
            diagnosisStatus: null,
            diagnosisRecords: [],
            recordsLoading: false,
            historyRecords: [],
            activeTab: "area-trend",
            // LLM
            llmProviders: [],
            selectedProvider: "",
            selectedModel: "",
            selectedRecordId: null,
            llmAdvice: null,
            llmLoading: false,
        };
    },
    computed: {
        currentProviderModels() {
            var provider = this.llmProviders.find(p => p.id === this.selectedProvider);
            return provider ? provider.models : [];
        },
        completedRecords() {
            return this.diagnosisRecords.filter(r => r.status === "completed");
        }
    },
    created() {
        this.patientId = this.$route.params.id;
        document.title = "肿瘤辅助诊断系统 - 趋势分析";
        this.fetchPatient();
        this.fetchTrendData();
        this.fetchDiagnosisRecords();
        this.fetchLlmProviders();
    },
    mounted() {
        // 全局 resize 监听，让所有已创建的图表自动适应
        var self = this;
        this._resizeHandler = function() {
            var ids = ["trend-area-chart", "trend-perimeter-chart", "trend-radar-chart"];
            ids.forEach(function(id) {
                var dom = document.getElementById(id);
                if (dom) {
                    var inst = self.$echarts.getInstanceByDom(dom);
                    if (inst) inst.resize();
                }
            });
        };
        window.addEventListener("resize", this._resizeHandler);
    },
    beforeDestroy() {
        if (this._resizeHandler) {
            window.removeEventListener("resize", this._resizeHandler);
        }
    },
    methods: {
        goBack() {
            this.$router.push({ name: "home" });
        },
        goToDiagnosis() {
            this.$router.push({ name: "diagnosis", params: { id: this.patientId } });
        },
        formatDate(dateStr) {
            if (!dateStr) return "-";
            if (dateStr.length >= 16) return dateStr.substring(0, 16);
            return dateStr;
        },
        trendTagType(trend) {
            if (trend === "减小") return "success";
            if (trend === "增大") return "danger";
            return "info";
        },
        fetchPatient() {
            axios.get(this.server_url + "/api/patients/" + this.patientId)
                .then(res => {
                    if (res.data.status === 1) {
                        this.patient = res.data.data;
                    }
                })
                .catch(err => {
                    console.warn("患者信息获取失败:", err);
                });
        },
        fetchTrendData() {
            axios.get(this.server_url + "/api/patients/" + this.patientId + "/trend")
                .then(res => {
                    if (res.data.status === 1 && res.data.data) {
                        this.trendData = res.data.data;
                        // 提取病情状态信息（后端新增字段）
                        if (res.data.data.diagnosis_status) {
                            this.diagnosisStatus = res.data.data.diagnosis_status;
                        }
                        this.$nextTick(() => {
                            this.renderAreaChart();
                        });
                    }
                })
                .catch(err => {
                    console.warn("趋势数据获取失败:", err);
                });
        },
        fetchDiagnosisRecords() {
            this.recordsLoading = true;
            axios.get(this.server_url + "/api/patients/" + this.patientId + "/records")
                .then(res => {
                    if (res.data.status === 1) {
                        this.diagnosisRecords = res.data.data || [];
                        this.historyRecords = this.diagnosisRecords.filter(r => r.status === "completed");
                        // 默认选中最新的记录
                        if (this.completedRecords.length > 0 && !this.selectedRecordId) {
                            this.selectedRecordId = this.completedRecords[0].id;
                        }
                    }
                })
                .catch(err => {
                    console.warn("诊断记录获取失败:", err);
                })
                .finally(() => {
                    this.recordsLoading = false;
                });
        },
        fetchLlmProviders() {
            axios.get(this.server_url + "/api/llm/providers")
                .then(res => {
                    if (res.data.status === 1) {
                        this.llmProviders = res.data.data;
                        var available = this.llmProviders.filter(p => p.available);
                        if (available.length > 0 && !this.selectedProvider) {
                            this.selectedProvider = available[0].id;
                            this.selectedModel = available[0].default_model;
                        }
                    }
                })
                .catch(err => {
                    console.warn("获取 LLM 模型列表失败:", err);
                });
        },
        onProviderChange(pid) {
            var provider = this.llmProviders.find(p => p.id === pid);
            if (provider) {
                this.selectedModel = provider.default_model;
            }
        },
        generateAdvice() {
            if (!this.selectedRecordId) {
                this.$message.warning("请先选择一条诊断记录");
                return;
            }
            this.llmLoading = true;
            this.llmAdvice = null;
            var payload = {};
            if (this.selectedProvider) payload.provider = this.selectedProvider;
            if (this.selectedModel) payload.model = this.selectedModel;
            axios.post(this.server_url + "/api/diagnosis/" + this.selectedRecordId + "/llm-advice", payload)
                .then(res => {
                    if (res.data.status === 1 && res.data.data) {
                        this.llmAdvice = res.data.data;
                    } else {
                        this.$message.warning(res.data.msg || "AI 建议生成失败");
                    }
                })
                .catch(err => {
                    console.warn("LLM 请求失败:", err);
                    this.$message.error("AI 建议生成失败，请稍后重试");
                })
                .finally(() => {
                    this.llmLoading = false;
                });
        },
        handleTabClick(tab) {
            var self = this;
            // 延迟 150ms，确保 tab-pane 的 display 已切换完毕，容器有真实宽度
            setTimeout(function() {
                if (tab.name === "area-trend") {
                    self.renderAreaChart();
                } else if (tab.name === "perimeter-trend") {
                    self.renderPerimeterChart();
                } else if (tab.name === "radar") {
                    self.renderRadarChart();
                }
            }, 150);
        },
        renderAreaChart() {
            if (!this.trendData) return;
            var chartDom = document.getElementById("trend-area-chart");
            if (!chartDom) return;
            // 复用已有实例，或新建
            var chart = this.$echarts.getInstanceByDom(chartDom) || this.$echarts.init(chartDom);
            var dates = this.trendData.dates || [];
            var values = (this.trendData.area && this.trendData.area.values) || [];

            chart.setOption({
                title: { text: "肿瘤面积变化趋势", left: "center", textStyle: { fontSize: 16 } },
                tooltip: { trigger: "axis", formatter: "{b}<br/>面积: {c} mm²" },
                grid: { left: 60, right: 30, top: 50, bottom: 40 },
                xAxis: { type: "category", data: dates, name: "时间" },
                yAxis: { type: "value", name: "面积 (mm²)" },
                series: [{
                    name: "面积",
                    type: "line",
                    data: values,
                    smooth: true,
                    symbol: "circle",
                    symbolSize: 8,
                    itemStyle: { color: "#409EFF" },
                    lineStyle: { width: 3 },
                    areaStyle: {
                        color: {
                            type: "linear", x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: "rgba(64,158,255,0.3)" },
                                { offset: 1, color: "rgba(64,158,255,0.05)" }
                            ]
                        }
                    }
                }]
            });
            chart.resize();
        },
        renderPerimeterChart() {
            if (!this.trendData) return;
            var chartDom = document.getElementById("trend-perimeter-chart");
            if (!chartDom) return;
            var chart = this.$echarts.getInstanceByDom(chartDom) || this.$echarts.init(chartDom);
            var dates = this.trendData.dates || [];
            var values = (this.trendData.perimeter && this.trendData.perimeter.values) || [];

            chart.setOption({
                title: { text: "肿瘤周长变化趋势", left: "center", textStyle: { fontSize: 16 } },
                tooltip: { trigger: "axis", formatter: "{b}<br/>周长: {c} mm" },
                grid: { left: 60, right: 30, top: 50, bottom: 40 },
                xAxis: { type: "category", data: dates, name: "时间" },
                yAxis: { type: "value", name: "周长 (mm)" },
                series: [{
                    name: "周长",
                    type: "line",
                    data: values,
                    smooth: true,
                    symbol: "circle",
                    symbolSize: 8,
                    itemStyle: { color: "#67C23A" },
                    lineStyle: { width: 3 },
                    areaStyle: {
                        color: {
                            type: "linear", x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: "rgba(103,194,58,0.3)" },
                                { offset: 1, color: "rgba(103,194,58,0.05)" }
                            ]
                        }
                    }
                }]
            });
            chart.resize();
        },
        renderRadarChart() {
            if (!this.trendData || !this.diagnosisRecords.length) return;
            var chartDom = document.getElementById("trend-radar-chart");
            if (!chartDom) return;
            var chart = this.$echarts.getInstanceByDom(chartDom) || this.$echarts.init(chartDom);

            // 获取最早和最新的完成记录
            var completed = this.diagnosisRecords.filter(r => r.status === "completed");
            if (completed.length < 1) return;

            var latest = completed[0];
            var earliest = completed[completed.length - 1];

            // 从 image_info 提取更多指标并转为数字
            var extractVal = function(record, key) {
                if (!record || !record.image_info) return 0;
                var item = record.image_info[key];
                if (Array.isArray(item) && item.length >= 2) {
                    return parseFloat(item[1]) || 0;
                }
                return 0;
            };

            var area1 = earliest.area || 0, area2 = latest.area || 0;
            var peri1 = earliest.perimeter || 0, peri2 = latest.perimeter || 0;
            var mean1 = extractVal(earliest, "mean"), mean2 = extractVal(latest, "mean");
            var std1 = extractVal(earliest, "std"), std2 = extractVal(latest, "std");
            var elli1 = extractVal(earliest, "ellipse"), elli2 = extractVal(latest, "ellipse");

            // 动态设置各坐标轴最大值（取最大值的1.5倍留出绘图空间）
            var areaMax = Math.max(area1, area2) * 1.5 || 100;
            var periMax = Math.max(peri1, peri2) * 1.5 || 100;
            var meanMax = Math.max(mean1, mean2) * 1.5 || 100;
            var stdMax = Math.max(std1, std2) * 1.5 || 100;
            var ellipseMax = Math.max(elli1, elli2) * 1.5 || 1.0;

            chart.setOption({
                title: { text: "肿瘤指标对比分析", left: "center", top: 10 },
                tooltip: { trigger: "item" },
                legend: { data: ["初诊", "最近诊断"], bottom: 10 },
                radar: {
                    center: ["50%", "55%"],
                    radius: "55%",
                    shape: "circle",
                    indicator: [
                        { name: "面积", max: areaMax },
                        { name: "周长", max: periMax },
                        { name: "灰度均值", max: meanMax },
                        { name: "似圆度", max: ellipseMax },
                        { name: "灰度方差", max: stdMax }
                    ]
                },
                series: [{
                    name: "初诊 vs 最近",
                    type: "radar",
                    data: [
                        {
                            value: [
                                earliest.area || 0,
                                earliest.perimeter || 0,
                                extractVal(earliest, "mean") || 0,
                                extractVal(earliest, "ellipse") || 0,
                                extractVal(earliest, "std") || 0
                            ],
                            name: "初诊",
                            symbolSize: 6,
                            lineStyle: { width: 2, color: "#E6A23C" },
                            areaStyle: { color: "rgba(230,162,60,0.3)" }
                        },
                        {
                            value: [
                                latest.area || 0,
                                latest.perimeter || 0,
                                extractVal(latest, "mean") || 0,
                                extractVal(latest, "ellipse") || 0,
                                extractVal(latest, "std") || 0
                            ],
                            name: "最近诊断",
                            symbolSize: 6,
                            lineStyle: { width: 2, color: "#67C23A" },
                            areaStyle: { color: "rgba(103,194,58,0.3)" }
                        }
                    ]
                }]
            });
            chart.resize();
        },
        // 获取诊断记录的病情状态文本
        getRecordStatusText(recordId) {
            if (!this.diagnosisStatus || !this.diagnosisStatus.records) return '成功';
            var found = this.diagnosisStatus.records.find(function(s) { return s.record_id === recordId; });
            return found ? found.status : '成功';
        },
        // 获取诊断记录的病情状态类型（用于 el-tag 颜色）
        getRecordStatusType(recordId) {
            if (!this.diagnosisStatus || !this.diagnosisStatus.records) return 'success';
            var found = this.diagnosisStatus.records.find(function(s) { return s.record_id === recordId; });
            return found ? found.status_type : 'success';
        },
    }
};
</script>

<style scoped>
.trend-analysis-page {
    width: 95%;
    max-width: 1500px;
    margin: 15px auto;
    padding: 0 15px;
}

.top-nav {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding: 12px 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

.nav-title {
    margin-left: auto;
    font-size: 15px;
    color: #303133;
    font-weight: bold;
}

.trend-content-wrapper {
    display: flex;
    gap: 20px;
}

.trend-sidebar {
    width: 380px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.trend-main {
    flex: 1;
    min-width: 0;
}

.patient-card .patient-info {
    font-size: 14px;
}

.info-row {
    padding: 5px 0;
    border-bottom: 1px dashed #f0f0f0;
}

.info-row:last-child {
    border-bottom: none;
}

.info-row .label {
    font-weight: bold;
    color: #606266;
    display: inline-block;
    width: 60px;
}

.chart-card {
    border-radius: 12px;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

.ai-analysis-section {
    margin-top: 20px;
    border-radius: 12px;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

.patient-card, .records-card {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

.chart-header {
    margin-bottom: 10px;
}

.image-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.image-item {
    width: calc(33.333% - 8px);
    min-width: 200px;
}

@media screen and (max-width: 1200px) {
    .trend-content-wrapper {
        flex-direction: column;
    }
    .trend-sidebar {
        width: 100%;
        flex-direction: row;
    }
    .patient-card, .records-card {
        flex: 1;
    }
    .image-item {
        width: calc(50% - 6px);
    }
}
</style>
