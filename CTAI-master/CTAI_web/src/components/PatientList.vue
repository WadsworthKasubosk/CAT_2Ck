<template>
    <div class="patient-list-page">
        <div class="page-header">
            <h2><i class="el-icon-user"></i> 患者管理</h2>
            <div class="header-actions">
                <el-input
                    v-model="searchKeyword"
                    placeholder="搜索患者姓名/电话"
                    prefix-icon="el-icon-search"
                    style="width:250px;margin-right:12px;"
                    size="small"
                    clearable
                    @input="handleSearch"
                ></el-input>
                <el-button type="primary" size="small" icon="el-icon-plus" @click="addDialogVisible = true">添加患者</el-button>
                <el-button size="small" icon="el-icon-refresh" @click="fetchPatients">刷新</el-button>
            </div>
        </div>

        <!-- 系统简介 -->
        <el-row :gutter="15" style="margin-bottom:15px;">
            <el-col :span="8">
                <el-card shadow="hover" class="stat-card">
                    <div class="stat-icon" style="background:#409EFF;">
                        <i class="el-icon-user-solid"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-num">{{ patients.length }}</div>
                        <div class="stat-label">患者总数</div>
                    </div>
                </el-card>
            </el-col>
            <el-col :span="8">
                <el-card shadow="hover" class="stat-card">
                    <div class="stat-icon" style="background:#67C23A;">
                        <i class="el-icon-monitor"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-num">CT</div>
                        <div class="stat-label">AI 辅助诊断</div>
                    </div>
                </el-card>
            </el-col>
            <el-col :span="8">
                <el-card shadow="hover" class="stat-card">
                    <div class="stat-icon" style="background:#E6A23C;">
                        <i class="el-icon-data-analysis"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-num">LSTM</div>
                        <div class="stat-label">时序预测 + 趋势分析</div>
                    </div>
                </el-card>
            </el-col>
        </el-row>

        <el-card shadow="hover" class="table-card">
            <el-table
                :data="filteredPatients"
                v-loading="loading"
                element-loading-text="加载中..."
                border
                stripe
                style="width:100%"
                @row-click="handleRowClick"
                highlight-current-row
                :default-sort="{ prop: 'id', order: 'descending' }"
            >
                <el-table-column prop="id" label="ID" width="80" sortable></el-table-column>
                <el-table-column prop="name" label="姓名" width="120">
                    <template slot-scope="scope">
                        <span style="font-weight:bold;color:#303133;">{{ scope.row.name }}</span>
                    </template>
                </el-table-column>
                <el-table-column prop="gender" label="性别" width="80">
                    <template slot-scope="scope">
                        <el-tag :type="scope.row.gender === '男' ? '' : 'danger'" size="mini" effect="plain">
                            {{ scope.row.gender }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="age" label="年龄" width="80" sortable></el-table-column>
                <el-table-column prop="phone" label="电话" width="150"></el-table-column>
                <el-table-column prop="body_part" label="检查部位" width="120">
                    <template slot-scope="scope">
                        <el-tag type="info" size="mini">{{ scope.row.body_part || '直肠' }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="建档时间" width="180" sortable></el-table-column>
                <el-table-column label="操作" min-width="200">
                    <template slot-scope="scope">
                        <el-button type="primary" size="mini" icon="el-icon-view" 
                            @click.stop="goToDiagnosis(scope.row)">进入诊断</el-button>
                        <el-button type="success" size="mini" icon="el-icon-data-analysis" 
                            @click.stop="goToTrend(scope.row)">趋势分析</el-button>
                        <el-button type="danger" size="mini" icon="el-icon-delete" 
                            @click.stop="deletePatient(scope.row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <div class="table-footer">
                <span class="total-text">共 {{ filteredPatients.length }} 位患者</span>
            </div>
        </el-card>

        <!-- 添加患者对话框 -->
        <el-dialog title="添加患者" :visible.sync="addDialogVisible" width="450px" :close-on-click-modal="false">
            <el-form :model="newPatientForm" label-width="80px" size="small" ref="patientForm">
                <el-form-item label="姓名" required>
                    <el-input v-model="newPatientForm.name" placeholder="请输入患者姓名"></el-input>
                </el-form-item>
                <el-form-item label="性别" required>
                    <el-radio-group v-model="newPatientForm.gender">
                        <el-radio label="男">男</el-radio>
                        <el-radio label="女">女</el-radio>
                    </el-radio-group>
                </el-form-item>
                <el-form-item label="年龄">
                    <el-input-number v-model="newPatientForm.age" :min="0" :max="150" style="width:100%;"></el-input-number>
                </el-form-item>
                <el-form-item label="电话">
                    <el-input v-model="newPatientForm.phone" placeholder="请输入联系电话"></el-input>
                </el-form-item>
                <el-form-item label="检查部位">
                    <el-input v-model="newPatientForm.body_part" placeholder="直肠"></el-input>
                </el-form-item>
            </el-form>
            <span slot="footer" class="dialog-footer">
                <el-button size="small" @click="addDialogVisible = false">取消</el-button>
                <el-button type="primary" size="small" @click="submitAddPatient" :loading="submitting">确 定</el-button>
            </span>
        </el-dialog>
    </div>
</template>

<script>
import axios from "axios";

export default {
    name: "PatientList",
    data() {
        return {
            server_url: "http://127.0.0.1:5003",
            patients: [],
            loading: false,
            searchKeyword: "",
            addDialogVisible: false,
            submitting: false,
            newPatientForm: {
                name: "",
                gender: "男",
                age: 50,
                phone: "",
                body_part: "直肠"
            }
        };
    },
    computed: {
        filteredPatients() {
            if (!this.searchKeyword) return this.patients;
            var kw = this.searchKeyword.toLowerCase();
            return this.patients.filter(function(p) {
                return (
                    (p.name && p.name.toLowerCase().indexOf(kw) !== -1) ||
                    (p.phone && p.phone.indexOf(kw) !== -1) ||
                    (p.id && String(p.id).indexOf(kw) !== -1)
                );
            });
        }
    },
    created() {
        document.title = "肿瘤辅助诊断系统 - 患者列表";
        this.fetchPatients();
    },
    methods: {
        fetchPatients() {
            this.loading = true;
            axios.get(this.server_url + "/api/patients")
                .then(res => {
                    if (res.data.status === 1) {
                        this.patients = res.data.data || [];
                    }
                })
                .catch(err => {
                    this.$message.error("获取患者列表失败，请检查后端服务");
                    console.warn("患者列表获取失败:", err);
                })
                .finally(() => {
                    this.loading = false;
                });
        },
        handleSearch() {
            // computed 会自动过滤
        },
        handleRowClick(row) {
            this.goToDiagnosis(row);
        },
        goToDiagnosis(patient) {
            this.$router.push({ name: "diagnosis", params: { id: patient.id } });
        },
        goToTrend(patient) {
            this.$router.push({ name: "trend", params: { id: patient.id } });
        },
        deletePatient(patient) {
            this.$confirm(
                "确认删除患者「" + patient.name + "」及其所有诊断记录？此操作不可恢复。",
                "警告",
                {
                    confirmButtonText: "确定删除",
                    cancelButtonText: "取消",
                    type: "warning"
                }
            )
                .then(() => {
                    axios.delete(this.server_url + "/api/patients/" + patient.id)
                        .then(res => {
                            if (res.data.status === 1) {
                                this.$message.success("删除成功");
                                this.fetchPatients();
                            } else {
                                this.$message.error(res.data.msg || "删除失败");
                            }
                        })
                        .catch(() => {
                            this.$message.error("删除请求失败");
                        });
                })
                .catch(() => {});
        },
        submitAddPatient() {
            if (!this.newPatientForm.name || !this.newPatientForm.gender) {
                this.$message.warning("姓名和性别为必填项");
                return;
            }
            this.submitting = true;
            axios.post(this.server_url + "/api/patients", this.newPatientForm)
                .then(res => {
                    if (res.data.status === 1) {
                        this.$message.success("添加患者成功");
                        this.addDialogVisible = false;
                        this.newPatientForm = { name: "", gender: "男", age: 50, phone: "", body_part: "直肠" };
                        this.fetchPatients();
                    } else {
                        this.$message.error(res.data.msg || "添加失败");
                    }
                })
                .catch(err => {
                    this.$message.error("添加患者失败");
                    console.warn(err);
                })
                .finally(() => {
                    this.submitting = false;
                });
        }
    }
};
</script>

<style scoped>
.patient-list-page {
    width: 90%;
    max-width: 1400px;
    margin: 20px auto;
    padding: 0 20px;
}

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding: 15px 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
    border-radius: 8px;
}

.page-header h2 {
    color: #21b3b9;
    margin: 0;
    font-size: 22px;
    letter-spacing: 3px;
}

.page-header h2 i {
    margin-right: 8px;
}

.header-actions {
    display: flex;
    align-items: center;
}

.table-card {
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

.table-card .el-table {
    border-radius: 6px;
}

.table-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 15px;
    padding: 0 5px;
}

.total-text {
    font-size: 13px;
    color: #909399;
}

/* 表格行鼠标悬浮效果 */
.el-table .el-table__row {
    cursor: pointer;
    transition: background-color 0.2s;
}

/* 统计卡片高级悬浮效果 */
.stat-card {
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
}

.stat-card /deep/ .el-card__body {
    display: flex;
    align-items: center;
    padding: 18px 20px !important;
}

.stat-icon {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 16px;
    flex-shrink: 0;
}

.stat-icon i {
    font-size: 26px;
    color: #fff;
}

.stat-info {
    flex: 1;
    text-align: center;
}

.stat-num {
    font-size: 24px;
    font-weight: bold;
    color: #303133;
    line-height: 1.2;
}

.stat-label {
    font-size: 13px;
    color: #909399;
    margin-top: 4px;
}
</style>
