<template>
    <div id="Header">
        <div class="contact-info">
            <span class="phone">
                <i class="el-icon-phone-outline"></i>免费咨询：010-56732656
            </span>
            <span>
                <i class="el-icon-time"></i>工作时间：9:00-18:00
            </span>
        </div>

        <div id="word">
            <h1>{{ msg }}</h1>
        </div>

        <!-- 右侧导航 -->
        <div class="nav-links">
            <el-button v-if="showNav" size="mini" type="text" icon="el-icon-user" @click="goHome">患者列表</el-button>
            <el-button size="mini" type="text" icon="el-icon-setting" @click="settingsVisible = true" style="color:#E6A23C;">设置</el-button>
        </div>

        <!-- API Key 设置弹窗 -->
        <el-dialog title="🔧 AI 模型配置" :visible.sync="settingsVisible" width="640px" :close-on-click-modal="false">
            <el-alert
                title="配置 API Key 和接口地址后即可使用对应的 AI 模型。仅在内存中生效，重启服务后需重新配置。"
                type="info"
                :closable="false"
                style="margin-bottom:15px;"
                show-icon
            ></el-alert>

            <div v-for="p in providers" :key="p.id" class="provider-item">
                <div class="provider-header">
                    <span class="provider-name">{{ p.name }}</span>
                    <el-tag :type="p.available ? 'success' : 'info'" size="mini" effect="plain">
                        {{ p.available ? '已配置 ✓' : '未配置' }}
                    </el-tag>
                </div>

                <!-- API Key -->
                <div class="config-row">
                    <span class="config-label">API Key</span>
                    <el-input
                        v-model="apiKeys[p.id]"
                        :placeholder="p.id === 'ollama' ? '本地服务无需 Key' : '请输入 API Key'"
                        size="small"
                        show-password
                        :disabled="p.id === 'ollama'"
                    ></el-input>
                </div>

                <!-- Base URL -->
                <div class="config-row">
                    <span class="config-label">接口地址</span>
                    <el-input
                        v-model="baseUrls[p.id]"
                        :placeholder="getDefaultUrl(p.id)"
                        size="small"
                    ></el-input>
                </div>

                <!-- 可用模型 + 默认模型 -->
                <div class="config-row">
                    <span class="config-label">默认模型</span>
                    <el-select
                        v-model="defaultModels[p.id]"
                        placeholder="选择默认模型"
                        size="small"
                        style="flex:1;"
                        filterable
                        allow-create
                    >
                        <el-option v-for="m in p.models" :key="m" :label="m" :value="m"></el-option>
                    </el-select>
                </div>

                <div style="text-align:right;margin-top:8px;">
                    <el-button
                        type="primary"
                        size="small"
                        :loading="savingProvider === p.id"
                        @click="saveConfig(p)"
                    >保存配置</el-button>
                </div>
            </div>

            <span slot="footer" class="dialog-footer">
                <el-button size="small" @click="settingsVisible = false">关闭</el-button>
            </span>
        </el-dialog>
    </div>
</template>

<script>
import axios from "axios";

export default {
    name: "Header",
    data() {
        return {
            msg: "肿瘤辅助诊断系统",
            activeIndex: "1",
            settingsVisible: false,
            server_url: "http://127.0.0.1:5003",
            providers: [],
            apiKeys: {},
            baseUrls: {},
            defaultModels: {},
            savingProvider: "",
        };
    },
    computed: {
        showNav() {
            return this.$route && this.$route.name !== 'home';
        }
    },
    created() {
        this.fetchProviders();
    },
    methods: {
        goHome() {
            this.$router.push({ name: 'home' });
        },
        handleSelect(key, keyPath) {
            console.log(key, keyPath);
        },
        getDefaultUrl(pid) {
            var urlMap = {
                'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'deepseek': 'https://api.deepseek.com/v1',
                'openai': 'https://api.openai.com/v1',
                'ollama': 'http://localhost:11434/v1'
            };
            return urlMap[pid] || 'https://api.example.com/v1';
        },
        fetchProviders() {
            axios.get(this.server_url + "/api/llm/providers")
                .then(res => {
                    if (res.data.status === 1) {
                        this.providers = res.data.data;
                        // 初始化默认模型选择
                        var self = this;
                        this.providers.forEach(function(p) {
                            if (!self.defaultModels[p.id]) {
                                self.$set(self.defaultModels, p.id, p.default_model);
                            }
                        });
                    }
                })
                .catch(() => {});
        },
        saveConfig(provider) {
            var key = this.apiKeys[provider.id] || '';
            var base_url = this.baseUrls[provider.id] || '';
            var default_model = this.defaultModels[provider.id] || '';

            // Ollama：用固定 key
            if (provider.id === 'ollama') {
                key = 'ollama';
            } else if (!key) {
                this.$message.warning("请输入 API Key");
                return;
            }

            this.savingProvider = provider.id;
            var payload = {
                provider: provider.id,
                api_key: key
            };
            if (base_url) payload.base_url = base_url;
            if (default_model) payload.default_model = default_model;

            axios.post(this.server_url + "/api/llm/config", payload)
                .then(res => {
                    if (res.data.status === 1) {
                        this.$message.success(res.data.msg || "配置成功");
                        this.fetchProviders();
                    } else {
                        this.$message.error(res.data.msg || "配置失败");
                    }
                })
                .catch(err => {
                    this.$message.error("配置请求失败");
                    console.warn(err);
                })
                .finally(() => {
                    this.savingProvider = "";
                });
        }
    }
};
</script>

<style scoped>
#Header {
    padding: 30px 0;
    width: 90%;
    margin: 10px auto;
    position: relative;
}

/* 联系方式 */
.contact-info {
    position: absolute;
    left: 20px;
    top: 30px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.contact-info span {
    font-size: 16px;
    color: #999;
    display: flex;
    align-items: center;
}

.phone {
    color: #21b3b9 !important;
    font-weight: bold;
}

.contact-info i {
    font-size: 23px;
    margin-right: 10px;
}

/* 标题居中 */
#word {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    margin: 0;
    height: 60px;
    line-height: 3.2em;
}

h1 {
    color: #21b3b9;
    letter-spacing: 10px;
    font-size: 2.3em;
    text-align: center;
    white-space: nowrap;
}

/* 导航按钮 */
.nav-links {
    position: absolute;
    right: 20px;
    top: 35px;
    display: flex;
    gap: 5px;
}

/* 设置弹窗里的 provider 项 */
.provider-item {
    padding: 12px;
    margin-bottom: 10px;
    background: #f9fafc;
    border-radius: 8px;
    border: 1px solid #ebeef5;
}

.provider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.provider-name {
    font-weight: bold;
    font-size: 14px;
    color: #303133;
}

.config-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.config-label {
    width: 65px;
    flex-shrink: 0;
    font-size: 13px;
    color: #606266;
    text-align: right;
}

.config-row .el-input,
.config-row .el-select {
    flex: 1;
}

i,
input,
label {
    vertical-align: middle;
}

i {
    border: 0;
    display: block;
    cursor: pointer;
}
</style>
