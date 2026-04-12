// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import VueRouter from 'vue-router'
import axios from 'axios'
import Element from 'element-ui'
import echarts from "echarts";

Vue.prototype.$echarts = echarts;
import '../node_modules/element-ui/lib/theme-chalk/index.css'
import '../src/assets/style.css'
import './theme/index.css'

// 导入页面组件
import PatientList from './components/PatientList'
import Content from './components/Content'
import TrendAnalysis from './components/TrendAnalysis'

Vue.use(Element)
Vue.config.productionTip = false
Vue.use(VueRouter)
Vue.prototype.$http = axios

const router = new VueRouter({
    routes: [
        {
            path: "/",
            component: PatientList,
            name: "home",
            meta: { title: "肿瘤辅助诊断系统 - 患者列表" }
        },
        {
            path: "/patient/:id",
            component: Content,
            name: "diagnosis",
            meta: { title: "肿瘤辅助诊断系统 - 患者诊断" }
        },
        {
            path: "/patient/:id/trend",
            component: TrendAnalysis,
            name: "trend",
            meta: { title: "肿瘤辅助诊断系统 - 趋势分析" }
        },
        // 保留旧路由兼容
        {
            path: "/App",
            redirect: "/"
        },
    ],
    mode: "hash"
})

// 路由前置守卫，设置页面标题
router.beforeEach((to, from, next) => {
    document.title = to.meta.title || '肿瘤辅助诊断系统';
    next();
})

// 全局注册组件
Vue.component("App", App);
Vue.component("PatientList", PatientList);
Vue.component("Content", Content);
Vue.component("TrendAnalysis", TrendAnalysis);

/* eslint-disable no-new */
new Vue({
    el: '#app',
    router,
    render: h => h(App)
})
