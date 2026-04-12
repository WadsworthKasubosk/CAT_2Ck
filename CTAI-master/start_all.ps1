# ============================================================
# CTAI 一键启动脚本 (PowerShell)
# 功能：启动 Flask 后端 + 前端开发/静态服务 + 生成测试 DCM
# 日志：输出到 logs/ 目录
# 使用 fnm 管理 Node.js 版本 (Node 16)
# ============================================================

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   CTAI 一键启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---------- 创建日志目录 ----------
$LogDir = Join-Path $Root "logs"
if (!(Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$FlaskLog  = Join-Path $LogDir "flask_$Timestamp.log"
$FrontLog  = Join-Path $LogDir "frontend_$Timestamp.log"

Write-Host "[INFO] 日志目录: $LogDir" -ForegroundColor Gray
Write-Host ""

# ---------- 0. 生成测试 DCM 文件 ----------
Write-Host "[STEP 0] 生成测试 DCM 文件..." -ForegroundColor Yellow
$TestDcmPath = Join-Path $Root "CTAI_flask\test_data\test_sample.dcm"
if (Test-Path $TestDcmPath) {
    Write-Host "  [OK] 测试 DCM 文件已存在，跳过生成" -ForegroundColor Green
} else {
    try {
        Push-Location $Root
        python generate_test_dcm.py 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "generate_dcm_$Timestamp.log")
        Pop-Location
        if (Test-Path $TestDcmPath) {
            Write-Host "  [OK] 测试 DCM 文件生成成功" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] 测试 DCM 文件生成可能失败，请检查日志" -ForegroundColor Red
        }
    } catch {
        Write-Host "  [ERROR] 生成测试 DCM 失败: $_" -ForegroundColor Red
    }
}
Write-Host ""

# ---------- 1. 创建必要目录 ----------
Write-Host "[STEP 1] 创建必要目录..." -ForegroundColor Yellow
$FlaskRoot = Join-Path $Root "CTAI_flask"
$Dirs = @("uploads", "tmp\ct", "tmp\image", "tmp\mask", "tmp\draw")
foreach ($d in $Dirs) {
    $FullDir = Join-Path $FlaskRoot $d
    if (!(Test-Path $FullDir)) {
        New-Item -ItemType Directory -Path $FullDir -Force | Out-Null
    }
}
Write-Host "  [OK] 必要目录已就绪" -ForegroundColor Green
Write-Host ""

# ---------- 2. 检查 Python 和 Node.js ----------
Write-Host "[STEP 2] 检查环境..." -ForegroundColor Yellow
try {
    $PyVer = python --version 2>&1
    Write-Host "  [OK] Python: $PyVer" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] 未检测到 Python" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit 1
}

# 设置 fnm 环境
try {
    fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
    fnm use 16 2>&1 | Out-Null
    $NodeVer = node --version 2>&1
    Write-Host "  [OK] Node.js: $NodeVer (via fnm)" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] fnm 未安装或 Node 16 未安装" -ForegroundColor Yellow
    Write-Host "  [INFO] 运行: winget install Schniz.fnm && fnm install 16" -ForegroundColor Gray
    Write-Host "  [INFO] 将使用系统默认 Node.js" -ForegroundColor Gray
}
Write-Host ""

# ---------- 3. 启动 Flask 后端 ----------
Write-Host "[STEP 3] 启动 Flask 后端 (端口 5003)..." -ForegroundColor Yellow
$FlaskJob = Start-Job -ScriptBlock {
    param($FlaskRoot, $FlaskLog)
    Set-Location $FlaskRoot
    python app.py *>&1 | Out-File -FilePath $FlaskLog -Encoding utf8
} -ArgumentList $FlaskRoot, $FlaskLog

Write-Host "  [OK] Flask 后端启动中 (Job ID: $($FlaskJob.Id))" -ForegroundColor Green
Write-Host "  [INFO] 后端地址: http://127.0.0.1:5003" -ForegroundColor Cyan
Write-Host ""

# 等待后端启动
Write-Host "  等待后端启动 (5秒)..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# 检查后端健康
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:5003/" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  [OK] Flask 后端启动成功!" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Flask 后端可能还在启动中，请查看日志" -ForegroundColor Yellow
    if (Test-Path $FlaskLog) {
        Get-Content $FlaskLog -Tail 5 | ForEach-Object { Write-Host "  | $_" -ForegroundColor Gray }
    }
}
Write-Host ""

# ---------- 4. 启动前端服务 ----------
Write-Host "[STEP 4] 启动前端服务 (端口 8080)..." -ForegroundColor Yellow
$WebRoot = Join-Path $Root "CTAI_web"
$DistPath = Join-Path $WebRoot "dist"

# 检查 node_modules 是否存在
$NodeModules = Join-Path $WebRoot "node_modules"
if (!(Test-Path $NodeModules)) {
    Write-Host "  [INFO] node_modules 不存在，先安装依赖..." -ForegroundColor Yellow
    Push-Location $WebRoot
    npm install --registry https://registry.npmmirror.com 2>&1 | Out-File -FilePath (Join-Path $LogDir "npm_install_$Timestamp.log") -Encoding utf8
    Pop-Location
}

# 启动开发服务器
$FrontJob = Start-Job -ScriptBlock {
    param($WebRoot, $FrontLog)
    Set-Location $WebRoot
    # 尝试使用 fnm 切换到 Node 16
    try {
        fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
        fnm use 16 2>&1 | Out-Null
    } catch {}
    npx -y serve ./dist -l 8080 *>&1 | Out-File -FilePath $FrontLog -Encoding utf8
} -ArgumentList $WebRoot, $FrontLog

Write-Host "  [OK] 前端服务启动中 (Job ID: $($FrontJob.Id))" -ForegroundColor Green
Write-Host "  [INFO] 前端地址: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""

# ---------- 5. 汇总信息 ----------
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   CTAI 启动完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  前端地址:   http://localhost:8080" -ForegroundColor Cyan
Write-Host "  后端地址:   http://127.0.0.1:5003" -ForegroundColor Cyan
Write-Host ""
Write-Host "  测试 DCM 文件:" -ForegroundColor White
Write-Host "    - CTAI_flask\test_data\test_sample.dcm (模拟)" -ForegroundColor Gray
Write-Host "    - CTAI_flask\uploads\10013.dcm          (真实, 532KB)" -ForegroundColor Gray
Write-Host "    - CTAI_flask\uploads\测试用_10013.dcm    (真实, 532KB)" -ForegroundColor Gray
Write-Host ""
Write-Host "  AI 模型配置:" -ForegroundColor White
Write-Host "    设置环境变量后可使用对应模型:" -ForegroundColor Gray
Write-Host "    - DASHSCOPE_API_KEY  -> 通义千问 (qwen-plus/turbo)" -ForegroundColor Gray
Write-Host "    - DEEPSEEK_API_KEY   -> DeepSeek (deepseek-chat)" -ForegroundColor Gray
Write-Host "    - OPENAI_API_KEY     -> OpenAI (gpt-4o-mini)" -ForegroundColor Gray
Write-Host "    - Ollama 本地运行    -> 无需 Key" -ForegroundColor Gray
Write-Host ""
Write-Host "  日志:" -ForegroundColor White
Write-Host "    Flask: $FlaskLog" -ForegroundColor Gray
Write-Host "    前端:  $FrontLog" -ForegroundColor Gray
Write-Host ""
Write-Host "  按 Ctrl+C 停止所有服务" -ForegroundColor Yellow
Write-Host ""

# ---------- 6. 保持运行 & 监控 ----------
try {
    while ($true) {
        if ($FlaskJob.State -eq 'Failed' -or $FlaskJob.State -eq 'Completed') {
            Write-Host "[WARN] Flask 后端已退出 (状态: $($FlaskJob.State))" -ForegroundColor Red
            Receive-Job $FlaskJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            if (Test-Path $FlaskLog) {
                Write-Host "  --- 最近日志 ---" -ForegroundColor Gray
                Get-Content $FlaskLog -Tail 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
            break
        }
        Start-Sleep -Seconds 5
    }
} finally {
    Write-Host ""
    Write-Host "[INFO] 正在停止所有服务..." -ForegroundColor Yellow
    if ($FlaskJob) { Stop-Job $FlaskJob -ErrorAction SilentlyContinue; Remove-Job $FlaskJob -Force -ErrorAction SilentlyContinue }
    if ($FrontJob) { Stop-Job $FrontJob -ErrorAction SilentlyContinue; Remove-Job $FrontJob -Force -ErrorAction SilentlyContinue }
    Write-Host "[OK] 所有服务已停止" -ForegroundColor Green
}
