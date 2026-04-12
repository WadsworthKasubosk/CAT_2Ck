@echo off
chcp 65001 >nul 2>&1
title CTAI 一键启动
echo.
echo ========================================
echo    CTAI 一键启动脚本 (BAT)
echo ========================================
echo.

set ROOT=%~dp0
set FLASK_DIR=%ROOT%CTAI_flask
set DIST_DIR=%ROOT%CTAI_web\dist
set LOG_DIR=%ROOT%logs

:: 创建日志目录
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: 获取时间戳
for /f "tokens=1-6 delims=/:. " %%a in ("%date:~0,10% %time: =0%") do set TS=%%a-%%b-%%c_%%d-%%e-%%f
set FLASK_LOG=%LOG_DIR%\flask_%TS%.log
set FRONT_LOG=%LOG_DIR%\frontend_%TS%.log

:: -------- 0. 生成测试 DCM --------
echo [STEP 0] 生成测试 DCM 文件...
if exist "%FLASK_DIR%\test_data\test_sample.dcm" (
    echo   [OK] 测试 DCM 已存在，跳过
) else (
    cd /d "%ROOT%"
    python generate_test_dcm.py > "%LOG_DIR%\generate_dcm_%TS%.log" 2>&1
    if exist "%FLASK_DIR%\test_data\test_sample.dcm" (
        echo   [OK] 测试 DCM 生成成功
    ) else (
        echo   [WARN] 测试 DCM 生成失败，查看日志: %LOG_DIR%\generate_dcm_%TS%.log
    )
)
echo.

:: -------- 1. 创建必要目录 --------
echo [STEP 1] 创建必要目录...
if not exist "%FLASK_DIR%\uploads" mkdir "%FLASK_DIR%\uploads"
if not exist "%FLASK_DIR%\tmp\ct" mkdir "%FLASK_DIR%\tmp\ct"
if not exist "%FLASK_DIR%\tmp\image" mkdir "%FLASK_DIR%\tmp\image"
if not exist "%FLASK_DIR%\tmp\mask" mkdir "%FLASK_DIR%\tmp\mask"
if not exist "%FLASK_DIR%\tmp\draw" mkdir "%FLASK_DIR%\tmp\draw"
echo   [OK] 目录已就绪
echo.

:: -------- 2. 检查 Python --------
echo [STEP 2] 检查 Python...
python --version 2>nul
if errorlevel 1 (
    echo   [ERROR] 未找到 Python！请先安装 Python 3.7+
    pause
    exit /b 1
)
echo.

:: -------- 3. 启动 Flask 后端 --------
echo [STEP 3] 启动 Flask 后端 (端口 5003)...
cd /d "%FLASK_DIR%"
start "CTAI-Flask后端" cmd /c "python app.py > "%FLASK_LOG%" 2>&1 || (echo [ERROR] Flask 启动失败！错误日志: && type "%FLASK_LOG%" && pause)"
echo   [OK] Flask 后端启动中...
echo   地址: http://127.0.0.1:5003
echo.

:: 等几秒让后端初始化
echo   等待后端初始化 (5秒)...
timeout /t 5 /nobreak >nul
echo.

:: -------- 4. 启动前端 --------
echo [STEP 4] 启动前端静态服务 (端口 8080)...
if not exist "%DIST_DIR%" (
    echo   [ERROR] 前端 dist 目录不存在！
    echo   请先在 CTAI_web 目录执行: npm run build
) else (
    start "CTAI-前端服务" cmd /c "npx -y serve "%DIST_DIR%" -l 8080 > "%FRONT_LOG%" 2>&1 || (echo [ERROR] 前端启动失败！ && type "%FRONT_LOG%" && pause)"
    echo   [OK] 前端服务启动中...
    echo   地址: http://localhost:8080
)
echo.

:: -------- 5. 显示汇总 --------
echo ========================================
echo    CTAI 启动完成！
echo ========================================
echo.
echo   前端:     http://localhost:8080
echo   后端:     http://127.0.0.1:5003
echo.
echo   测试 DCM: CTAI_flask\test_data\test_sample.dcm
echo   真实 DCM: CTAI_flask\uploads\10013.dcm
echo   真实 DCM: CTAI_flask\uploads\测试用_10013.dcm
echo.
echo   Flask日志: %FLASK_LOG%
echo   前端日志:  %FRONT_LOG%
echo.
echo   关闭此窗口不会停止服务
echo   要停止服务请关闭对应的命令行窗口
echo.

:: 打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost:8080

pause
