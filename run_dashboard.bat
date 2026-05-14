@echo off
chcp 65001 > nul
echo.
echo ====================================================
echo   CMAFM Detection Dashboard
echo   RGB + LWIR Multispectral Object Detection
echo ====================================================
echo.

:: 가상환경이 있으면 사용, 없으면 시스템 Python 사용
set ROOT=%~dp0
if exist "%ROOT%venv\Scripts\streamlit.exe" (
    echo [가상환경 모드로 실행]
    set PYTHON="%ROOT%venv\Scripts\python.exe"
    set STREAMLIT="%ROOT%venv\Scripts\streamlit.exe"
) else (
    echo [시스템 Python 모드로 실행]
    set PYTHON=python
    set STREAMLIT=streamlit
)

cd /d "%ROOT%RGBThermal"
set KMP_DUPLICATE_LIB_OK=TRUE

echo.
echo 브라우저에서 http://localhost:8501 로 접속하세요.
echo 종료하려면 이 창에서 Ctrl+C 를 누르세요.
echo.

%STREAMLIT% run dashboard.py --server.port 8501 --server.maxUploadSize 500
pause
