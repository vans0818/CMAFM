@echo off
chcp 65001 > nul
echo.
echo ====================================================
echo   CMAFM Dashboard - 설치 프로그램
echo ====================================================
echo.

:: Python 설치 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo Python 3.11 이상을 먼저 설치해주세요:
    echo https://www.python.org/downloads/
    echo.
    echo 설치 시 반드시 "Add Python to PATH" 체크!
    pause
    exit /b 1
)

python --version
echo.

:: 가상환경 생성
if not exist "venv" (
    echo [1/4] 가상환경 생성 중...
    python -m venv venv
    echo 완료.
) else (
    echo [1/4] 가상환경이 이미 존재합니다. 건너뜁니다.
)

:: pip 업그레이드
echo [2/4] pip 업그레이드 중...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo 완료.

:: GPU 확인 후 torch 설치 결정
echo [3/4] PyTorch 설치 중...
nvidia-smi > nul 2>&1
if %errorlevel% equ 0 (
    echo     GPU(CUDA) 감지됨 - CUDA 버전으로 설치합니다.
    venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
) else (
    echo     GPU 없음 - CPU 버전으로 설치합니다. (추론 속도가 느릴 수 있습니다)
    venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
)
echo 완료.

:: 나머지 패키지 설치
echo [4/4] 나머지 패키지 설치 중...
venv\Scripts\pip.exe install streamlit opencv-python numpy pandas Pillow albumentations --quiet
echo 완료.

echo.
echo ====================================================
echo   설치 완료!
echo   이제 run_dashboard.bat 을 실행하세요.
echo ====================================================
echo.
pause
