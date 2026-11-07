@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -c "import gradio, ultralytics, pytorch_grad_cam, tabulate, scipy" 2>nul
if errorlevel 1 (
  echo [A2] Installing missing libraries - one time, a few minutes...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :err
)
if not exist weights\yolo11s.pt (
  echo [A2] Downloading YOLO11 + SAM 2 weights - one time...
  python get_weights.py
  if errorlevel 1 goto :err
)
echo [A2] Starting the app: http://127.0.0.1:7860
python app.py
if errorlevel 1 goto :err
exit /b 0
:err
echo.
echo [A2] Something went wrong - see the message above.
pause
