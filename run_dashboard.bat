@echo off
REM Запускает Streamlit в новом окне и открывает браузер
SETLOCAL
cd /d "%~dp0"
start "Streamlit" cmd /k "streamlit run dashboard.py"
REM Подождём пару секунд, затем откроем браузер
timeout /t 2 > nul
start "" "http://localhost:8503"
ENDLOCAL
