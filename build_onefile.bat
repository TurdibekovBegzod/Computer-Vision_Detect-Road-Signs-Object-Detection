@echo off
cd /d "%~dp0"
venv\Scripts\pyinstaller.exe RoadSignsDetector.spec
