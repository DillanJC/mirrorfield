@echo off
cd /d "%~dp0"
.venv\Scripts\python parallax_pipeline.py %*
