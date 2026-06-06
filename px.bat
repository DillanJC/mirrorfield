@echo off
REM Quick parallax shortcut - parallel mode by default
cd /d "%~dp0"
.venv\Scripts\python parallax_pipeline.py "%*" -v -m C
