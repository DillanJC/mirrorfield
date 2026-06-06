@echo off
REM Optimal parallax sequence: Gemini -> Claude -> GPT -> Claude (synthesis)
cd /d "%~dp0"
.venv\Scripts\python parallax_sequence.py "%*" -s optimal -v
