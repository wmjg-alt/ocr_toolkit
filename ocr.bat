

@echo off
REM ========================================================
REM OCR TOOL LAUNCHER
REM Wraps the local_ocr Conda environment without activating it globally
REM ========================================================

REM 1. Define Paths (EDIT THIS LINE WITH YOUR PYTHON PATH)
set "PYTHON_EXE=C:\...pythonpath.."

REM 2. Define the Script Path (Assumes C:\Tools\OCR)
set "TOOL_SCRIPT=C:\Tools\OCR\cli.py"

REM 3. Run it
REM %* passes all arguments (image.jpg --engine qwen) to the python script
"%PYTHON_EXE%" "%TOOL_SCRIPT%" %*
