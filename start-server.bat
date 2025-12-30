@echo off
echo Starting local server for Wplace Color Converter...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python -m http.server 8000

if errorlevel 1 (
    echo.
    echo ERROR: Python not found or failed to start server
    echo Please make sure Python is installed and added to PATH
    echo.
    pause
)
