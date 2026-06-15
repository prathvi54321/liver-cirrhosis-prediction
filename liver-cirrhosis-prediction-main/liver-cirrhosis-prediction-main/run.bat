@echo off
cd /d "%~dp0"

REM Check if virtual environment exists, if not create one
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn[standard] pydantic sqlalchemy python-multipart python-jose[cryptography] passlib[bcrypt] requests fpdf2

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install
cd ..

REM Start backend in a new window
echo Starting backend server on port 8000...
start "Liver Cirrhosis Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && python main.py"

REM Start frontend in a new window
echo Starting frontend server on port 3000...
start "Liver Cirrhosis Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo Both servers are starting:
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo ========================================
echo.
pause
