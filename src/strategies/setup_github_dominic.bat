@echo off
echo 🔧 Configuring Git for Dominic - ProQuants Professional Repository
echo ================================================================

:: Configure Git with correct name
echo Setting up Git configuration for Dominic...
git config --global user.name "Dominic"
git config --global user.email "mzie_mvelo@yahoo.co.uk"

:: Verify configuration
echo.
echo Git configuration:
echo Name: 
git config --global user.name
echo Email: 
git config --global user.email

:: Navigate to project directory
cd /d "C:\Users\mzie_\source\vs_code_Deriv\WORKSPACE\ProQuants_Professional"

:: Initialize git if needed
if not exist ".git" (
    echo Initializing Git repository...
    git init
)

:: Add all files
echo Adding ProQuants Professional files to repository...
git add .

:: Commit with proper attribution to Dominic
echo Committing changes by Dominic...
git commit -m "ProQuants Professional Trading System by Dominic

Complete CREAM Strategy Implementation:
- ML-Enhanced Goloji Bhudasi Trading Logic  
- Advanced Neural Network with Mathematical Certainty
- Fractal Learning System M1→H4 timeframes
- User-configurable Risk Management (min RRR 1:4)
- Professional 9-panel GUI interface
- MT5 integration for Deriv synthetic indices
- Trading Bible compliant (14 timeframes)
- Python 3.11.9 compatible
- Manipulation detection and protection
- Scientific analysis with chaos theory and fractals

Author: Dominic
Email: mzie_mvelo@yahoo.co.uk
Repository: ProQuants-Professional-Trading-System"

echo.
echo ✅ SUCCESS! Git configured for Dominic and changes committed
echo 👤 Name: Dominic
echo 📧 Email: mzie_mvelo@yahoo.co.uk
echo.
echo Next steps:
echo 1. Create repository on GitHub: ProQuants-Professional-Trading-System
echo 2. Run: git remote add origin https://github.com/dominic-username/ProQuants-Professional-Trading-System.git
echo 3. Run: git push -u origin main
pause