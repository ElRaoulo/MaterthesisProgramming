@echo off
cd /d "%~dp0"
call C:\Users\raoul\anaconda3\Scripts\activate.bat
call conda activate MasterThesis
python stock_prediction_script.py >> prediction_runs.log 2>&1
echo Job completed at %date% %time% >> prediction_runs.log