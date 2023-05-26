@ECHO OFF
TITLE Doccano conda prompt

setlocal disabledelayedexpansion
FOR /F "tokens=*" %%A IN ('type "config.ini"') DO SET %%A
cls

start "" /min doccano-server.bat
timeout 2
start "" /min doccano-task.bat
timeout 2
start "" /min doccano-export.bat
cls

call %powershell_path% -ExecutionPolicy ByPass -NoExit -Command "& %conda_hook% ; conda activate %env_root% ;"
