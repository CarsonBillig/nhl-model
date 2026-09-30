@echo off
REM Daily NHL run: grade last night's picks, refresh data, make picks for today + the next 2 days,
REM rebuild docs\index.html, open it, and publish it to GitHub if this folder is connected to a repository.
REM Re-run in the late afternoon after entering confirmed starting goalies in goalie_overrides.csv.
REM Options:  run_daily.bat --days 1     (today only)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe nhl_predict.py %*
if errorlevel 1 (
  echo.
  echo The run failed - read the messages above.
  pause
  exit /b 1
)
if exist docs\index.html start "" "docs\index.html"
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  echo Not connected to GitHub yet - the page was only updated on this computer.
) else (
  echo Publishing to GitHub...
  git add -A
  git commit -q -m "Daily update %date% %time:~0,5%"
  git push -q
  if errorlevel 1 (echo GitHub push failed - check your internet connection or GitHub sign-in.) else (echo Published.)
)
echo.
echo Web page: docs\index.html    Today's board: output\picks_*.csv    Pick history: output\history.csv
pause
