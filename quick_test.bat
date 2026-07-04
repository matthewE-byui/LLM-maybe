@echo off
setlocal

cd /d "%~dp0"

echo ================================================
echo JARVIS Quick Test Runner
echo ================================================

set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Could not find venv python at: %PY%
  echo Run setup first, then retry.
  exit /b 1
)

echo [1/4] Startup and loop status...
"%PY%" -c "from jarvis_ai import JARVIS; j=JARVIS(); print(j.autolearn_status()); print(j.proactive_status()); print('stop', j.stop_autolearn(), j.stop_proactive_mode())"
if errorlevel 1 goto :fail

echo [2/4] Goal system smoke test...
"%PY%" -c "from jarvis_ai import JARVIS; j=JARVIS(); g=j.goals.add_goal('Ship remote assistant API', priority='high'); print('goal_id', g['id']); print('active', len(j.goals.list_goals('active'))); j.goals.update_progress(g['id'], 35); print('focus', j.goals.next_focus_goal()['id']); j.goals.complete_goal(g['id']); print('done', len(j.goals.list_goals('done'))); print('stop', j.stop_autolearn(), j.stop_proactive_mode())"
if errorlevel 1 goto :fail

echo [3/4] Skill routing smoke test...
"%PY%" -c "from jarvis_ai import JARVIS; j=JARVIS(); print('auto', j.skills.choose_skill('make me a deployment plan')); print('planner', j._run_skill('planner','deploy this assistant everywhere','')); print('stop', j.stop_autolearn(), j.stop_proactive_mode())"
if errorlevel 1 goto :fail

echo [4/4] Lookup reliability test...
"%PY%" -c "from jarvis_ai import JARVIS; j=JARVIS(); r=j.web_lookup('apple fruit', silent=True, train=False); print('results', len(r)); print(r[0][:180] if r else 'none'); print('stop', j.stop_autolearn(), j.stop_proactive_mode())"
if errorlevel 1 goto :fail

echo.
echo [PASS] Quick tests completed.
exit /b 0

:fail
echo.
echo [FAIL] One of the quick tests failed. Check output above.
exit /b 1
