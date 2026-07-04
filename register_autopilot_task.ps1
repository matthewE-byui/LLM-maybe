param(
    [string]$TaskName = "JARVIS-Autopilot",
    [string]$ProjectRoot = $PSScriptRoot,
    [int]$CycleMinutes = 20,
    [int]$ForceMaintenanceEvery = 18,
    [string]$DailyStartTime = "03:00"
)

$launcher = Join-Path $ProjectRoot "launch_autopilot_learning.bat"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found: $launcher"
}

$argString = "$CycleMinutes 0 $ForceMaintenanceEvery"
$action = New-ScheduledTaskAction -Execute $launcher -Argument $argString -WorkingDirectory $ProjectRoot
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn

$time = [datetime]::ParseExact($DailyStartTime, "HH:mm", $null)
$triggerDaily = New-ScheduledTaskTrigger -Daily -At $time

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances Parallel
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($triggerLogon, $triggerDaily) -Settings $settings -Principal $principal -Force

Write-Host "Registered task '$TaskName'"
Write-Host "Launcher: $launcher"
Write-Host "Arguments: $argString"
Write-Host "Triggers: AtLogOn + Daily $DailyStartTime"
