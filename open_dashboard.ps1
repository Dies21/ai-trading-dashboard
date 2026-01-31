# Открывает браузер на локальном Streamlit-дэшборде (порт можно изменить)
param(
    [string]$Url = "http://localhost:8503",
    [int]$Port = 8503,
    [int]$TimeoutSeconds = 30
)

Write-Host "Открываю $Url ..."
# Ждём пока порт станет доступен (до $TimeoutSeconds)
$end = (Get-Date).AddSeconds($TimeoutSeconds)
while (-not (Test-NetConnection -ComputerName 'localhost' -Port $Port -InformationLevel Quiet)) {
    if ((Get-Date) -gt $end) {
        Write-Host "Порт $Port недоступен — открою ссылку сразу." -ForegroundColor Yellow
        break
    }
    Start-Sleep -Seconds 1
}
Start-Process $Url
