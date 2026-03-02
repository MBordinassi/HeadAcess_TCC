param(
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python do venv nao encontrado em '$venvPython'. Crie o venv antes de gerar o exe."
}

Push-Location $PSScriptRoot
try {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install pyinstaller

    if ($OneFile) {
        & $venvPython -m PyInstaller --noconfirm --clean --onefile --name HeadAccess --collect-all mediapipe --collect-all cv2 --hidden-import pyautogui._pyautogui_win --add-data "models\face_landmarker.task;models" main.py
    }
    else {
        & $venvPython -m PyInstaller --noconfirm --clean headaccess.spec
    }

    Write-Host "Build finalizado. Veja a pasta dist\."
}
finally {
    Pop-Location
}
