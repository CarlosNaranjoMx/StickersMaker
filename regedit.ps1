$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$pythonLauncher = '<USR_DIR>\AppData\Local\Microsoft\WindowsApps\py.exe'
$pythonScript = Join-Path $scriptDir 'remove_background_objects.py'
$command = 'powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -Command "& ''{0}'' ''{1}'' --output-dir resultados --save-full --min-width 40 --min-height 40 ''%1''"' -f $pythonLauncher, $pythonScript

$root = [Microsoft.Win32.Registry]::CurrentUser
$menuKey = $root.CreateSubKey('Software\\Classes\\SystemFileAssociations\\image\\shell\\QuitarFondoConPython')
$menuKey.SetValue('', 'Quitar fondo con Python', [Microsoft.Win32.RegistryValueKind]::String)
$menuKey.SetValue('Icon', $pythonLauncher, [Microsoft.Win32.RegistryValueKind]::String)
$menuKey.Close()

$commandKey = $root.CreateSubKey('Software\\Classes\\SystemFileAssociations\\image\\shell\\QuitarFondoConPython\\command')
$commandKey.SetValue('', $command, [Microsoft.Win32.RegistryValueKind]::String)
$commandKey.Close()

Write-Host "Opción 'Quitar fondo con Python' agregada al menú contextual de imágenes." -ForegroundColor Green
Write-Host "Cierra y vuelve a abrir el Explorador de Windows si no aparece inmediatamente." -ForegroundColor Yellow
