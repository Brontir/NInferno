# package-native.ps1 [-Archs 120a,100,90,89,86,80,75] [-OutDir dist] [-Version 0.1.0] [-Vcpkg C:\vcpkg]
#                    [-Cuda70 "C:\...\CUDA\v12.8"] [-Jobs N]
#
# Builds NInferno with MSVC + CUDA on Windows and packs the one archive NInferno-Setup.exe installs
# on every machine (no WSL), whatever its GPUs:
#
#   ninferno-<version>-windows-x86_64/
#     bin/    ninferno-serve.exe           the entry point (apps/dispatch): starts the engine for the
#                                          GPU a command runs on
#             ninferno-serve-sm<tag>.exe   one engine per GPU generation (120a Blackwell RTX/PRO,
#                                          100 datacenter Blackwell, 90 Hopper, 89 Ada (with PTX: any
#                                          later GPU), 86 and 80 Ampere, 75 Turing, 70 Volta)
#             ninferno-ui.exe, ninferno-convert.exe, the DLLs they load (FFmpeg, curl, OpenSSL, zlib
#             from vcpkg; the Visual C++ runtime); the CUDA runtime is linked statically
#     share/ninferno/   engines.json (which engine runs where), route tables, drafter rankings, the
#                       chat window (llama-ui), release.json
#
#   -> ninferno-windows-x86_64.zip (+ .sha256), the name the release manifest lists.
#
# Each architecture builds in its own tree (build-windows-<arch>; NINFERNO_BUILD_ROOT moves them).
# Needs Visual Studio 2022 Build Tools (C++), the CUDA Toolkit 13.x (12.8 for Volta: -Cuda70),
# CMake, Ninja and vcpkg (vcpkg.json; the first build compiles FFmpeg and curl once).
param(
    [string]$Archs = "120a,100,90,89,86,80,75",
    [string]$OutDir = "dist",
    [string]$Version = "",
    [string]$Vcpkg = $env:VCPKG_ROOT,
    [string]$Cuda70 = "",
    [int]$Jobs = 0
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (-not $Version) { $Version = (git -C $Root describe --tags --always --dirty 2>$null); if (-not $Version) { $Version = 'dev' } }
# The manifest's spelling (0.1.0, not v0.1.0): the installer finds ninferno-<version>-windows-x86_64 by it.
$Version = $Version -replace '^v', ''
if (-not $Vcpkg -or -not (Test-Path "$Vcpkg\scripts\buildsystems\vcpkg.cmake")) { throw "vcpkg not found: pass -Vcpkg or set VCPKG_ROOT" }
$OutDir = [IO.Path]::GetFullPath($OutDir)
$BuildRoot = if ($env:NINFERNO_BUILD_ROOT) { $env:NINFERNO_BUILD_ROOT } else { $Root }
$Name = "ninferno-$Version-windows-x86_64"
$Stage = "$OutDir\$Name"
$Zip = "ninferno-windows-x86_64.zip"
$ArchList = $Archs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
# -Cuda70 adds the Volta engine (CUDA 12.8 builds it; the default list's CUDA 13 cannot).
if ($Cuda70 -and -not ($ArchList -contains '70')) { $ArchList += '70' }

# The MSVC environment (cl, link, the Windows SDK) for x64, as a "x64 Native Tools" prompt has it.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    # Visual Studio 2022 (17.x): the host compiler the CUDA Toolkit supports (a newer one may not be).
    $vs = & $vswhere -latest -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "Visual Studio 2022 Build Tools with the C++ workload not found" }
    $vars = cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>nul && set"
    foreach ($line in $vars) { if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
}

# CMake, nvcc and cl write warnings to stderr; their exit codes are checked instead.
$ErrorActionPreference = 'Continue'
$installed = if ($env:VCPKG_INSTALLED_DIR) { $env:VCPKG_INSTALLED_DIR } else { "$Root\vcpkg_installed" }
$parallel = if ($Jobs -gt 0) { @('--parallel', $Jobs) } else { @('--parallel') }
$engines = @()
$first = $null
foreach ($arch in $ArchList) {
    # Volta needs CUDA 12.8 (CUDA 13 dropped sm_70); without it the release has no Volta engine.
    $nvcc = $null
    if ($arch -eq '70') {
        if (-not $Cuda70) { Write-Warning "no -Cuda70 (CUDA 12.8): the archive has no Volta engine"; continue }
        $nvcc = "$Cuda70\bin\nvcc.exe"
    }
    # Ada's engine carries its PTX too: the driver compiles it, once, for a GPU no engine names (a
    # generation newer than this release).
    $ptx = $arch -eq '89'
    $build = "$BuildRoot\build-windows-$arch"
    $configure = @('-S', $Root, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_CUDA_ARCHITECTURES=$arch", "-DNINFERNO_CUDA_EMBED_PTX=$(if ($ptx) { 'ON' } else { 'OFF' })",
        '-DNINFERNO_CUDA_LINEINFO=OFF', '-DNINFERNO_BUILD_APPS=ON', '-DBUILD_TESTING=OFF',
        '-DNINFERNO_BUILD_BENCHMARKS=OFF', '-DNINFERNO_SOURCE_TREE_PATHS=OFF',
        "-DNINFERNO_VERSION_STRING=$Version",
        "-DCMAKE_TOOLCHAIN_FILE=$Vcpkg\scripts\buildsystems\vcpkg.cmake", '-DVCPKG_TARGET_TRIPLET=x64-windows',
        "-DVCPKG_INSTALLED_DIR=$installed")
    if ($nvcc) { $configure += "-DCMAKE_CUDA_COMPILER=$nvcc" }
    & cmake @configure
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (sm_$arch)" }
    # The first tree builds the programs every machine shares; the others their engine alone.
    $targets = if ($null -eq $first) { @('ninferno-serve', 'ninferno-ui', 'ninferno-convert', 'ninferno-dispatch') } else { @('ninferno-serve') }
    & cmake --build $build @parallel --target $targets
    if ($LASTEXITCODE -ne 0) { throw "build failed (sm_$arch)" }
    if ($null -eq $first) {
        $first = $build
        if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
        & cmake --install $build --prefix $Stage --strip
        if ($LASTEXITCODE -ne 0) { throw "cmake --install failed" }
        Remove-Item "$Stage\bin\ninferno-serve.exe"
        # The chat window (llama-ui), fetched at configure time: a configure that could not fetch it
        # only warns, and an archive without it has a page with no chat.
        if (-not (Test-Path "$Stage\share\ninferno\llama-ui\index.html")) {
            throw "share\ninferno\llama-ui\index.html missing: configure with network access (NINFERNO_UI_CHAT)"
        }
    }
    Copy-Item "$build\apps\ninferno-serve.exe" "$Stage\bin\ninferno-serve-sm$arch.exe"
    $engines += [ordered]@{ tag = $arch; arch = [int]($arch -replace '[^0-9]', ''); specific = [bool]($arch -match '[af]$'); ptx = $ptx }
}
if ($null -eq $first) { throw "no architecture built" }
Copy-Item "$first\apps\ninferno-dispatch.exe" "$Stage\bin\ninferno-serve.exe"
[ordered]@{ engines = $engines } | ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii "$Stage\share\ninferno\engines.json"

# The DLLs the executables load, from vcpkg's bin (the system's and the NVIDIA driver's stay out).
$vcpkgBin = Get-ChildItem -Path $first -Recurse -Filter 'avcodec*.dll' -ErrorAction SilentlyContinue |
    Select-Object -First 1 | ForEach-Object { $_.DirectoryName }
if (-not $vcpkgBin) { throw "vcpkg's DLLs not found under $first" }
Get-ChildItem "$vcpkgBin\*.dll" | Copy-Item -Destination "$Stage\bin"
# Their licenses: each vcpkg port's copyright file (vcpkg_installed\<triplet>\share\<port>).
$copyrights = @(Get-ChildItem "$installed\x64-windows\share\*\copyright" -ErrorAction SilentlyContinue)
if ($copyrights.Count -eq 0) { throw "no vcpkg copyright files under $installed\x64-windows\share" }
$copyrights | ForEach-Object {
    $port = Join-Path "$Stage\share\licenses" $_.Directory.Name
    New-Item -ItemType Directory -Force $port | Out-Null
    Copy-Item $_.FullName (Join-Path $port 'copyright')
}
# The Visual C++ runtime (msvcp140*, vcruntime140*), app-local as Microsoft's redistribution terms
# allow: a machine without the VC++ Redistributable runs NInferno too (the UCRT is Windows' own).
$crt = Get-ChildItem "$env:VCToolsRedistDir\x64" -Directory -Filter 'Microsoft.VC*.CRT' | Select-Object -First 1
if (-not $crt) { throw "the Visual C++ runtime (VCToolsRedistDir\x64\Microsoft.VC*.CRT) not found" }
Get-ChildItem "$($crt.FullName)\*.dll" | Copy-Item -Destination "$Stage\bin"
# pdfium for PDF documents, loaded at run time from bin\ (the pin pdfium.sh reads too).
$pdfiumPin = @{}
Get-Content "$PSScriptRoot\..\pdfium.pin" | Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object { $key, $value = $_ -split ' ', 2; $pdfiumPin[$key] = $value }
$pdfiumTag = $pdfiumPin['tag']
$pdfiumSha = $pdfiumPin['win-x64']
$pdfiumTmp = Join-Path ([IO.Path]::GetTempPath()) "ninferno-pdfium-$PID"
New-Item -ItemType Directory -Force $pdfiumTmp | Out-Null
try {
    $pdfiumTgz = Join-Path $pdfiumTmp 'pdfium-win-x64.tgz'
    if ($env:NINFERNO_PDFIUM_ARCHIVE) { Copy-Item $env:NINFERNO_PDFIUM_ARCHIVE $pdfiumTgz }
    else { Invoke-WebRequest -UseBasicParsing -OutFile $pdfiumTgz "https://github.com/bblanchon/pdfium-binaries/releases/download/$($pdfiumTag -replace '/', '%2F')/pdfium-win-x64.tgz" }
    if ((Get-FileHash -Algorithm SHA256 $pdfiumTgz).Hash.ToLower() -ne $pdfiumSha) { throw "pdfium $pdfiumTag checksum mismatch" }
    & "$env:SystemRoot\System32\tar.exe" -xzf $pdfiumTgz -C $pdfiumTmp
    Copy-Item "$pdfiumTmp\bin\pdfium.dll" "$Stage\bin"
    New-Item -ItemType Directory -Force "$Stage\share\licenses\pdfium" | Out-Null
    Copy-Item "$pdfiumTmp\LICENSE" "$Stage\share\licenses\pdfium\LICENSE"
    Copy-Item -Recurse "$pdfiumTmp\licenses" "$Stage\share\licenses\pdfium\third_party"
} finally { Remove-Item -Recurse -Force $pdfiumTmp }
& "$Stage\bin\ninferno-serve.exe" --version
$cuda = (& nvcc --version | Select-String 'release ([0-9.]+)').Matches[0].Groups[1].Value
@{ version = $Version; archs = @($engines | ForEach-Object { $_.tag }); cuda = $cuda; os = 'windows'; built = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') } |
    ConvertTo-Json -Compress | Set-Content -Encoding ascii "$Stage\share\ninferno\release.json"
Copy-Item "$Root\LICENSE", "$Root\NOTICE" $Stage

Push-Location $OutDir
try {
    if (Test-Path $Zip) { Remove-Item $Zip }
    & "$env:SystemRoot\System32\tar.exe" -a -cf $Zip $Name
    if ($LASTEXITCODE -ne 0) { throw "zip failed" }
    $hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower()
    Set-Content -Encoding ascii "$Zip.sha256" "$hash  $Zip"
} finally { Pop-Location }
Get-Item "$OutDir\$Zip" | Select-Object Name, Length
