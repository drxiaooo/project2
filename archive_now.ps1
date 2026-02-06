param(
  [switch]$do  # 默认不传就是 Dry-Run；传了 -do 才会真的移动
)

$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host $msg }

# ---- config ----
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$ARCH_ROOT = "archive\$TS"
$ARCH_SRC  = Join-Path $ARCH_ROOT "src"
$ARCH_OUT  = Join-Path $ARCH_ROOT "outputs"

# 当前你明确要保留的 outputs（旧结果 + 当前 weakloc + splits + index）
$KEEP_OUT_DIRS = @(
  "outputs\cls\BEST",
  "outputs\cls\resnetrs50_POST",
  "outputs\splits",
  "outputs\weakloc"
)
$KEEP_OUT_FILES = @(
  "outputs\bs80k_index.csv",
  "outputs\cls\leaderboard.csv"
)

# 当前你明确要归档的 src（暂时/后续都不用）
$ARCHIVE_SRC_FILES = @(
  "src\cls\infer_split_resnet50.py",
  "src\cls\find_threshold.py"
)

# ---- sanity ----
if (!(Test-Path "src") -or !(Test-Path "outputs")) {
  throw "Please run from project root (where src/ and outputs/ exist)."
}

Say "==[ARCHIVE] mode: $(if($do){'DO (move files)'} else {'DRY-RUN (print only)'}) =="
Say "Archive target: $ARCH_ROOT"
Say ""

# ---- prepare dirs (only when do) ----
if ($do) {
  New-Item -ItemType Directory -Force -Path $ARCH_SRC | Out-Null
  New-Item -ItemType Directory -Force -Path $ARCH_OUT | Out-Null
}

# ---- 1) src files ----
Say "==[1/2] src files to archive =="
foreach ($p in $ARCHIVE_SRC_FILES) {
  if (Test-Path $p) {
    Say "  ARCHIVE: $p"
    if ($do) {
      $destDir = Split-Path (Join-Path $ARCH_SRC $p) -Parent
      New-Item -ItemType Directory -Force -Path $destDir | Out-Null
      Move-Item -Force $p (Join-Path $ARCH_SRC $p)
    }
  } else {
    Say "  SKIP (not found): $p"
  }
}
Say ""

# ---- 2) outputs top-level items ----
Say "==[2/2] outputs items to archive (except keep list) =="

# resolve keep paths to absolute for robust compare
$KEEP_ABS_DIRS = @()
foreach ($k in $KEEP_OUT_DIRS) {
  if (Test-Path $k) { $KEEP_ABS_DIRS += (Resolve-Path $k).Path }
  else { Say "  WARN keep-dir not found: $k" }
}
$KEEP_ABS_FILES = @()
foreach ($kf in $KEEP_OUT_FILES) {
  if (Test-Path $kf) { $KEEP_ABS_FILES += (Resolve-Path $kf).Path }
  else { Say "  WARN keep-file not found: $kf" }
}

$OUT_ROOT_ABS = (Resolve-Path "outputs").Path
$OUT_ITEMS = Get-ChildItem -Path "outputs" -Force

foreach ($it in $OUT_ITEMS) {
  $abs = (Resolve-Path $it.FullName).Path
  $keep = $false

  foreach ($kd in $KEEP_ABS_DIRS) {
    if ($abs -like ($kd + "*")) { $keep = $true; break }
  }
  if (!$keep) {
    foreach ($kf in $KEEP_ABS_FILES) {
      if ($abs -eq $kf) { $keep = $true; break }
    }
  }

  if ($keep) {
    Say "  KEEP: $($it.FullName)"
    continue
  }

  $rel = $it.FullName.Substring($OUT_ROOT_ABS.Length).TrimStart("\")
  $dest = Join-Path $ARCH_OUT $rel

  Say "  ARCHIVE: outputs\$rel  ->  $dest"

  if ($do) {
    $destDir = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    Move-Item -Force $it.FullName $dest
  }
}

Say ""
if (!$do) {
  Say "DRY-RUN finished. If everything looks correct, run:"
  Say "  powershell -ExecutionPolicy Bypass -File .\archive_now.ps1 -do"
} else {
  # write restore script
  $restorePath = Join-Path $ARCH_ROOT "restore_$TS.ps1"
  $restore = @"
param()

`$ErrorActionPreference = "Stop"
Write-Host "==[RESTORE] from $ARCH_ROOT =="

# Restore src
if (Test-Path "$ARCH_SRC\src") {
  Get-ChildItem -Recurse -File "$ARCH_SRC\src" | ForEach-Object {
    `$rel = `$_.FullName.Substring((Resolve-Path "$ARCH_SRC\src").Path.Length).TrimStart("\")
    `$dst = Join-Path "." ("src\" + `$rel)
    `$dstDir = Split-Path `$dst -Parent
    New-Item -ItemType Directory -Force -Path `$dstDir | Out-Null
    Move-Item -Force `$_.FullName `$dst
    Write-Host "  restored: `$dst"
  }
}

# Restore outputs (top-level moved items)
if (Test-Path "$ARCH_OUT") {
  Get-ChildItem -Force "$ARCH_OUT" | ForEach-Object {
    `$dst = Join-Path "." ("outputs\" + `$_.Name)
    Move-Item -Force `$_.FullName `$dst
    Write-Host "  restored: `$dst"
  }
}

Write-Host "==[RESTORE] done =="
"@
  $restore | Out-File -Encoding UTF8 $restorePath
  Say "DONE. Archived to: $ARCH_ROOT"
  Say "Restore script: $restorePath"
  Say "To restore: powershell -ExecutionPolicy Bypass -File $restorePath"
}
