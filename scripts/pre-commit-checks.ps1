# Depth Surge 3D - Pre-Commit Quality Gate (PowerShell)
Write-Host "🔍 Depth Surge 3D - Pre-Commit Quality Gate" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists and activate it
$venvActivated = $false

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
    $venvActivated = $true
}
elseif (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    $venvActivated = $true
}
else {
    Write-Host "❌ Error: Virtual environment not found." -ForegroundColor Red
    Write-Host "   Run .\setup.sh first to create a virtual environment." -ForegroundColor Red
    exit 1
}

if (-not $venvActivated) {
    Write-Host "❌ Error: Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

# Track overall status
$overallStatus = 0

# =============================================================================
# Step 1: Black Formatting Check
# =============================================================================
Write-Host "📝 Step 1/4: Checking code formatting (black)..." -ForegroundColor Yellow
Write-Host "   Command: black --check src/ tests/ app.py"
Write-Host ""

black --check src/ tests/ app.py

$blackExit = $LASTEXITCODE
if ($blackExit -eq 0) {
    Write-Host "✅ Black formatting check passed" -ForegroundColor Green
}
else {
    Write-Host "❌ Black formatting check failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 To fix formatting issues, run:" -ForegroundColor Yellow
    Write-Host "   black src/ tests/ app.py"
    $overallStatus = 1
}

Write-Host ""
Write-Host "---"
Write-Host ""

# =============================================================================
# Step 2: Flake8 Linting
# =============================================================================
Write-Host "🔎 Step 2/4: Running code linting (flake8)..." -ForegroundColor Yellow
Write-Host "   Command: flake8 src/ tests/ app.py --count --show-source --statistics"
Write-Host ""

flake8 src/ tests/ app.py --count --show-source --statistics

$flake8Exit = $LASTEXITCODE
if ($flake8Exit -eq 0) {
    Write-Host "✅ Flake8 linting passed (0 errors)" -ForegroundColor Green
}
else {
    Write-Host "❌ Flake8 linting failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Fix all linting errors above before committing" -ForegroundColor Yellow
    $overallStatus = 1
}

Write-Host ""
Write-Host "---"
Write-Host ""

# =============================================================================
# Step 3: Type Checking (mypy)
# =============================================================================
Write-Host "🔍 Step 3/4: Running type checking (mypy)..." -ForegroundColor Yellow
Write-Host "   Command: mypy src/depth_surge_3d --ignore-missing-imports"
Write-Host ""

mypy src/depth_surge_3d --ignore-missing-imports

$mypyExit = $LASTEXITCODE
if ($mypyExit -eq 0) {
    Write-Host "✅ Mypy type checking passed (0 errors)" -ForegroundColor Green
}
else {
    Write-Host "❌ Mypy type checking failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Fix all type errors above before committing" -ForegroundColor Yellow
    $overallStatus = 1
}

Write-Host ""
Write-Host "---"
Write-Host ""

# =============================================================================
# Step 4: Unit Tests with Coverage
# =============================================================================
Write-Host "🧪 Step 4/4: Running unit tests with coverage..." -ForegroundColor Yellow
Write-Host "   Command: pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=term --cov-fail-under=85"
Write-Host ""

pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=term --cov-fail-under=85

$pytestExit = $LASTEXITCODE
if ($pytestExit -eq 0) {
    Write-Host "✅ Unit tests passed with coverage ≥ 85%" -ForegroundColor Green
}
else {
    Write-Host "❌ Unit tests failed or coverage < 85%" -ForegroundColor Red
    $overallStatus = 1
}

Write-Host ""
Write-Host "============================================"
Write-Host ""

# =============================================================================
# Final Summary
# =============================================================================
if ($overallStatus -eq 0) {
    Write-Host "✅ All pre-commit checks passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📦 Your code is ready to commit:" -ForegroundColor Cyan
    Write-Host "   git add ."
    Write-Host "   git commit -m `"your message`""
    Write-Host ""
}
else {
    Write-Host "❌ Pre-commit checks failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "🚫 DO NOT COMMIT until all checks pass" -ForegroundColor Red
    Write-Host ""
    Write-Host "Summary:"
    if ($blackExit -ne 0) { Write-Host "   ❌ Black formatting" -ForegroundColor Red }
    if ($flake8Exit -ne 0) { Write-Host "   ❌ Flake8 linting" -ForegroundColor Red }
    if ($mypyExit -ne 0) { Write-Host "   ❌ Type checking (mypy)" -ForegroundColor Red }
    if ($pytestExit -ne 0) { Write-Host "   ❌ Unit tests/coverage" -ForegroundColor Red }
    Write-Host ""
    Write-Host "💡 Fix the issues above and run this script again" -ForegroundColor Yellow
    Write-Host ""
}

exit $overallStatus
