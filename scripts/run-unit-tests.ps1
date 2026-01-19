# Depth Surge 3D - Unit Test Runner (PowerShell)
Write-Host "🧪 Depth Surge 3D - Unit Test Runner" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists and activate it
$venvActivated = $false

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment (.venv)..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
    $venvActivated = $true
}
elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment (venv)..." -ForegroundColor Yellow
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

# Check if pytest is installed
try {
    python -c "import pytest" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "pytest not found"
    }
}
catch {
    Write-Host "❌ Error: pytest not installed in virtual environment." -ForegroundColor Red
    Write-Host "   Run: pip install pytest pytest-cov" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Check if pytest-cov is available for coverage
$hasCov = $false
try {
    python -c "import pytest_cov" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $hasCov = $true
    }
}
catch {}

# Parse command line arguments
$pytestArgs = "-v tests/unit"
$showCoverage = $true  # Show coverage by default

foreach ($arg in $args) {
    switch ($arg) {
        { $_ -in "--coverage", "-c" } {
            $showCoverage = $true
        }
        { $_ -in "--verbose", "-v" } {
            $pytestArgs += " -vv"
        }
        { $_ -in "--help", "-h" } {
            Write-Host "Usage: .\scripts\run-unit-tests.ps1 [OPTIONS]"
            Write-Host ""
            Write-Host "Options:"
            Write-Host "  -c, --coverage    Show coverage report (requires pytest-cov)"
            Write-Host "  -v, --verbose     Verbose output"
            Write-Host "  -h, --help        Show this help message"
            Write-Host ""
            Write-Host "Examples:"
            Write-Host "  .\scripts\run-unit-tests.ps1              # Run unit tests"
            Write-Host "  .\scripts\run-unit-tests.ps1 --coverage   # Run with coverage"
            Write-Host ""
            exit 0
        }
        default {
            # Pass through any other arguments to pytest
            $pytestArgs += " $arg"
        }
    }
}

# Add coverage args if requested and available
if ($showCoverage) {
    if ($hasCov) {
        Write-Host "📊 Coverage reporting enabled" -ForegroundColor Yellow
        $pytestArgs += " --cov=src/depth_surge_3d --cov-report=term-missing --cov-report=html"
    }
    else {
        Write-Host "⚠️  Warning: pytest-cov not installed. Running without coverage." -ForegroundColor Yellow
        Write-Host "   Install with: pip install pytest-cov" -ForegroundColor Yellow
        Write-Host ""
    }
}

# Run the tests
Write-Host "🚀 Running unit tests..." -ForegroundColor Cyan
Write-Host "   Command: pytest $pytestArgs"
Write-Host ""

# Execute pytest with arguments
$command = "pytest " + $pytestArgs
Invoke-Expression $command

# Capture exit code
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "✅ All unit tests passed!" -ForegroundColor Green

    if ($showCoverage -and $hasCov) {
        Write-Host ""
        Write-Host "📊 Coverage report generated:" -ForegroundColor Yellow
        Write-Host "   Terminal: See above"
        Write-Host "   HTML: htmlcov\index.html"
        Write-Host ""
        Write-Host "💡 Open coverage report:" -ForegroundColor Yellow
        Write-Host "   start htmlcov\index.html  (Windows)"
    }
}
else {
    Write-Host "❌ Some unit tests failed (exit code: $exitCode)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Tips:" -ForegroundColor Yellow
    Write-Host "   - Review the test output above for details"
    Write-Host "   - Run with --verbose for more information"
    Write-Host "   - Check that all dependencies are installed"
}

Write-Host ""
exit $exitCode
