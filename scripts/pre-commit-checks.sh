#!/bin/bash

echo "🔍 Depth Surge 3D - Pre-Commit Quality Gate"
echo "============================================"
echo ""

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment not found."
    echo "   Run ./setup.sh first to create a virtual environment."
    exit 1
fi

# Track overall status
OVERALL_STATUS=0

# =============================================================================
# Step 1: Black Formatting Check
# =============================================================================
echo "📝 Step 1/4: Checking code formatting (black)..."
echo "   Command: black --check src/ tests/ app.py"
echo ""

black --check src/ tests/ app.py

BLACK_EXIT=$?
if [ $BLACK_EXIT -eq 0 ]; then
    echo "✅ Black formatting check passed"
else
    echo "❌ Black formatting check failed"
    echo ""
    echo "💡 To fix formatting issues, run:"
    echo "   black src/ tests/ app.py"
    OVERALL_STATUS=1
fi

echo ""
echo "---"
echo ""

# =============================================================================
# Step 2: Flake8 Linting
# =============================================================================
echo "🔎 Step 2/4: Running code linting (flake8)..."
echo "   Command: flake8 src/ tests/ app.py --count --show-source --statistics"
echo ""

flake8 src/ tests/ app.py --count --show-source --statistics

FLAKE8_EXIT=$?
if [ $FLAKE8_EXIT -eq 0 ]; then
    echo "✅ Flake8 linting passed (0 errors)"
else
    echo "❌ Flake8 linting failed"
    echo ""
    echo "💡 Fix all linting errors above before committing"
    OVERALL_STATUS=1
fi

echo ""
echo "---"
echo ""

# =============================================================================
# Step 3: Type Checking (mypy)
# =============================================================================
echo "🔍 Step 3/4: Running type checking (mypy)..."
echo "   Command: mypy src/depth_surge_3d --ignore-missing-imports"
echo ""

mypy src/depth_surge_3d --ignore-missing-imports

MYPY_EXIT=$?
if [ $MYPY_EXIT -eq 0 ]; then
    echo "✅ Mypy type checking passed (0 errors)"
else
    echo "❌ Mypy type checking failed"
    echo ""
    echo "💡 Fix all type errors above before committing"
    OVERALL_STATUS=1
fi

echo ""
echo "---"
echo ""

# =============================================================================
# Step 4: Unit Tests with Coverage
# =============================================================================
echo "🧪 Step 4/4: Running unit tests with coverage..."
echo "   Command: pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=term --cov-fail-under=85"
echo ""

pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=term --cov-fail-under=85

PYTEST_EXIT=$?
if [ $PYTEST_EXIT -eq 0 ]; then
    echo "✅ Unit tests passed with coverage ≥ 85%"
else
    echo "❌ Unit tests failed or coverage < 85%"
    OVERALL_STATUS=1
fi

echo ""
echo "============================================"
echo ""

# =============================================================================
# Final Summary
# =============================================================================
if [ $OVERALL_STATUS -eq 0 ]; then
    echo "✅ All pre-commit checks passed!"
    echo ""
    echo "📦 Your code is ready to commit:"
    echo "   git add ."
    echo "   git commit -m \"your message\""
    echo ""
else
    echo "❌ Pre-commit checks failed"
    echo ""
    echo "🚫 DO NOT COMMIT until all checks pass"
    echo ""
    echo "Summary:"
    [ $BLACK_EXIT -ne 0 ] && echo "   ❌ Black formatting"
    [ $FLAKE8_EXIT -ne 0 ] && echo "   ❌ Flake8 linting"
    [ $MYPY_EXIT -ne 0 ] && echo "   ❌ Type checking (mypy)"
    [ $PYTEST_EXIT -ne 0 ] && echo "   ❌ Unit tests/coverage"
    echo ""
    echo "💡 Fix the issues above and run this script again"
    echo ""
fi

exit $OVERALL_STATUS
