# Contributing to Depth Surge 3D

Thank you for your interest in contributing! This guide is organized by contributor type.

---

## 👥 For Human Contributors

**TL;DR:** Don't stress about the strict rules below - they're mainly for our AI coding assistants. Submit your PR even if some tests fail or formatting is off. We appreciate your contribution and may have bots tidy things up later.

### Philosophy

The strict code quality requirements (Black, Flake8, complexity limits, etc.) are **primarily enforced for AI contributors** to maintain consistency and prevent AI-generated code bloat.

**As a human contributor:**
- ✅ Your expertise and ideas are valued
- ✅ It's OK to submit a PR that doesn't pass all checks
- ✅ We understand manual coding is different from AI-generated code
- ⚠️ Your PR might be refactored by AI tools after merge
- ⚠️ Code may be restructured multiple times as patterns evolve

We'd rather have your contribution with imperfect formatting than no contribution at all. The bots can handle the cleanup.

### Quick Start

```bash
git clone https://github.com/Tok/depth-surge-3d.git
cd depth-surge-3d
./setup.sh
./test.sh  # Verify setup
```

### Development Workflow

1. **Create a branch** from `dev`:
   ```bash
   git checkout dev
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code that works
   - Add comments where logic isn't obvious
   - Try to follow existing patterns when possible

3. **Test your changes** (if you can):
   ```bash
   ./test.sh                        # Basic verification
   pytest tests/unit -v             # Run unit tests (optional)
   ```

4. **Optional formatting** (helpful but not required):
   ```bash
   black src/ tests/                # Auto-format code
   flake8 src/ tests/               # Check for issues
   ```

5. **Submit PR** to `dev` branch:
   - Describe what you changed and why
   - Mention any tests you ran
   - Note any checks that fail (we'll handle it)

### Commit Message Format

Try to follow this format (but don't stress if you forget):

```
type: brief description

Detailed explanation if needed.
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Example:**
```
fix: handle null progress_tracker in CLI mode

Added null checks before calling update_progress() to prevent
crashes when running without web interface.

Fixes #14
```

### What Happens After You Submit?

1. CI runs automatically (tests, formatting checks)
2. Maintainers review your changes
3. If CI fails:
   - We may fix minor issues ourselves
   - We may ask you to update (with specific guidance)
   - We may merge and let AI tools clean up formatting
4. Your contribution gets merged!
5. Bots may refactor code later (don't take it personally!)

### Areas Where We'd Love Help

- **Bug reports** with reproduction steps
- **Performance optimizations** (especially GPU memory)
- **Documentation improvements** (examples, guides, typo fixes)
- **New VR headset presets** (if you have specific hardware)
- **Test coverage** for edge cases you encounter

### Questions?

- **Stuck?** Open an issue or discussion
- **Not sure if something is a bug?** Open an issue anyway
- **Want to add a feature?** Propose it in an issue first
- **Documentation unclear?** Let us know what confused you

---

## 🤖 For AI Contributors (Claude, GPT, etc.)

**This section is for AI coding assistants and autonomous agents.**

### Required Reading

All AI contributors MUST read and follow **[docs/CLAUDE.md](CLAUDE.md)** for:
- Code quality requirements (Black, Flake8, complexity limits)
- Testing requirements (unit tests, coverage ≥85%)
- Development workflow
- Commit message format
- Architecture guidelines

**Note:** While `CLAUDE.md` is named for Claude Code, it's written generically and can be used by any AI coding assistant (GPT Engineer, Cursor, Aider, etc.).

### Strict Requirements for AI

Unlike human contributors, AI tools are expected to:
- ✅ **Format all code** with Black (line length: 100)
- ✅ **Pass all Flake8 checks** (complexity ≤10, max line: 127)
- ✅ **Maintain coverage** ≥85% for all new code
- ✅ **Add type hints** to all functions
- ✅ **Write docstrings** for public functions
- ✅ **Run pre-commit checks** before every commit:
  ```bash
  ./scripts/pre-commit-checks.sh
  ```

### Why Stricter for AI?

AI can:
- Generate perfectly formatted code consistently
- Write comprehensive tests automatically
- Refactor without fatigue
- Follow complex rules without cognitive load
- Self-check against multiple criteria

Humans shouldn't be held to the same standard because manual coding has different constraints.

### AI Workflow

1. **Read CLAUDE.md** thoroughly before starting
2. **Check git status** to understand current state
3. **Run tests** before making changes
4. **Make changes** following all quality rules
5. **Run pre-commit checks**:
   ```bash
   ./scripts/pre-commit-checks.sh
   ```
6. **Verify all checks pass** (no exceptions)
7. **Commit with proper format**:
   ```bash
   git commit -m "type: description

   Co-Authored-By: AI Assistant <ai@example.com>"
   ```

### AI-Specific Guidelines

- **Don't skip steps** - run all checks every time
- **Don't commit failing code** - fix issues before committing
- **Don't add unnecessary complexity** - keep functions simple (≤10 complexity)
- **Don't leave TODOs** - complete tasks fully or don't start them
- **Don't break existing tests** - ensure all 770+ tests still pass
- **Do refactor existing code** - improve what you touch
- **Do maintain consistency** - follow existing patterns
- **Do write comprehensive docstrings** - explain the "why", not just the "what"

### Integration with Other AI Tools

If you're not Claude Code but another AI assistant:

1. **Read CLAUDE.md** - it's tool-agnostic
2. **Adapt to your workflow** - use your own commands for running tests
3. **Follow the same standards** - formatting, complexity, coverage
4. **Report issues** - if guidelines are unclear, open an issue

### Pre-Commit Checklist for AI

Before every commit, verify:
- [ ] Black formatting applied to all changed files
- [ ] Flake8 reports 0 errors (warnings OK)
- [ ] All unit tests pass
- [ ] Coverage ≥85% for changed modules
- [ ] Type hints added to all new functions
- [ ] Docstrings added to all public functions
- [ ] No functions with complexity >10
- [ ] Commit message follows conventional format

---

## Development Resources

### Documentation Structure

- **[CLAUDE.md](CLAUDE.md)** - Development guide (AI-focused but useful for all)
- **[CODING_GUIDE.md](CODING_GUIDE.md)** - Detailed coding standards
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design overview
- **[INSTALLATION.md](INSTALLATION.md)** - Setup instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues
- **[TODO.md](TODO.md)** - Roadmap and planned features

### Running Tests

```bash
# Quick verification (end users)
./test.sh

# Unit tests with coverage (developers)
./scripts/run-unit-tests.sh

# Pre-commit checks (required for AI)
./scripts/pre-commit-checks.sh
```

### Code Quality Tools

```bash
# Format code
black src/ tests/ app.py

# Lint code
flake8 src/ tests/ app.py --count --show-source --statistics

# Type checking
mypy src/depth_surge_3d/ --ignore-missing-imports

# Find complex functions
radon cc src/depth_surge_3d/ -a -nc

# Find dead code
vulture src/depth_surge_3d/
```

---

## Questions or Issues?

- **Bug reports**: Open an issue with minimal reproduction steps
- **Feature requests**: Open an issue with use case description
- **Documentation issues**: PR fixes directly or open an issue
- **General questions**: Check docs first, then open a discussion

**Thank you for contributing to Depth Surge 3D!**

Whether you're human or AI, we appreciate your help in making this project better. 🚀
