"""
Console output utilities for colored and formatted terminal text.

Provides ANSI color codes for terminal output matching the UI theme.
"""


# ANSI color codes
class Colors:
    """ANSI color codes matching the UI theme."""

    # UI theme colors
    LIME_GREEN = "\033[38;2;57;255;20m"  # --accent-lime: #39ff14
    GREEN = "\033[38;2;0;255;65m"  # --accent-green: #00ff41
    ELECTRIC_BLUE = "\033[38;2;0;217;255m"  # --accent-blue: #00d9ff (progress bar blue)

    # Standard colors
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"

    # Formatting
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def disable():
        """Disable all colors (for piped output or when colors not supported)."""
        Colors.LIME_GREEN = ""
        Colors.GREEN = ""
        Colors.ELECTRIC_BLUE = ""
        Colors.RED = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.GRAY = ""
        Colors.BOLD = ""
        Colors.DIM = ""
        Colors.RESET = ""


def success(text: str) -> str:
    """Format text as success message (lime green)."""
    return f"{Colors.LIME_GREEN}{text}{Colors.RESET}"


def error(text: str) -> str:
    """Format text as error message (red)."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def warning(text: str) -> str:
    """Format text as warning message (yellow)."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def info(text: str) -> str:
    """Format text as info message (blue)."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def dim(text: str) -> str:
    """Format text as dimmed/secondary (gray)."""
    return f"{Colors.GRAY}{text}{Colors.RESET}"


def bold(text: str) -> str:
    """Format text as bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def step_complete(text: str) -> str:
    """Format step completion line with lime green arrow."""
    return f"{Colors.LIME_GREEN}  -> {Colors.RESET}{text}"


def saved_to(text: str) -> str:
    """Format 'Saved to' line with electric blue arrow."""
    return f"{Colors.ELECTRIC_BLUE}  -> {Colors.RESET}{text}"


def title_bar(text: str) -> str:
    """Format title with lime green === markers."""
    # Extract the === markers and color them
    if text.startswith("=== ") and text.endswith(" ==="):
        content = text[4:-4]  # Extract content without === markers
        return (
            f"{Colors.LIME_GREEN}==={Colors.RESET} {content} {Colors.LIME_GREEN}==={Colors.RESET}"
        )
    return text
