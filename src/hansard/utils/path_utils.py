"""
Universal path utilities for Hansard project.
Ensures scripts work from any directory.
"""

from pathlib import Path
import sys


def find_project_root():
    """
    Find the project root directory by looking for marker files.
    Works from any subdirectory within the project.
    """
    current = Path.cwd()

    # Look for these marker files that indicate project root
    markers = ['pyproject.toml', 'environment.yml', 'README.md', '.git']

    # Check current directory and up to 5 levels up
    for _ in range(6):
        if any((current / marker).exists() for marker in markers):
            # Double check this looks like our project
            if (current / 'src' / 'hansard').exists():
                return current

        parent = current.parent
        if parent == current:
            break
        current = parent

    # If we can't find it, assume we're in a subdirectory
    # Try common patterns
    if 'hansard-nlp-explorer' in str(Path.cwd()):
        parts = str(Path.cwd()).split('hansard-nlp-explorer')
        root = Path(parts[0] + 'hansard-nlp-explorer')
        if root.exists():
            return root

    raise RuntimeError("Cannot find project root. Please run from within the hansard-nlp-explorer directory")


def get_data_dir():
    """Get the data directory path that works from anywhere."""
    root = find_project_root()

    # Check multiple possible locations
    candidates = [
        root / 'data',  # Project root level data
        root / 'src' / 'hansard' / 'data',  # Old location
    ]

    for data_dir in candidates:
        if data_dir.exists():
            return data_dir

    raise RuntimeError(f"Data directory not found. Tried: {candidates}")


def get_processed_data_dir():
    """Get processed data directory, checking for alternate names."""
    data_dir = get_data_dir()

    # Check for processed_fixed first (current name)
    if (data_dir / 'processed_fixed').exists():
        return data_dir / 'processed_fixed'

    # Fall back to processed
    if (data_dir / 'processed').exists():
        return data_dir / 'processed'

    raise RuntimeError(f"No processed data directory found in {data_dir}")


def get_gender_wordlists_dir():
    """Get gender wordlists directory."""
    data_dir = get_data_dir()
    wordlists_dir = data_dir / 'gender_wordlists'

    if not wordlists_dir.exists():
        raise RuntimeError(f"Gender wordlists not found at {wordlists_dir}")

    return wordlists_dir


def get_analysis_output_dir():
    """Get or create analysis output directory."""
    root = find_project_root()
    output_dir = root / 'analysis_output'
    output_dir.mkdir(exist_ok=True)
    return output_dir


def add_src_to_path():
    """Add src directory to Python path for imports."""
    root = find_project_root()
    src_path = str(root / 'src')

    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def resolve_path(path_str, base='project'):
    """
    Resolve a path string to an absolute path.

    Args:
        path_str: Path string (can be relative or absolute)
        base: Base for relative paths ('project', 'data', 'cwd')

    Returns:
        Absolute Path object
    """
    path = Path(path_str)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Otherwise resolve based on base
    if base == 'project':
        return find_project_root() / path
    elif base == 'data':
        return get_data_dir() / path
    elif base == 'cwd':
        return Path.cwd() / path
    else:
        raise ValueError(f"Unknown base: {base}")


# Example usage in scripts:
if __name__ == "__main__":
    print("Testing path utilities...")

    try:
        print(f"Project root: {find_project_root()}")
        print(f"Data dir: {get_data_dir()}")
        print(f"Processed data: {get_processed_data_dir()}")
        print(f"Gender wordlists: {get_gender_wordlists_dir()}")
        print(f"Analysis output: {get_analysis_output_dir()}")
        print("\nAll paths resolved successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)