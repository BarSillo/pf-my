import os
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # For Python <3.11; install via: pip install tomli

import pfhedge

def test_version():
    # Construct the path to pyproject.toml
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    
    # Open and load the TOML file
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
    
    # Extract the version from the TOML file
    version = config["tool"]["poetry"]["version"]
    
    # Compare it to the package version
    assert pfhedge.__version__ == version, (
        f"pfhedge.__version__ ({pfhedge.__version__}) does not match pyproject.toml version ({version})"
    )
