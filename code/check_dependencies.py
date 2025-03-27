#!/usr/bin/env python3
"""
Dependency Checker for Track 1 and Track 2 Grid Search
This script checks if all required packages are installed and provides installation commands if needed.
"""
import importlib
import sys
import subprocess
import pkg_resources

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "numpy": "1.20.0",
    "pandas": "1.2.0",
    "scikit-learn": "0.24.0",
    "nltk": "3.6.0",
    "tqdm": "4.50.0",
    "gensim": "4.0.0"
}

# Additional packages for Track 2
TRACK2_PACKAGES = {
    "multiprocessing": None,  # Part of standard library
}

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets minimum version requirements."""
    try:
        # Try to import the package
        package = importlib.import_module(package_name)
        
        # If minimum version specified, check version
        if min_version:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                meets_version = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version)
                return True, installed_version, meets_version
            except Exception as e:
                return True, "unknown", False
        
        return True, getattr(package, "__version__", "unknown"), True
    
    except ImportError:
        return False, None, False

def print_report(package, installed, version, meets_min):
    """Print formatted report about package status."""
    min_version = REQUIRED_PACKAGES.get(package) or "any"
    
    if installed and meets_min:
        print(f"✓ {package:<15} - Installed (version: {version})")
    elif installed and not meets_min:
        print(f"! {package:<15} - Installed (version: {version}) but minimum required is {min_version}")
    else:
        print(f"✗ {package:<15} - Not installed (required version: {min_version})")

def main():
    """Main function to check dependencies."""
    print("\n===== Checking dependencies for Track 1 and Track 2 Grid Search =====\n")
    
    all_packages = {**REQUIRED_PACKAGES, **TRACK2_PACKAGES}
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in all_packages.items():
        installed, version, meets_min = check_package(package, min_version)
        print_report(package, installed, version, meets_min)
        
        if not installed:
            missing_packages.append(package)
        elif not meets_min:
            outdated_packages.append(package)
    
    # Special check for NLTK data
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK data     - Punkt tokenizer installed")
        except LookupError:
            print("! NLTK data     - Punkt tokenizer not installed")
            # Don't add to missing packages as BLEU score calculation might still work
    except:
        # NLTK itself is already in the list if missing
        pass
    
    # Print installation instructions if needed
    if missing_packages or outdated_packages:
        print("\n===== Installation Instructions =====\n")
        
        if missing_packages:
            cmd = "pip install " + " ".join(missing_packages)
            print(f"To install missing packages, run:\n{cmd}\n")
        
        if outdated_packages:
            cmd = "pip install --upgrade " + " ".join(outdated_packages)
            print(f"To upgrade outdated packages, run:\n{cmd}\n")
        
        if "nltk" in missing_packages or "nltk" in outdated_packages:
            print("After installing NLTK, run the following to download required data:")
            print("import nltk\nnltk.download('punkt')")
    else:
        print("\n✓ All dependencies are installed correctly!\n")
    
    print("\nYou can now run the grid search scripts:")
    print("python code/track_1_search.py")
    print("python code/track_2_search.py")

if __name__ == "__main__":
    main()