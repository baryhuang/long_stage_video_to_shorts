#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that all dependencies are installed correctly.
"""

import sys
import importlib
import os
from dotenv import load_dotenv

# List of required packages
REQUIRED_PACKAGES = [
    "cv2",
    "numpy",
    "assemblyai",
    "anthropic",
    "moviepy.editor",
    "tqdm",
    "zhconv",
    "dotenv",
    "pydantic",
    "requests",
    "PIL"
]

def check_package(package_name):
    """Check if a package is installed and can be imported."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_api_keys():
    """Check if API keys are set in the environment."""
    load_dotenv()
    
    assembly_key = os.getenv("ASSEMBLY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not assembly_key:
        print("❌ ASSEMBLY_API_KEY not found in .env file")
        has_keys = False
    else:
        print("✅ ASSEMBLY_API_KEY found")
        has_keys = True
    
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found in .env file")
        has_keys = False
    else:
        print("✅ ANTHROPIC_API_KEY found")
        has_keys = has_keys and True
    
    return has_keys

def main():
    """Main function to check dependencies and API keys."""
    print("Testing setup for Long Video to Shorts...")
    print("\nChecking required packages:")
    
    all_packages_installed = True
    
    for package in REQUIRED_PACKAGES:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - Not found")
            all_packages_installed = False
    
    print("\nChecking API keys:")
    has_keys = check_api_keys()
    
    print("\nSummary:")
    if all_packages_installed:
        print("✅ All required packages are installed")
    else:
        print("❌ Some packages are missing. Please run 'pip install -r requirements.txt'")
    
    if has_keys:
        print("✅ API keys are set")
    else:
        print("❌ API keys are missing. Please edit the .env file and add your API keys")
    
    if all_packages_installed and has_keys:
        print("\n🎉 Setup is complete! You can now run the highlight generator.")
        print("To run the highlight generator, use: python highlight_generator.py <input_video>")
        return 0
    else:
        print("\n❌ Setup is incomplete. Please fix the issues above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 