#!/usr/bin/env python3
"""
Playwright Setup Script for Betika Virtual Games Prediction Model

This script installs Playwright and its browser dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Playwright for Betika Virtual Games Prediction Model")
    print("=" * 60)
    
    # Check if playwright is installed
    try:
        import playwright
        print("✅ Playwright is already installed")
    except ImportError:
        print("📦 Installing Playwright...")
        if not run_command("pip install playwright", "Playwright installation"):
            print("❌ Failed to install Playwright. Please install manually:")
            print("   pip install playwright")
            sys.exit(1)
    
    # Install browser binaries
    print("\n📥 Installing browser binaries...")
    if not run_command("python3 -m playwright install chromium", "Chromium browser installation"):
        print("❌ Failed to install Chromium. You may need to:")
        print("   1. Ensure you have sufficient disk space (>500MB)")
        print("   2. Check your internet connection")
        print("   3. Run manually: python3 -m playwright install chromium")
        sys.exit(1)
    
    # Test installation
    print("\n🧪 Testing Playwright installation...")
    test_script = '''
import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("data:text/html,<h1>Test</h1>")
        title = await page.inner_text("h1")
        await browser.close()
        return title == "Test"

result = asyncio.run(test())
print("SUCCESS" if result else "FAILED")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("✅ Playwright test passed!")
        else:
            print("❌ Playwright test failed")
            print(f"   Output: {result.stdout}")
            print(f"   Error: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ Playwright test timed out")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error testing Playwright: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Playwright setup completed successfully!")
    print("\nYou can now run:")
    print("   python3 main.py collect --once")
    print("   python3 main.py serve")
    print("\nFor more commands, run:")
    print("   python3 main.py --help")

if __name__ == "__main__":
    main()