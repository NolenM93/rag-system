"""
test_setup.py - Validate RAG system setup and dependencies
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies:")
    
    required_packages = {
        'openai': 'OpenAI API client',
        'dotenv': 'Environment variable loader',
        'langchain': 'LangChain framework',
        'sentence_transformers': 'Sentence Transformers',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'faiss': 'FAISS vector search',
        'transformers': 'Hugging Face Transformers',
        'requests': 'HTTP library',
        'bs4': 'BeautifulSoup',
    }
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'faiss':
                __import__('faiss')
            else:
                __import__(package)
            print(f"  ✓ {description} ({package})")
        except ImportError:
            print(f"  ✗ {description} ({package}) - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_files():
    """Check if required files exist"""
    print("\nChecking project files:")
    
    required_files = {
        'text_extractor.py': 'Document extraction script',
        'RAG_app.py': 'Main RAG application',
        'requirements.txt': 'Dependencies list',
        '.env': 'Environment variables (API keys)',
        '.gitignore': 'Git ignore rules',
        'README.md': 'Documentation',
    }
    
    optional_files = {
        'Selected_Document.txt': 'Extracted document',
    }
    
    all_present = True
    for filename, description in required_files.items():
        if Path(filename).exists():
            print(f"  ✓ {description} ({filename})")
        else:
            print(f"  ✗ {description} ({filename}) - MISSING")
            all_present = False
    
    for filename, description in optional_files.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"  ✓ {description} ({filename}) - {size:,} bytes")
        else:
            print(f"  ⚠ {description} ({filename}) - not yet created")
    
    return all_present


def check_env_file():
    """Check if .env file has API key configured"""
    print("\nChecking API key configuration:")
    
    if not Path('.env').exists():
        print("  ✗ .env file not found")
        print("    Create .env and add: OPENAI_API_KEY=your-key-here")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'OPENAI_API_KEY' not in content:
        print("  ✗ OPENAI_API_KEY not found in .env")
        return False
    
    # Check if placeholder value
    if 'your-api-key-here' in content or 'sk-' not in content:
        print("  ⚠ OPENAI_API_KEY appears to be placeholder")
        print("    Update .env with your actual OpenAI API key")
        return False
    
    print("  ✓ OPENAI_API_KEY configured in .env")
    return True


def check_document():
    """Check if document has been extracted"""
    print("\nChecking extracted document:")
    
    if not Path('Selected_Document.txt').exists():
        print("  ✗ Selected_Document.txt not found")
        print("    Run: python text_extractor.py")
        return False
    
    size = Path('Selected_Document.txt').stat().st_size
    if size < 1000:
        print(f"  ⚠ Selected_Document.txt seems too small ({size} bytes)")
        print("    Run: python text_extractor.py")
        return False
    
    # Read first line to check content
    with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    print(f"  ✓ Selected_Document.txt exists ({size:,} bytes)")
    print(f"    First line: {first_line[:80]}...")
    return True


def main():
    """Run all checks"""
    print("="*70)
    print("RAG SYSTEM SETUP VALIDATION")
    print("="*70)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Files': check_files(),
        'API Key': check_env_file(),
        'Document': check_document(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {check_name}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All checks passed! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Ensure .env has your actual OpenAI API key")
        print("2. Run: python RAG_app.py")
        print("3. Ask questions about Artificial Intelligence")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not checks['Dependencies']:
            print("- Install dependencies: pip install -r requirements.txt")
        if not checks['API Key']:
            print("- Add your API key to .env file")
        if not checks['Document']:
            print("- Extract document: python text_extractor.py")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
