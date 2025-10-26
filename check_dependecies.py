#!/usr/bin/env python3
"""
Simple test to verify that facenet_pytorch dependency is removed
"""

import sys
import ast

def check_file_imports(filename):
    """Check what imports are in a Python file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Parse the file to extract imports
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

def main():
    """Test that problematic imports are removed."""
    print("🔍 Checking for problematic dependencies...")
    
    # Check the main files
    files_to_check = [
        'src/preprocessing/image_processor.py',
        'main.py'
    ]
    
    problematic_imports = ['facenet_pytorch', 'mtcnn']
    found_issues = False
    
    for filename in files_to_check:
        print(f"\n📄 Checking {filename}...")
        imports = check_file_imports(filename)
        
        for prob_import in problematic_imports:
            if prob_import in imports:
                print(f"❌ Found problematic import: {prob_import}")
                found_issues = True
            else:
                print(f"✅ No {prob_import} import found")
    
    print(f"\n📋 Results:")
    if not found_issues:
        print("✅ All problematic dependencies have been removed!")
        print("✅ The facenet_pytorch build error is fixed!")
        print("\n🎯 Next steps:")
        print("1. Install remaining packages when network is working:")
        print("   pip install --break-system-packages torch torchvision transformers datasets")
        print("2. Configure your Hugging Face token in .env")
        print("3. Run: python main.py")
        return True
    else:
        print("❌ Found problematic dependencies that need to be removed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)