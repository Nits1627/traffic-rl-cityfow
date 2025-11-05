#!/usr/bin/env python3
"""
Test script to validate the Traffic RL SUMO installation.

This script checks if all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.10+")
        return False

def test_sumo_installation():
    """Test SUMO installation."""
    print("🔍 Testing SUMO installation...")
    
    # Check SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("   ❌ SUMO_HOME environment variable not set")
        return False
    
    if not os.path.exists(sumo_home):
        print(f"   ❌ SUMO_HOME directory not found: {sumo_home}")
        return False
    
    # Check SUMO binaries
    sumo_bin = os.path.join(sumo_home, 'bin')
    if not os.path.exists(sumo_bin):
        print(f"   ❌ SUMO bin directory not found: {sumo_bin}")
        return False
    
    # Test SUMO execution
    try:
        import subprocess
        result = subprocess.run(
            [os.path.join(sumo_bin, 'sumo'), '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   ✅ SUMO found: {sumo_home}")
            print(f"   ✅ Version: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ SUMO execution failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing SUMO: {e}")
        return False

def test_python_packages():
    """Test Python package imports."""
    print("📦 Testing Python packages...")
    
    packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('gymnasium', 'gymnasium'),
        ('stable_baselines3', 'stable_baselines3'),
    ]
    
    all_good = True
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError as e:
            print(f"   ❌ {package_name}: {e}")
            all_good = False
    
    return all_good

def test_project_structure():
    """Test project directory structure."""
    print("📁 Testing project structure...")
    
    required_dirs = ['scripts', 'traffic_rl', 'network', 'routes', 'sumo_configs', 'output', 'models', 'logs']
    required_files = [
        'scripts/generate_network.py',
        'scripts/train.py',
        'scripts/evaluate.py',
        'scripts/launch_gui.py',
        'traffic_rl/__init__.py',
        'traffic_rl/env.py',
        'traffic_rl/utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - Missing")
            all_good = False
    
    # Check files
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - Missing")
            all_good = False
    
    return all_good

def test_network_generation():
    """Test network generation."""
    print("🌐 Testing network generation...")
    
    try:
        from traffic_rl.utils import check_sumo_installation
        if not check_sumo_installation():
            print("   ❌ SUMO installation check failed")
            return False
        
        # Try to generate network
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/generate_network.py'],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("   ✅ Network generation successful")
            return True
        else:
            print(f"   ❌ Network generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Network generation error: {e}")
        return False

def test_environment_creation():
    """Test environment creation."""
    print("🤖 Testing environment creation...")
    
    try:
        from traffic_rl.env import SumoTrafficLightEnv
        
        config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
        if not config_path.exists():
            print("   ❌ SUMO config file not found - run generate_network.py first")
            return False
        
        # Try to create environment (this will test imports and basic setup)
        env = SumoTrafficLightEnv(
            config_path=str(config_path),
            max_steps=100,  # Short test
            reward_type="waiting_time"
        )
        
        print("   ✅ Environment creation successful")
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Traffic RL SUMO Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("SUMO Installation", test_sumo_installation),
        ("Python Packages", test_python_packages),
        ("Project Structure", test_project_structure),
        ("Network Generation", test_network_generation),
        ("Environment Creation", test_environment_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Installation is ready.")
        print("\n🚀 You can now run:")
        print("   python scripts/train.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
