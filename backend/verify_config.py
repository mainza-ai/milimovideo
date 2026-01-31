import os
import sys

def test_config():
    print("Testing Config Import...")
    try:
        import config
        print("Config imported successfully.")
    except ImportError as e:
        print(f"FAILED to import config: {e}")
        return

    print(f"PROJECTS_DIR: {config.PROJECTS_DIR}")
    print(f"LTX_CORE_DIR: {config.LTX_CORE_DIR}")
    
    assert os.path.exists(config.PROJECTS_DIR), "Projects dir not created"
    
    # Test Path Setup
    config.setup_paths()
    assert config.LTX_CORE_DIR in sys.path, "LTX Core not in sys.path"
    
    print("SUCCESS: Config verified.")

if __name__ == "__main__":
    test_config()
