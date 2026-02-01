import sys
import os
import asyncio

# Setup paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config
config.setup_paths()

from managers.element_manager import element_manager
from database import Session, engine, Element

async def test_visualize():
    element_id = "55168b0dc7e34346ab3d965845dcb51e"
    print(f"Testing generation for element: {element_id}")
    
    try:
        # Check if element exists
        with Session(engine) as session:
            el = session.get(Element, element_id)
            if not el:
                print("Element not found!")
                return
            print(f"Element found: {el.name} (Project: {el.project_id})")

        # Run visualize
        path = await element_manager.generate_visual(element_id)
        if path:
            print(f"Success! Path: {path}")
        else:
            print("Failed (returned None).")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_visualize())
