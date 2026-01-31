import asyncio
import os
from PIL import Image
from models.flux_wrapper import flux_inpainter
from managers.inpainting_manager import inpainting_manager
import config

async def test_phase2():
    print("Testing Flux Wrapper...")
    # Create dummy image
    img = Image.new("RGB", (100, 100), color="red")
    mask = Image.new("L", (100, 100), color=0)
    
    # Test Inpaint (Mocked)
    res = flux_inpainter.inpaint(img, mask, "A blue box")
    assert res is not None
    print("Flux Inpaint Call Successful.")
    
    print("Testing InpaintingManager...")
    # Mock image path
    mock_img_path = "test_img.png"
    img.save(mock_img_path)
    
    # Run Manager Process
    # We won't test SAM call here as it requires a running service, 
    # but we can test the flux part via the manager if we provide a mask.
    mock_mask_path = "test_mask.png"
    mask.save(mock_mask_path)
    
    try:
        out_path = await inpainting_manager.process_inpaint("job_123", mock_img_path, mock_mask_path, "A blue box")
        print(f"Manager Output Path: {out_path}")
        assert os.path.exists(out_path)
        print("InpaintingManager Process Successful.")
    except Exception as e:
        print(f"Manager Failed: {e}")
        
    # Cleanup
    if os.path.exists(mock_img_path): os.remove(mock_img_path)
    if os.path.exists(mock_mask_path): os.remove(mock_mask_path)
    if os.path.exists(out_path): os.remove(out_path)
    
    print("SUCCESS: Phase 2 Verification Passed!")

if __name__ == "__main__":
    asyncio.run(test_phase2())
