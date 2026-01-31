from database import init_db, Element, Project, get_session
from managers.element_manager import element_manager
import uuid

def test_phase1():
    print("Initializing DB...")
    init_db()
    
    print("Creating Project...")
    p_id = uuid.uuid4().hex
    
    print("Creating Element...")
    el = element_manager.create_element(
        project_id=p_id, 
        name="Test Hero", 
        type="character", 
        description="A cool hero", 
        trigger_word="@Hero"
    )
    assert el.id is not None
    print(f"Element created: {el.name} (ID: {el.id})")
    
    print("Testing Injection...")
    prompt = "The @Hero runs fast."
    new_prompt = element_manager.inject_elements_into_prompt(prompt, p_id)
    print(f"Original: {prompt}")
    print(f"Injected: {new_prompt}")
    
    assert "A cool hero" in new_prompt, "Injection failed!"
    
    print("Deleting Element...")
    success = element_manager.delete_element(el.id)
    assert success, "Delete failed!"
    
    # Verify deletion
    elements = element_manager.get_elements(p_id)
    assert len(elements) == 0, "Element not deleted!"
    
    print("SUCCESS: Phase 1 Verification Passed!")

if __name__ == "__main__":
    test_phase1()
