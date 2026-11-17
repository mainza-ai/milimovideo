def test_element_crud_round_trip(app_client, project):
    pid = project["id"]

    created = app_client.post(f"/projects/{pid}/elements", json={
        "name": "Mara",
        "type": "character",
        "description": "Lead pilot",
        "trigger_word": "mara",
    })
    assert created.status_code == 200, created.text
    element = created.json()
    assert element["name"] == "Mara"
    assert element["trigger_word"] == "@mara"

    listed = app_client.get(f"/projects/{pid}/elements")
    assert listed.status_code == 200
    assert any(item["id"] == element["id"] for item in listed.json())

    updated = app_client.put(f"/elements/{element['id']}", json={
        "description": "Lead pilot, red jacket",
        "trigger_word": "@mara_v2",
    })
    assert updated.status_code == 200, updated.text
    assert updated.json()["description"] == "Lead pilot, red jacket"
    assert updated.json()["trigger_word"] == "@mara_v2"

    deleted = app_client.delete(f"/elements/{element['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True

    remaining = app_client.get(f"/projects/{pid}/elements").json()
    assert all(item["id"] != element["id"] for item in remaining)


def test_element_update_rejects_empty_payload(app_client, project):
    pid = project["id"]
    created = app_client.post(f"/projects/{pid}/elements", json={
        "name": "Rig",
        "type": "object",
        "description": "Camera rig",
        "trigger_word": "rig",
    }).json()

    response = app_client.put(f"/elements/{created['id']}", json={})
    assert response.status_code == 400


def test_element_trigger_defaults_from_name(app_client, project):
    pid = project["id"]
    created = app_client.post(f"/projects/{pid}/elements", json={
        "name": "Red Harbor",
        "type": "location",
        "description": "Docks at dusk",
    })
    assert created.status_code == 200, created.text
    assert created.json()["trigger_word"] == "@RedHarbor"
