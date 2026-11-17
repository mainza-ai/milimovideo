from sqlalchemy import inspect, text

from database import engine


def test_alembic_baseline_is_stamped(app_client):
    inspector = inspect(engine)
    assert "alembic_version" in inspector.get_table_names()
    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == "0004_asset_thumbnail"


def test_current_schema_columns_are_present(app_client):
    inspector = inspect(engine)
    shot_columns = {column["name"] for column in inspector.get_columns("shot")}
    assert {
        "sequence_index", "selected_take_id", "favorite_take_id", "keyframes", "blocking",
        "speed", "audio_gain", "audio_muted", "audio_fade_in", "audio_fade_out",
        "seed_locked",
    } <= shot_columns

    asset_columns = {column["name"] for column in inspector.get_columns("asset")}
    assert "thumbnail_url" in asset_columns

    project_columns = {column["name"] for column in inspector.get_columns("project")}
    assert {
        "revision", "board_layout", "board_revision", "view_state", "captions",
        "captions_burn", "render_preset", "render_crf", "render_speed",
        "render_audio_bitrate", "render_audio_normalize", "script_content",
        "storyboard_archive",
    } <= project_columns
