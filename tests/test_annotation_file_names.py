from src.preprocess.annotations import resolve_image_file_names


def test_resolve_image_file_names_uses_existing_extension(tmp_path):
    (tmp_path / "door_top_native.tiff").write_bytes(b"x")
    (tmp_path / "ok.png").write_bytes(b"x")
    coco = {"images": [
        {"file_name": "door_top_native.jpg"},
        {"file_name": "ok.png"},
        {"file_name": "missing.jpg"},
    ]}

    assert resolve_image_file_names(coco, tmp_path) == 1
    assert [image["file_name"] for image in coco["images"]] == ["door_top_native.tiff", "ok.png", "missing.jpg"]
