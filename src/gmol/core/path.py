def safe_filename(filename: str) -> str:
    return "".join(
        [c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in filename]
    )
