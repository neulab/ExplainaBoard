def load_file_as_str(path) -> str:
    content = ""
    with open(path, 'r') as f:
        content = f.read()
    return content
