import os, os.path

SUPPORTED_FILE_EXTS = ['.jpg', '.png', '.jpeg']

def get_image_files(dir):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            # Only include files whose extension is a known image type
            if is_image_file(file):
                full_path = os.path.join(root, file)
                file_list.extend([full_path])
    return file_list

def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    if ext.lower() in SUPPORTED_FILE_EXTS:
        return True
    return False
