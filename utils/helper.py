import os

ALLOWED_EXTENSIONS = set(['csv'])

def is_valid_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_file_exists(dirname, filename):
  return os.path.exists(os.path.join(dirname, filename))
