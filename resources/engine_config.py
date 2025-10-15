import sys
MAC_DIR = '../raw_data/data/'
WINDOW_DIR = '/modeling_module/raw_data/'
if sys.platform == 'win32':
    DIR = WINDOW_DIR
else:
    DIR = MAC_DIR



IS_RUNNING = False