import sys


IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')
IS_WINDOWS = sys.platform == 'win32' or sys.platform == 'cygwin'
