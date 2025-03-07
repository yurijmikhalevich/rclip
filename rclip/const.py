import sys


IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform == "win32" or sys.platform == "cygwin"

# these images are always processed
IMAGE_EXT = ["jpg", "jpeg", "png", "webp"]
# RAW images are processed only if there is no processed image alongside it
IMAGE_RAW_EXT = ["arw", "cr2"]
