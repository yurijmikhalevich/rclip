import sys


IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform == "win32" or sys.platform == "cygwin"

# these images are always processed
IMAGE_EXT = ["jpg", "jpeg", "png", "webp", "heic", "tiff", "tif", "bmp", "gif", "jp2", "pnm", "pbm", "pgm", "ppm"]
# RAW images are processed only if there is no processed image alongside it
IMAGE_RAW_EXT = [
  "3fr",
  "arw",
  "cr2",
  "cr3",
  "dng",
  "erf",
  "iiq",
  "mos",
  "mrw",
  "nef",
  "nrw",
  "orf",
  "pef",
  "raf",
  "rw2",
  "srw",
]
