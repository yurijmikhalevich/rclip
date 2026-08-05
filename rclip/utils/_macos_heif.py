"""HEIF decoding through macOS Image I/O and Core Graphics."""

import ctypes
import io
import os
from pathlib import Path

from PIL import Image

from rclip.utils.native_heif import (
  NativeHeifCodecUnavailableError,
  NativeHeifDecodeError,
  NativeHeifTooLargeError,
)


_CORE_FOUNDATION = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_CORE_GRAPHICS = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
_IMAGE_IO = ctypes.CDLL("/System/Library/Frameworks/ImageIO.framework/ImageIO")

_CORE_FOUNDATION.CFURLCreateFromFileSystemRepresentation.argtypes = [
  ctypes.c_void_p,
  ctypes.POINTER(ctypes.c_ubyte),
  ctypes.c_long,
  ctypes.c_bool,
]
_CORE_FOUNDATION.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFRelease.argtypes = [ctypes.c_void_p]
_CORE_FOUNDATION.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_CORE_FOUNDATION.CFDictionaryGetValue.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_CORE_FOUNDATION.CFNumberGetValue.restype = ctypes.c_bool
_CORE_FOUNDATION.CFDataCreateMutable.argtypes = [ctypes.c_void_p, ctypes.c_long]
_CORE_FOUNDATION.CFDataCreateMutable.restype = ctypes.c_void_p
_CORE_FOUNDATION.CFDataGetLength.argtypes = [ctypes.c_void_p]
_CORE_FOUNDATION.CFDataGetLength.restype = ctypes.c_long
_CORE_FOUNDATION.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
_CORE_FOUNDATION.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
_CORE_FOUNDATION.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
_CORE_FOUNDATION.CFStringCreateWithCString.restype = ctypes.c_void_p
_IMAGE_IO.CGImageSourceCreateWithURL.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_IMAGE_IO.CGImageSourceCreateWithURL.restype = ctypes.c_void_p
_IMAGE_IO.CGImageSourceCreateImageAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_IMAGE_IO.CGImageSourceCreateImageAtIndex.restype = ctypes.c_void_p
_IMAGE_IO.CGImageSourceCopyPropertiesAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_IMAGE_IO.CGImageSourceCopyPropertiesAtIndex.restype = ctypes.c_void_p
_IMAGE_IO.CGImageDestinationCreateWithData.argtypes = [
  ctypes.c_void_p,
  ctypes.c_void_p,
  ctypes.c_size_t,
  ctypes.c_void_p,
]
_IMAGE_IO.CGImageDestinationCreateWithData.restype = ctypes.c_void_p
_IMAGE_IO.CGImageDestinationAddImage.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_IMAGE_IO.CGImageDestinationFinalize.argtypes = [ctypes.c_void_p]
_IMAGE_IO.CGImageDestinationFinalize.restype = ctypes.c_bool
_CORE_GRAPHICS.CGImageGetWidth.argtypes = [ctypes.c_void_p]
_CORE_GRAPHICS.CGImageGetWidth.restype = ctypes.c_size_t
_CORE_GRAPHICS.CGImageGetHeight.argtypes = [ctypes.c_void_p]
_CORE_GRAPHICS.CGImageGetHeight.restype = ctypes.c_size_t

_K_CF_NUMBER_INT_TYPE = 9
_K_CF_STRING_ENCODING_UTF8 = 0x08000100


def _copy_orientation(image_source: int) -> int:
  properties = _IMAGE_IO.CGImageSourceCopyPropertiesAtIndex(image_source, 0, None)
  if not properties:
    return 1
  try:
    orientation_key = ctypes.c_void_p.in_dll(_IMAGE_IO, "kCGImagePropertyOrientation").value
    orientation_value = _CORE_FOUNDATION.CFDictionaryGetValue(properties, orientation_key)
    orientation = ctypes.c_int(1)
    if orientation_value:
      _CORE_FOUNDATION.CFNumberGetValue(orientation_value, _K_CF_NUMBER_INT_TYPE, ctypes.byref(orientation))
    return orientation.value
  finally:
    _CORE_FOUNDATION.CFRelease(properties)


def _apply_orientation(image: Image.Image, orientation: int) -> Image.Image:
  transforms = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
  }
  transform = transforms.get(orientation)
  return image.transpose(transform) if transform is not None else image


def _copy_as_pillow_image(cg_image: int) -> Image.Image:
  output_data = _CORE_FOUNDATION.CFDataCreateMutable(None, 0)
  png_type = _CORE_FOUNDATION.CFStringCreateWithCString(None, b"public.png", _K_CF_STRING_ENCODING_UTF8)
  destination = None
  try:
    if not output_data or not png_type:
      raise NativeHeifDecodeError("macOS could not allocate an HEIF conversion buffer")
    destination = _IMAGE_IO.CGImageDestinationCreateWithData(output_data, png_type, 1, None)
    if not destination:
      raise NativeHeifDecodeError("macOS could not create an HEIF image destination")
    _IMAGE_IO.CGImageDestinationAddImage(destination, cg_image, None)
    if not _IMAGE_IO.CGImageDestinationFinalize(destination):
      raise NativeHeifDecodeError("macOS could not render the HEIF image")
    data_length = int(_CORE_FOUNDATION.CFDataGetLength(output_data))
    encoded_image = ctypes.string_at(_CORE_FOUNDATION.CFDataGetBytePtr(output_data), data_length)
    with Image.open(io.BytesIO(encoded_image)) as image:
      image.load()
      return image.convert("RGB")
  finally:
    for value in (destination, png_type, output_data):
      if value:
        _CORE_FOUNDATION.CFRelease(value)


def decode_path(path: str | os.PathLike[str], max_pixels: int | None) -> Image.Image:
  encoded_path = os.fsencode(Path(path))
  path_buffer = (ctypes.c_ubyte * len(encoded_path)).from_buffer_copy(encoded_path)
  file_url = _CORE_FOUNDATION.CFURLCreateFromFileSystemRepresentation(None, path_buffer, len(encoded_path), False)
  if not file_url:
    raise NativeHeifDecodeError(f'Image I/O could not open "{path}"')
  image_source = None
  cg_image = None
  try:
    image_source = _IMAGE_IO.CGImageSourceCreateWithURL(file_url, None)
    if not image_source:
      raise NativeHeifCodecUnavailableError("macOS Image I/O does not recognize this HEIF image")
    cg_image = _IMAGE_IO.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if not cg_image:
      raise NativeHeifCodecUnavailableError("macOS has no codec capable of decoding this HEIF image")
    width = int(_CORE_GRAPHICS.CGImageGetWidth(cg_image))
    height = int(_CORE_GRAPHICS.CGImageGetHeight(cg_image))
    pixels = width * height
    if width <= 0 or height <= 0:
      raise NativeHeifDecodeError("macOS returned invalid HEIF image dimensions")
    if max_pixels is not None and pixels > max_pixels:
      raise NativeHeifTooLargeError(pixels, max_pixels)
    return _apply_orientation(_copy_as_pillow_image(cg_image), _copy_orientation(image_source))
  finally:
    for value in (cg_image, image_source, file_url):
      if value:
        _CORE_FOUNDATION.CFRelease(value)
