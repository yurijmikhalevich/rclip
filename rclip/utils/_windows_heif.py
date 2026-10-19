"""HEIF decoding through Windows Imaging Component (WIC)."""

import ctypes
import math
import os
import uuid

from PIL import Image

from rclip.utils.native_heif import (
  NativeHeifCodecUnavailableError,
  NativeHeifDecodeError,
  NativeHeifError,
  NativeHeifTooLargeError,
)


class _GUID(ctypes.Structure):
  _fields_ = [
    ("data1", ctypes.c_uint32),
    ("data2", ctypes.c_uint16),
    ("data3", ctypes.c_uint16),
    ("data4", ctypes.c_ubyte * 8),
  ]

  @classmethod
  def parse(cls, value: str) -> "_GUID":
    parsed = uuid.UUID(value)
    fields = parsed.fields
    tail = bytes((fields[3], fields[4])) + fields[5].to_bytes(6, "big")
    return cls(fields[0], fields[1], fields[2], (ctypes.c_ubyte * 8).from_buffer_copy(tail))


_CLSID_WIC_IMAGING_FACTORY = _GUID.parse("cacaf262-9370-4615-a13b-9f5539da4c0a")
_IID_WIC_IMAGING_FACTORY = _GUID.parse("ec5ec8a9-c395-4314-9c77-54d7a935ff70")
_GUID_WIC_PIXEL_FORMAT_32BPP_RGBA = _GUID.parse("f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9")

_CLSCTX_INPROC_SERVER = 1
_COINIT_MULTITHREADED = 0
_RPC_E_CHANGED_MODE = ctypes.c_int32(0x80010106).value
_GENERIC_READ = 0x80000000
_WIC_DECODE_METADATA_CACHE_ON_DEMAND = 0
_WIC_BITMAP_DITHER_TYPE_NONE = 0
_WIC_BITMAP_PALETTE_TYPE_CUSTOM = 0
_WIC_BITMAP_INTERPOLATION_MODE_FANT = 3

_FACILITY_WIN32 = 7

_HRESULT = ctypes.c_int32
_WINFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


def _failed(result: int) -> bool:
  return result < 0


def _is_file_error(result: int) -> bool:
  """Whether a failed HRESULT wraps a Win32 error, such as a missing, locked, or unreadable file."""
  return _failed(result) and (result >> 16) & 0x1FFF == _FACILITY_WIN32


def _check(result: int, action: str, error: type[NativeHeifError] = NativeHeifDecodeError) -> None:
  if _failed(result):
    raise error(f"Windows WIC could not {action} (HRESULT 0x{result & 0xFFFFFFFF:08X})")


def _method(interface: ctypes.c_void_p, index: int, result_type, *argument_types):
  vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
  address = vtable[index]
  return _WINFUNCTYPE(result_type, ctypes.c_void_p, *argument_types)(address)


def _release(interface: ctypes.c_void_p | None) -> None:
  if interface and interface.value:
    _method(interface, 2, ctypes.c_ulong)(interface)


def _scaled_size(width: int, height: int, min_side: int) -> tuple[int, int]:
  """The smallest size whose shortest side still covers ``min_side``, never upscaling."""
  scale = min(1.0, min_side / min(width, height))
  return math.ceil(width * scale), math.ceil(height * scale)


def decode_path(path: str | os.PathLike[str], max_pixels: int | None, min_side: int | None = None) -> Image.Image:
  ole32 = getattr(ctypes, "OleDLL")("ole32")
  ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
  ole32.CoInitializeEx.restype = _HRESULT
  ole32.CoCreateInstance.argtypes = [
    ctypes.POINTER(_GUID),
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(_GUID),
    ctypes.POINTER(ctypes.c_void_p),
  ]
  ole32.CoCreateInstance.restype = _HRESULT
  ole32.CoUninitialize.argtypes = []

  initialize_result = int(ole32.CoInitializeEx(None, _COINIT_MULTITHREADED))
  if _failed(initialize_result) and initialize_result != _RPC_E_CHANGED_MODE:
    raise NativeHeifDecodeError(f"Windows could not initialize COM (HRESULT 0x{initialize_result & 0xFFFFFFFF:08X})")
  should_uninitialize = not _failed(initialize_result)

  factory = ctypes.c_void_p()
  decoder = ctypes.c_void_p()
  frame = ctypes.c_void_p()
  scaler = ctypes.c_void_p()
  converter = ctypes.c_void_p()
  try:
    _check(
      int(
        ole32.CoCreateInstance(
          ctypes.byref(_CLSID_WIC_IMAGING_FACTORY),
          None,
          _CLSCTX_INPROC_SERVER,
          ctypes.byref(_IID_WIC_IMAGING_FACTORY),
          ctypes.byref(factory),
        )
      ),
      "create the imaging factory",
      NativeHeifCodecUnavailableError,
    )

    create_decoder_from_filename = _method(
      factory,
      3,
      _HRESULT,
      ctypes.c_wchar_p,
      ctypes.POINTER(_GUID),
      ctypes.c_uint32,
      ctypes.c_int,
      ctypes.POINTER(ctypes.c_void_p),
    )
    create_decoder_result = int(
      create_decoder_from_filename(
        factory,
        os.fspath(path),
        None,
        _GENERIC_READ,
        _WIC_DECODE_METADATA_CACHE_ON_DEMAND,
        ctypes.byref(decoder),
      )
    )
    # WIC opens the file before it looks for a decoder, so a Win32 error means the file couldn't be read
    if _is_file_error(create_decoder_result):
      _check(create_decoder_result, "open the HEIF file")
    _check(create_decoder_result, "find an installed HEIF decoder", NativeHeifCodecUnavailableError)

    get_frame = _method(decoder, 13, _HRESULT, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
    _check(int(get_frame(decoder, 0, ctypes.byref(frame))), "read the first HEIF frame")

    # the pixel limit applies to the source image, which is what the scaler below reads
    get_frame_size = _method(frame, 3, _HRESULT, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
    frame_width = ctypes.c_uint32()
    frame_height = ctypes.c_uint32()
    _check(
      int(get_frame_size(frame, ctypes.byref(frame_width), ctypes.byref(frame_height))),
      "read the HEIF dimensions",
    )
    if not frame_width.value or not frame_height.value:
      raise NativeHeifDecodeError("Windows WIC returned invalid HEIF image dimensions")
    pixels = int(frame_width.value) * int(frame_height.value)
    if max_pixels is not None and pixels > max_pixels:
      raise NativeHeifTooLargeError(pixels, max_pixels)

    # Decoding straight to the requested size keeps the full-resolution RGBA buffer out of memory.
    # WIC hands the scaling to the HEIF codec's own downscaler when the codec has one, so the
    # interpolation mode below only applies to WIC's software resampling.
    source = frame
    target_width, target_height = frame_width.value, frame_height.value
    if min_side is not None:
      target_width, target_height = _scaled_size(frame_width.value, frame_height.value, min_side)
    if (target_width, target_height) != (frame_width.value, frame_height.value):
      create_bitmap_scaler = _method(factory, 11, _HRESULT, ctypes.POINTER(ctypes.c_void_p))
      _check(int(create_bitmap_scaler(factory, ctypes.byref(scaler))), "create a bitmap scaler")
      initialize_scaler = _method(scaler, 8, _HRESULT, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int)
      _check(
        int(initialize_scaler(scaler, frame, target_width, target_height, _WIC_BITMAP_INTERPOLATION_MODE_FANT)),
        "scale the HEIF image",
      )
      source = scaler

    create_format_converter = _method(factory, 10, _HRESULT, ctypes.POINTER(ctypes.c_void_p))
    _check(int(create_format_converter(factory, ctypes.byref(converter))), "create an RGBA converter")
    initialize_converter = _method(
      converter,
      8,
      _HRESULT,
      ctypes.c_void_p,
      ctypes.POINTER(_GUID),
      ctypes.c_int,
      ctypes.c_void_p,
      ctypes.c_double,
      ctypes.c_int,
    )
    _check(
      int(
        initialize_converter(
          converter,
          source,
          ctypes.byref(_GUID_WIC_PIXEL_FORMAT_32BPP_RGBA),
          _WIC_BITMAP_DITHER_TYPE_NONE,
          None,
          0.0,
          _WIC_BITMAP_PALETTE_TYPE_CUSTOM,
        )
      ),
      "convert the HEIF image to RGBA",
    )

    width = ctypes.c_uint32()
    height = ctypes.c_uint32()
    get_size = _method(converter, 3, _HRESULT, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
    _check(int(get_size(converter, ctypes.byref(width), ctypes.byref(height))), "read the decoded dimensions")
    if not width.value or not height.value:
      raise NativeHeifDecodeError("Windows WIC returned invalid HEIF image dimensions")

    stride = int(width.value) * 4
    buffer_size = stride * int(height.value)
    pixel_buffer = (ctypes.c_ubyte * buffer_size)()
    copy_pixels = _method(
      converter,
      7,
      _HRESULT,
      ctypes.c_void_p,
      ctypes.c_uint32,
      ctypes.c_uint32,
      ctypes.c_void_p,
    )
    _check(int(copy_pixels(converter, None, stride, buffer_size, pixel_buffer)), "decode the HEIF pixels")
    return Image.frombytes("RGBA", (width.value, height.value), bytes(pixel_buffer))
  finally:
    _release(converter)
    _release(scaler)
    _release(frame)
    _release(decoder)
    _release(factory)
    if should_uninitialize:
      ole32.CoUninitialize()
