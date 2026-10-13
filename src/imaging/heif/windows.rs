//! HEIF decoding through the Windows Imaging Component (WIC).

use windows::Win32::Graphics::Imaging::{
  CLSID_WICImagingFactory, GUID_WICPixelFormat32bppRGBA, IWICImagingFactory, IWICPalette, WICBitmapDitherTypeNone,
  WICBitmapPaletteTypeCustom, WICDecodeMetadataCacheOnDemand,
};
use windows::Win32::System::Com::{
  CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};
use windows::core::HRESULT;

use crate::imaging::{ImageError, RgbImage, check_pixel_limit};

const RPC_E_CHANGED_MODE: HRESULT = HRESULT(0x80010106_u32 as i32);

pub fn decode(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  // SAFETY: COM is initialized for this thread and uninitialized only if this call initialized it.
  let initialized = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
  if initialized.is_err() && initialized != RPC_E_CHANGED_MODE {
    return Err(ImageError::decode(format!("Windows could not initialize COM ({initialized})")));
  }
  let result = decode_with_com(data, max_pixels);
  if initialized.is_ok() {
    // SAFETY: balances the successful CoInitializeEx above.
    unsafe { CoUninitialize() };
  }
  result
}

fn decode_with_com(data: &[u8], max_pixels: Option<u64>) -> Result<RgbImage, ImageError> {
  let wic = |action: &str| {
    let action = action.to_string();
    move |error: windows::core::Error| ImageError::decode(format!("Windows WIC could not {action} ({error})"))
  };
  // SAFETY: plain WIC calls on interfaces owned by this function.
  unsafe {
    let factory: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)
      .map_err(wic("create the imaging factory"))?;
    let stream = factory.CreateStream().map_err(wic("create a stream"))?;
    stream.InitializeFromMemory(data).map_err(wic("read the HEIF data"))?;
    let decoder = factory
      .CreateDecoderFromStream(&stream, std::ptr::null(), WICDecodeMetadataCacheOnDemand)
      .map_err(wic("find an installed HEIF decoder"))?;
    let frame = decoder.GetFrame(0).map_err(wic("read the first HEIF frame"))?;
    let converter = factory.CreateFormatConverter().map_err(wic("create an RGBA converter"))?;
    converter
      .Initialize(
        &frame,
        &GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        None::<&IWICPalette>,
        0.0,
        WICBitmapPaletteTypeCustom,
      )
      .map_err(wic("convert the HEIF image to RGBA"))?;
    let (mut width, mut height) = (0u32, 0u32);
    converter.GetSize(&mut width, &mut height).map_err(wic("read the HEIF dimensions"))?;
    if width == 0 || height == 0 {
      return Err(ImageError::decode("Windows WIC returned invalid HEIF image dimensions"));
    }
    check_pixel_limit(width, height, max_pixels)?;
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    converter.CopyPixels(std::ptr::null(), width * 4, &mut rgba).map_err(wic("decode the HEIF pixels"))?;
    Ok(RgbImage::from_rgba(width, height, &rgba))
  }
}
