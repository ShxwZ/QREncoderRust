use image::GrayImage;
use qrcode::QrCode;
use image::Luma;
use std::error::Error;

pub fn generate_qrg_image_gray(
  content: &[u8]
) -> Result<GrayImage, Box<dyn Error>> {
  // Crear el código QR utilizando una versión fija (aquí versión 40, que es la máxima) y nivel de corrección bajo
  let code = QrCode::with_version(content, qrcode::Version::Normal(40), qrcode::EcLevel::L)?;
  
  // Renderizar el código QR en una imagen en escala de grises
  let image: image::ImageBuffer<Luma<u8>, Vec<u8>>  = code.render::<Luma<u8>>().module_dimensions(8, 8).build();
  
  // Devolver la ruta del archivo generado
  Ok(image)
}

pub fn prepare_qr_data(qrs: &[GrayImage]) -> Vec<u32> {
  qrs.iter().flat_map(|qr| qr.as_raw().iter().map(|&pixel| (255 << 24) | (pixel as u32) << 16 | (pixel as u32) << 8 | pixel as u32)).collect()
}

pub fn get_frame_dimensions(frames_qrs: &[Vec<GrayImage>]) -> (u32, u32) {
  let qr_width = frames_qrs[0][0].width();
  let qr_height = frames_qrs[0][0].height();
  (qr_width * 3, qr_height * 2)
}
