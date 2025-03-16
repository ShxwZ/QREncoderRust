use std::path::Path;
use std::time::Duration;
use image::{GrayImage, ImageEncoder};

pub fn encode_image(frame: &GrayImage) ->( Vec<u8>, Duration) {
  let capacity = frame.width() as usize * frame.height() as usize * 2; // Estimación conservadora
  let mut buffer = Vec::with_capacity(capacity);
  let start_time = std::time::Instant::now();
  // Usar configuraciones de máximo rendimiento para PNG
  image::codecs::png::PngEncoder::new(
      &mut buffer,
  )
  .write_image(
      frame.as_raw(),
      frame.width(),
      frame.height(),
      image::ExtendedColorType::L8
  )
  .expect("Failed to encode image");
  
  (buffer, start_time.elapsed())
}


pub fn write_to_disk(output_dir: &Path, index: usize, buffer: &[u8]) -> Duration {
  use std::io::{BufWriter, Write};
  use std::fs::File;
  let start_time = std::time::Instant::now();
  let output_path = output_dir.join(format!("frame_{}.png", index));
  
  // Usar un BufWriter con el tamaño del buffer exactamente igual
  // al del archivo para hacer una sola operación de escritura
  let file = File::create(&output_path).expect("Failed to create file");
  let mut writer = BufWriter::with_capacity(buffer.len(), file);
  
  // Una sola operación de escritura
  writer.write_all(buffer).expect("Failed to write image to disk");
  
  // Flush explícito pero sin syscall adicional (más eficiente)
  writer.flush().expect("Failed to flush buffer");

  return start_time.elapsed();
}

pub fn create_output_directory(output_dir: &Path) {
  if !output_dir.exists() {
      std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
  }
}
