use std::path::PathBuf;
use image::GrayImage;
use image::Luma;
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;
use crate::image_utils::{encode_image, write_to_disk};

pub fn spawn_save_thread(
  rx: mpsc::Receiver<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>,
  output_dir: PathBuf,
) -> std::thread::JoinHandle<()> {
  // Usar más hilos para I/O paralelo
  let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(16)  // Más hilos para mejor paralelismo
      .build()
      .unwrap();
  
  thread::spawn(move || {
      let mut pending_frames = Vec::with_capacity(16);
      
      while let Ok((frame, index, compute_time, transfer_time, frame_start_time)) = rx.recv() {
          // Acumular frames hasta un umbral
          pending_frames.push((frame, index, compute_time, transfer_time, frame_start_time));
          
          // Procesar en batch cuando tengamos suficientes o se acabe el stream
          if pending_frames.len() >= 6 {
              // Tomar propiedad de los frames pendientes y reemplazar con vector vacío
              let frames_to_process = std::mem::replace(&mut pending_frames, Vec::with_capacity(16));
              
              // Usar el método común para procesar frames
              process_pending_frames(frames_to_process, &output_dir, &pool);
          }
      }
      
      // Procesar los frames restantes
      if !pending_frames.is_empty() {
          let frames_to_process = std::mem::replace(&mut pending_frames, Vec::with_capacity(16));
          process_pending_frames(frames_to_process, &output_dir, &pool);
      }
  })
}

pub fn process_pending_frames(
  frames: Vec<(GrayImage, usize, std::time::Duration, std::time::Duration, std::time::Instant)>,
  output_dir: &PathBuf,
  pool: &rayon::ThreadPool
) {
  pool.install(|| {
      frames.into_par_iter().for_each(|(frame, idx, compute_time, transfer_time, frame_start_time)| {
          // Usar los métodos existentes para codificación y escritura
          let (buffer, encode_time) = encode_image(&frame);
          
          // Escribir a disco usando el método existente
          let write_time = write_to_disk(&output_dir, idx, &buffer);
          
          // Reportar tiempos
          let real_total = frame_start_time.elapsed();
          let actual_io_time = encode_time + write_time;
          let overhead_time = real_total - compute_time - transfer_time - actual_io_time;
          
          println!("Frame {} saved: Total={:.2?}, GPU={:.2?} (Compute={:.2?}, Transfer={:.2?}), I/O={:.2?} (Encode={:.2?}, Write={:.2?}), Queue/Overhead={:.2?}", 
              idx, real_total, 
              compute_time.saturating_add(transfer_time),
              compute_time, transfer_time, 
              actual_io_time, encode_time, write_time,
              overhead_time);
      });
  });
}

pub async fn read_frame_data(
  device: &wgpu::Device,
  result_staging_buffer: wgpu::Buffer,
  frame_width: u32,
  frame_height: u32,
) -> GrayImage {
  // Mapear buffer y esperar resultados
  let buffer_slice = result_staging_buffer.slice(..);
  let scale_factor = 2;  // Debe coincidir con el valor en create_buffers y run_compute_shader
  let scaled_width = frame_width / scale_factor;
  let scaled_height = frame_height / scale_factor;
  let (sender, receiver) = futures::channel::oneshot::channel();
  
  buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
      sender.send(result).unwrap();
  });
  
  // Esto es un punto de sincronización crítico - espera a que la GPU termine
  device.poll(wgpu::Maintain::Wait);
  let _ = receiver.await.ok().expect("Failed to map buffer");

  let data = buffer_slice.get_mapped_range();
  let frame_data_u32: &[u32] = bytemuck::cast_slice(&*data);
  let frame_data_u8: Vec<u8> = frame_data_u32.iter()
      .map(|&rgba| ((rgba >> 16) & 0xFF) as u8)
      .collect();
  
  // Crear imagen
  return GrayImage::from_raw(scaled_width, scaled_height, frame_data_u8)
   .expect("No se pudo crear la imagen desde los datos de la GPU");


}

