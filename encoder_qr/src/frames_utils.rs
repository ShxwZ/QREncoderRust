use std::path::PathBuf;
use image::GrayImage;
use image::Luma;
use rayon::prelude::*;
use tokio::sync::mpsc;
use tokio::task;
use crate::image_utils::{encode_image, write_to_disk};
use std::sync::Arc;

pub fn spawn_save_thread(
  mut rx: mpsc::Receiver<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>,
  output_dir: PathBuf,
) -> task::JoinHandle<()> {
  // Aumentar el número de hilos para el procesamiento de frames
  let pool = Arc::new(rayon::ThreadPoolBuilder::new()
      .num_threads(128)
      .build()
      .unwrap());
  
  // Crear múltiples hilos de procesamiento
  const NUM_WORKER_THREADS: usize = 3;
  
  // En lugar de un Mutex, usamos canales de un solo productor, múltiples consumidores
  let mut receivers = Vec::with_capacity(NUM_WORKER_THREADS);
  let mut senders = Vec::with_capacity(NUM_WORKER_THREADS);
  
  for _ in 0..NUM_WORKER_THREADS {
      let (tx, rx) = tokio::sync::mpsc::channel(10);
      senders.push(tx);
      receivers.push(rx);
  }
  
  // Crear hilos de procesamiento
  let mut worker_handles = Vec::with_capacity(NUM_WORKER_THREADS);
  for (thread_id, mut receiver) in receivers.into_iter().enumerate() {
      let output_dir_clone = output_dir.clone();
      let pool_clone = Arc::clone(&pool);
      
      let handle = std::thread::spawn(move || {
          let mut local_frames = Vec::with_capacity(16);
          let runtime = tokio::runtime::Builder::new_current_thread()
              .enable_all()
              .build()
              .unwrap();
          
          runtime.block_on(async {
              while let Some(frame) = receiver.recv().await {
                  local_frames.push(frame);
                  
                  // Procesar inmediatamente si hay suficientes frames acumulados
                  if local_frames.len() >= 2 {
                      let frames_to_process = std::mem::replace(&mut local_frames, Vec::with_capacity(16));
                      process_pending_frames(frames_to_process, &output_dir_clone, &pool_clone);
                  }
              }
              
              // Procesar frames finales
              if !local_frames.is_empty() {
                  let frames_to_process = std::mem::replace(&mut local_frames, Vec::with_capacity(0));
                  process_pending_frames(frames_to_process, &output_dir_clone, &pool_clone);
              }
          });
          
          println!("[WORKER {}] Finalizando", thread_id);
      });
      
      worker_handles.push(handle);
  }

  // Hilo principal que distribuye frames de manera equitativa
  task::spawn(async move {
      let mut frame_counter = 0;
      let num_workers = senders.len();
      
      // Recibir frames del canal principal y distribuirlos a los trabajadores
      while let Some(frame) = rx.recv().await {
          // Distribuir de forma circular usando contador
          let worker_index = frame_counter % num_workers;
          frame_counter += 1;
          
          // Enviar directamente al trabajador correspondiente
          if senders[worker_index].send(frame).await.is_err() {
              println!("[ERROR] Trabajador {} desconectado", worker_index);
          }
      }
      
      // Cerrar todos los canales de trabajo
      senders.clear();
      
      // Esperar a que todos los trabajadores terminen
      for (i, handle) in worker_handles.into_iter().enumerate() {
          if let Err(e) = handle.join() {
              println!("[ERROR] El hilo trabajador {} falló: {:?}", i, e);
          }
      }
      
      println!("[SAVE THREAD] Todos los hilos de trabajadores han finalizado");
  })
}

pub fn process_pending_frames(
  frames: Vec<(GrayImage, usize, std::time::Duration, std::time::Duration, std::time::Instant)>,
  output_dir: &PathBuf,
  pool: &rayon::ThreadPool
) {
  pool.install(|| {
      let start_process_time = std::time::Instant::now();
      let total_frames = frames.len();
      frames.into_par_iter().for_each(|(frame, idx, compute_time, transfer_time, frame_start_time)| {
          // Capturar el momento exacto cuando este frame comienza a procesarse en este hilo
          let thread_start_time = std::time::Instant::now();
          
          // Usar los métodos existentes para codificación y escritura
          let (buffer, encode_time) = encode_image(&frame);
          let write_time = write_to_disk(&output_dir, idx, &buffer);
          
          // Tiempo total desde que se generó el frame hasta que se completó la escritura
          let real_total = frame_start_time.elapsed();
          
          // Tiempo de procesamiento en este hilo específicamente
          let thread_time = thread_start_time.elapsed();
          
          // Tiempo que pasó en colas antes de comenzar a procesarse en este hilo
          let queue_time = real_total - compute_time - transfer_time - thread_time;
          
          // Overhead dentro del procesamiento en este hilo (si hay alguno)
          let actual_io_time = encode_time + write_time;
          let thread_overhead = thread_time - actual_io_time;
          
          println!("Frame {} saved: Total={:.2?}, En cola={:.2?}, En hilo={:.2?} (I/O={:.2?}, Overhead={:.2?})",
                  idx, real_total, queue_time, thread_time, actual_io_time, thread_overhead);
          
          // Versión detallada para depuración
          println!("  Desglose: GPU={:.2?} (Compute={:.2?}, Transfer={:.2?}), I/O={:.2?} (Encode={:.2?}, Write={:.2?})", 
                  compute_time + transfer_time, compute_time, transfer_time, 
                  actual_io_time, encode_time, write_time);
      });
      
      println!("Batch procesado en {:.2?} ({} frames)", 
              start_process_time.elapsed(), total_frames);
  });
}

pub async fn read_frame_data(
  device: &wgpu::Device,
  result_staging_buffer: wgpu::Buffer,
  frame_width: u32,
  frame_height: u32,
) -> GrayImage {
  // Reducir la contención de la GPU iniciando la operación de mapeo lo antes posible
  let buffer_slice = result_staging_buffer.slice(..);
  let scale_factor = 2;
  let scaled_width = frame_width / scale_factor;
  let scaled_height = frame_height / scale_factor;
  
  // Iniciar el mapeo inmediatamente
  let (sender, receiver) = futures::channel::oneshot::channel();
  buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
      // Manejar posibles errores al enviar el resultado
      if let Err(_) = sender.send(result) {
          println!("Error: El receptor fue descartado antes de recibir el resultado");
      }
  });
  
  // Método más simple y confiable: esperar de forma asíncrona a que la GPU termine
  device.poll(wgpu::Maintain::Wait);
  
  // Esperar el resultado de forma asíncrona
  _ = receiver.await
      .expect("El canal fue cerrado prematuramente")
      .expect("Error al mapear el buffer");
      
  // El buffer ahora está garantizado que está mapeado
  let data = buffer_slice.get_mapped_range();
  let frame_data_u32: &[u32] = bytemuck::cast_slice(&*data);
  let frame_data_u8: Vec<u8> = frame_data_u32.iter()
      .map(|&rgba| ((rgba >> 16) & 0xFF) as u8)
      .collect();
  
  // Crear la imagen a partir de los datos extraídos
  GrayImage::from_raw(scaled_width, scaled_height, frame_data_u8)
    .expect("No se pudo crear la imagen desde los datos de la GPU")
}