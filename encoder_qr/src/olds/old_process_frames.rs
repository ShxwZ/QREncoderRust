async fn process_frames_batch(
  device: &wgpu::Device,
  queue: &wgpu::Queue,
  shader_source: &str,
  frames_qrs: &[Vec<GrayImage>],
  output_dir: &Path,
  start_index: usize,
) {
  if !output_dir.exists() {
      std::fs::create_dir_all(output_dir).expect("No se pudo crear el directorio de salida");
  }
  
  // Crear un canal para enviar imágenes a un hilo de guardado dedicado
  let (tx, rx) = mpsc::channel::<(
      image::ImageBuffer<image::Luma<u8>, Vec<u8>>, // GrayImage
      usize,                                        // índice del frame
      std::time::Duration,                          // tiempo de preparación
      std::time::Duration,                          // tiempo de cómputo
      std::time::Duration,                          // tiempo de transferencia
      std::time::Instant                            // tiempo de inicio del frame
  )>();

  
  // Iniciar un hilo dedicado para guardar imágenes
  let output_dir_clone = output_dir.to_path_buf();
  let save_thread = thread::spawn(move || {
      let mut frames_saved = 0;
      let mut buffer = Vec::with_capacity(1024 * 1024); // Preasignar 1MB
      
      while let Ok((frame, index, prepare_time, compute_time, transfer_time, frame_start_time)) = rx.recv() {
          buffer.clear();
          
          
          // FASE 1: CODIFICACIÓN
          let start_encode_time = std::time::Instant::now();
          
          // Probar con JPEG en lugar de PNG (mucho más rápido)
          let encoder = image::codecs::png::PngEncoder::new_with_quality(
              &mut buffer, 
              image::codecs::png::CompressionType::Fast,
              image::codecs::png::FilterType::NoFilter
          );
          
          let encode_result = encoder.write_image(
              frame.as_raw(),
              frame.width(),
              frame.height(),
              image::ExtendedColorType::L8
          );
          
          let encode_duration = start_encode_time.elapsed();
          
          if let Ok(_) = encode_result {
              // FASE 2: ESCRITURA A DISCO
              let write_start = std::time::Instant::now();
              let output_path = output_dir_clone.join(format!("frame_{}.png", index));
              
              if let Ok(_) = std::fs::write(&output_path, &buffer) {
                  let write_duration = write_start.elapsed();
                  let io_duration = encode_duration + write_duration;
                  
                  // Tiempo total desde el inicio del frame
                  let real_total = frame_start_time.elapsed();
                  
                  // Tiempo de espera (diferencia entre el tiempo real y los tiempos de procesamiento)
                  let wait_time = real_total - (prepare_time + compute_time + transfer_time + io_duration);
                  
                  println!(
                      "Frame {} guardado: Total={:.2?} (Activo={:.2?}, Espera={:.2?}) | GPU={:.2?}, I/O={:.2?} (Encode={:.2?}, Write={:.2?})",
                      index,
                      real_total,
                      prepare_time + compute_time + transfer_time + io_duration,
                      wait_time,
                      compute_time + transfer_time,
                      io_duration,
                      encode_duration,
                      write_duration
                  );
                  
                  frames_saved += 1;
              }
          }
      }
      println!("Hilo de guardado finalizado. Se guardaron {} frames.", frames_saved);
  });
  
  // Si no hay frames para procesar, salir
  if frames_qrs.is_empty() {
      return;
  }
  
  // Asumimos que todos los QRs tienen el mismo tamaño
  let qr_width = frames_qrs[0][0].width();
  let qr_height = frames_qrs[0][0].height();
  let frame_width = qr_width * 3;
  let frame_height = qr_height * 2;
  
  // Procesar cada frame y enviar al hilo de guardado
  for (batch_index, qrs) in frames_qrs.iter().enumerate() {
      let absolute_batch_index = start_index + batch_index;
      
      if qrs.is_empty() {
          continue;
      }
      
      // Iniciar cronómetro para el frame completo
      let frame_start_time = std::time::Instant::now();
      println!("Procesando frame {} con {} QRs", absolute_batch_index, qrs.len());
      
      // ===== FASE 1: PREPARACIÓN =====
      let start_time_prepare = std::time::Instant::now();
      
      // Preparar datos para este frame específico
      let mut qr_data = Vec::with_capacity((qr_width * qr_height * qrs.len() as u32) as usize);
      
      for qr in qrs {
          qr_data.extend(
              qr.as_raw().iter().map(|&pixel| {
                  let gray = pixel as u32;
                  (255 << 24) | (gray << 16) | (gray << 8) | gray
              })
          );
      }
      
      // Crear buffers WGPU
      let qr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: Some(&format!("QR Buffer Frame {}", absolute_batch_index)),
          contents: bytemuck::cast_slice(&qr_data),
          usage: wgpu::BufferUsages::STORAGE,
      });
              // En tu función process_frames_batch:
      let scale_factor = 2;  // Escalar a la mitad (1/2)
      let scaled_width = frame_width / scale_factor;
      let scaled_height = frame_height / scale_factor;
      // Crear buffer para la imagen escalada en lugar de la original
      let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
          label: Some(&format!("Frame Buffer {}", absolute_batch_index)),
          // El buffer ahora es más pequeño, para la imagen escalada
          size: (scaled_width * scaled_height * 4) as u64,
          usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
          mapped_at_creation: false,
      });

      // Actualizar UniformParams
      let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: Some(&format!("Uniform Buffer {}", absolute_batch_index)),
          contents: bytemuck::bytes_of(&UniformParams {
              qr_width,
              qr_height,
              frame_width,
              frame_height,
              qrs_count: qrs.len() as u32,
              qr_start_index: 0,
              scaled_width,
              scaled_height,
              scale_factor,
          }),
          usage: wgpu::BufferUsages::UNIFORM,
      });

      // Crear shader
      let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
          label: Some(&format!("Compute Shader {}", absolute_batch_index)),
          source: wgpu::ShaderSource::Wgsl(shader_source.into()),
      });
      
      // Bind group layout
      let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
          label: Some(&format!("Bind Group Layout {}", absolute_batch_index)),
          entries: &[
              wgpu::BindGroupLayoutEntry {
                  binding: 0,
                  visibility: wgpu::ShaderStages::COMPUTE,
                  ty: wgpu::BindingType::Buffer {
                      ty: wgpu::BufferBindingType::Storage { read_only: true },
                      has_dynamic_offset: false,
                      min_binding_size: None,
                  },
                  count: None,
              },
              wgpu::BindGroupLayoutEntry {
                  binding: 1,
                  visibility: wgpu::ShaderStages::COMPUTE,
                  ty: wgpu::BindingType::Buffer {
                      ty: wgpu::BufferBindingType::Storage { read_only: false },
                      has_dynamic_offset: false,
                      min_binding_size: None,
                  },
                  count: None,
              },
              wgpu::BindGroupLayoutEntry {
                  binding: 2,
                  visibility: wgpu::ShaderStages::COMPUTE,
                  ty: wgpu::BindingType::Buffer {
                      ty: wgpu::BufferBindingType::Uniform,
                      has_dynamic_offset: false,
                      min_binding_size: None,
                  },
                  count: None,
              },
          ],
      });
      
      // Bind group
      let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
          layout: &bind_group_layout,
          entries: &[
              wgpu::BindGroupEntry {
                  binding: 0,
                  resource: qr_buffer.as_entire_binding(),
              },
              wgpu::BindGroupEntry {
                  binding: 1,
                  resource: frame_buffer.as_entire_binding(),
              },
              wgpu::BindGroupEntry {
                  binding: 2,
                  resource: uniform_buffer.as_entire_binding(),
              },
          ],
          label: Some(&format!("Bind Group {}", absolute_batch_index)),
      });
      
      // Pipeline layout
      let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
          label: Some(&format!("Pipeline Layout {}", absolute_batch_index)),
          bind_group_layouts: &[&bind_group_layout],
          push_constant_ranges: &[],
      });
      
      // Compute pipeline
      let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
          label: Some(&format!("Compute Pipeline {}", absolute_batch_index)),
          layout: Some(&pipeline_layout),
          module: &shader,
          entry_point: Some("main"),
          compilation_options: Default::default(),
          cache: None,
      });
      
      // Calcular número de workgroups
      let workgroup_count_x = (scaled_width + 31) / 32;
      let workgroup_count_y = (scaled_height + 7) / 8;
      
      let prepare_duration = start_time_prepare.elapsed();
  
      // ===== FASE 2: CÓMPUTO EN GPU =====
      let start_time_compute = std::time::Instant::now();

      // Encoder y ejecución del shader
      let mut compute_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
          label: Some(&format!("Compute Encoder {}", absolute_batch_index)),
      });

      {
          let mut compute_pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
              label: Some(&format!("Compute Pass {}", absolute_batch_index)),
              ..Default::default()
          });
          compute_pass.set_pipeline(&compute_pipeline);
          compute_pass.set_bind_group(0, &bind_group, &[]);
          compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
      }

      // Enviar los comandos de computación
      queue.submit(std::iter::once(compute_encoder.finish()));

      let compute_duration = start_time_compute.elapsed();

      // ===== FASE 3: TRANSFERENCIA DE DATOS =====
      // Crear buffer de lectura y preparar transferencia
      let result_buffer_size = (scaled_width * scaled_height * 4) as u64;
      let result_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
          label: Some(&format!("Result Buffer {}", absolute_batch_index)),
          size: result_buffer_size,
          usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
          mapped_at_creation: false,
      });

      // Crear un NUEVO encoder para la operación de copia
      let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
          label: Some(&format!("Copy Encoder {}", absolute_batch_index)),
      });

      // Añadir comandos de copia al nuevo encoder
      copy_encoder.copy_buffer_to_buffer(
          &frame_buffer, 0, 
          &result_staging_buffer, 0, 
          result_buffer_size
      );

      // Enviar los comandos de copia
      queue.submit(std::iter::once(copy_encoder.finish()));

      // ===== FASE 4: LECTURA DE RESULTADOS =====
      let start_time_read = std::time::Instant::now();
      
      // Mapear buffer y esperar resultados
      let buffer_slice = result_staging_buffer.slice(..);
      let (sender, receiver) = futures::channel::oneshot::channel();
      
      buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
          sender.send(result).unwrap();
      });
      
      // Esto es un punto de sincronización crítico - espera a que la GPU termine
      device.poll(wgpu::Maintain::Wait);
      
      if let Ok(_) = receiver.await {
          // Leer datos del buffer
          let data = buffer_slice.get_mapped_range();
          let frame_data_u32: &[u32] = bytemuck::cast_slice(&*data);
          let frame_data_u8: Vec<u8> = frame_data_u32.iter()
              .map(|&rgba| ((rgba >> 16) & 0xFF) as u8)
              .collect();
          
          // Crear imagen
          let full_frame = GrayImage::from_raw(scaled_width, scaled_height, frame_data_u8)
           .expect("No se pudo crear la imagen desde los datos de la GPU");
          
          // ===== FASE 5: PROCESAMIENTO DE IMAGEN =====
          let start_time_scaling = std::time::Instant::now();
          
          // Reducir tamaño de la imagen
          
          let scaling_duration = start_time_scaling.elapsed();
          let read_duration = start_time_read.elapsed() - scaling_duration;
          
          // Medir tiempo total de procesamiento GPU
          let gpu_duration = compute_duration + read_duration;
          
          // Log detallado en consola
          println!("Frame {} completado: Preparación={:.2?}, Cómputo={:.2?}, Lectura={:.2?}, Escalado={:.2?}, Total GPU={:.2?}",
              absolute_batch_index,
              prepare_duration,
              compute_duration,
              read_duration,
              scaling_duration,
              gpu_duration
          );
          
          // Enviar imagen reducida al hilo de guardado
          tx.send((
              full_frame,
              absolute_batch_index,
              prepare_duration,
              compute_duration,
              read_duration + scaling_duration,
              frame_start_time
          )).expect("Error al enviar frame al hilo de guardado");
      }
      
      // Liberar memoria explícitamente
      drop(qr_data);
  }
  
  // Cerrar el canal para que el hilo de guardado finalice
  drop(tx);
  
  // Esperar a que finalice el guardado
  save_thread.join().expect("Error al esperar el hilo de guardado");
}

