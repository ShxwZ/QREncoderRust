use std::fs;
use std::path::Path;
use std::io::{self};
use std::sync::Arc;
use image::GrayImage;
use image::Luma;
use rayon::prelude::*;
use std::env::*;
use tokio::sync::mpsc;
use futures::executor::block_on;
use zstd::stream::encode_all;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
// Custom imports
mod image_utils;
mod qr_utils;
mod gpu_utils;
mod frames_utils;
use gpu_utils::*;
use image_utils::*;
use qr_utils::*;
use frames_utils::*;

struct AppConfig {
    start_frame_index: usize,
    max_frames: usize,
    input_file: String,
    output_dir: String,
    shader_path: String,
    fragment_size: usize,
    frames_per_batch: usize,
    chunks_per_frame: usize,
    worker_threads: usize,
}

fn main() -> io::Result<()> {
    // 1. Parsear argumentos y configurar la aplicación
    let config = parse_arguments();
    
    // 2. Inicializar recursos
    let (device, queue) = block_on(initialize_wgpu());
    let device = std::sync::Arc::new(device);  // Envolver en Arc
    let queue = std::sync::Arc::new(queue);    // Envolver en Arc
    let shader_source = load_shader(&config.shader_path)?;
    initialize_output_directory(&config.output_dir)?;
    
    // 3. Procesar el archivo
    process_file(&config, &device, &queue, &shader_source)?;
    
    Ok(())
}

fn parse_arguments() -> AppConfig {
    // Obtener argumentos de línea de comandos
    let args: Vec<String> = Args::collect(self::args());
    let start_frame_index = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };
    
    let max_frames = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(usize::MAX)
    } else {
        usize::MAX
    };
    
    AppConfig {
        start_frame_index,
        max_frames,
        input_file: "C:\\Users\\Shaw\\Desktop\\QREncoderRust\\resources\\1gb.mp4".to_string(),
        output_dir: "C:\\Users\\Shaw\\Desktop\\QREncoderRust\\qrs".to_string(),
        shader_path: "C:\\Users\\Shaw\\Desktop\\QREncoderRust\\encoder_qr\\src\\shader.wgsl".to_string(),
        fragment_size: 2800,
        frames_per_batch: 120,
        chunks_per_frame: 6,
        worker_threads: 6,
    }
}

fn initialize_output_directory(output_dir: &str) -> io::Result<()> {
    let path = Path::new(output_dir);
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn load_shader(shader_path: &str) -> io::Result<String> {
    fs::read_to_string(shader_path)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, 
                 format!("No se pudo leer el archivo WGSL: {}", e)))
}

fn process_file(
    config: &AppConfig, 
    device: &std::sync::Arc<wgpu::Device>, 
    queue: &std::sync::Arc<wgpu::Queue>,
    shader_source: &str
) -> io::Result<()> {
    // Crear un runtime de Tokio optimizado
    let runtime = build_tokio_runtime(config.worker_threads);
    
    runtime.block_on(async {
        // 1. Abrir y analizar el archivo
        let (_, estimated_total_frames, end_frame_index) = 
            open_and_analyze_file(&config).await?;
        
        // 2. Preparar el lector y buffer
        let (mut reader, mut buffer) = prepare_reader_and_buffer(&config).await?;
        
        // 3. Saltar al punto de inicio si es necesario
        if config.start_frame_index > 0 {
            skip_to_starting_position(&mut reader, config).await?;
        }
        
        // 4. Procesar los frames por lotes
        let current_frame = process_frames_in_batches(
            &mut reader, 
            &mut buffer,
            device, 
            queue, 
            shader_source,
            &config,
            end_frame_index
        ).await?;
        
        // 5. Mostrar resumen y opciones para continuar
        display_summary(config.start_frame_index, current_frame, estimated_total_frames);
        
        Ok(()) as io::Result<()>
    })
}

fn build_tokio_runtime(worker_threads: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .unwrap()
}

async fn open_and_analyze_file(config: &AppConfig) -> io::Result<(u64, usize, usize)> {
    // Abrir archivo para streaming
    let file = tokio::fs::File::open(&config.input_file).await?;
    let file_size = file.metadata().await?.len();
    
    println!("[INFO] Tamaño del archivo: {} bytes", file_size);
    
    // Calcular cuántos frames aproximados hay
    let estimated_total_frames = (file_size as f64 / 
        (config.fragment_size * config.chunks_per_frame) as f64) as usize;
    println!("[INFO] Número estimado de frames: {}", estimated_total_frames);
    
    // Determinar cuántos frames procesar en esta ejecución
    let end_frame_index = (config.start_frame_index + config.max_frames)
        .min(estimated_total_frames);
    println!("[INFO] Procesando frames {} a {}", 
             config.start_frame_index, end_frame_index - 1);
             
    Ok((file_size, estimated_total_frames, end_frame_index))
}

async fn prepare_reader_and_buffer(
    config: &AppConfig
) -> io::Result<(tokio::io::BufReader<tokio::fs::File>, Vec<u8>)> {
    let file = tokio::fs::File::open(&config.input_file).await?;
    let reader = tokio::io::BufReader::new(file);
    
    // Crear buffer para leer datos
    let buffer_size = config.fragment_size * config.chunks_per_frame * config.frames_per_batch;
    let buffer = vec![0u8; buffer_size];
    
    Ok((reader, buffer))
}

async fn skip_to_starting_position(
    reader: &mut tokio::io::BufReader<tokio::fs::File>,
    config: &AppConfig
) -> io::Result<()> {
    let skip_bytes = config.start_frame_index * config.chunks_per_frame * config.fragment_size;
    reader.seek(std::io::SeekFrom::Start(skip_bytes as u64)).await?;
    
    println!("[INFO] Saltados {} bytes para comenzar en el frame {}", 
             skip_bytes, config.start_frame_index);
             
    Ok(())
}

async fn process_frames_in_batches(
    reader: &mut tokio::io::BufReader<tokio::fs::File>,
    buffer: &mut Vec<u8>,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    shader_source: &str,
    config: &AppConfig,
    end_frame_index: usize
) -> io::Result<usize> {
    let mut current_frame = config.start_frame_index;
    let mut batch_index = 0;
    let output_dir = Path::new(&config.output_dir);
    
    while current_frame < end_frame_index {
        batch_index += 1;
        
        // Procesar un lote
        let frames_processed = process_single_batch(
            reader, buffer, device, queue, shader_source,
            config, output_dir, current_frame, batch_index
        ).await?;
        
        if frames_processed == 0 {
            println!("[INFO] Final del archivo alcanzado");
            break;
        }
        
        // Actualizar conteo de frames
        current_frame += frames_processed;
        
        println!("[PROGRESO] Completados {}/{} frames", 
                 (current_frame - config.start_frame_index), 
                 (end_frame_index - config.start_frame_index));
    }
    
    Ok(current_frame)
}

async fn process_single_batch(
    reader: &mut tokio::io::BufReader<tokio::fs::File>,
    buffer: &mut Vec<u8>,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    shader_source: &str,
    config: &AppConfig,
    output_dir: &Path,
    current_frame: usize,
    batch_index: usize
) -> io::Result<usize> {
    // 1. Leer datos con lectura completa del buffer
    let start_read = std::time::Instant::now();
    let mut bytes_read = 0;
    let target_bytes = buffer.len();
    
    // Realizar múltiples lecturas hasta llenar el buffer o llegar al EOF
    while bytes_read < target_bytes {
        let read_result = reader.read(&mut buffer[bytes_read..]).await?;
        if read_result == 0 {
            // EOF alcanzado
            break;
        }
        bytes_read += read_result;
    }
    
    let read_time = start_read.elapsed();
    
    if bytes_read == 0 {
        return Ok(0);
    }
    
    // 2. Comprimir datos con nivel óptimo para velocidad/compresión
    let start_compress = std::time::Instant::now();
    let compression_level = 5;  // Más rápido que 5, pero todavía efectivo
    let compressed_data = encode_all(&buffer[..bytes_read], compression_level).unwrap();
    let compression_time = start_compress.elapsed();
    
    let compression_ratio = bytes_read as f64 / compressed_data.len() as f64;
    println!("[DEBUG] Batch {}: Leídos {} bytes en {:.2?}, comprimidos a {} bytes ({:.2}x) en {:.2?}", 
             batch_index, bytes_read, read_time, compressed_data.len(), 
             compression_ratio, compression_time);
    
    // 3. Dividir en chunks para los frames
    let chunks: Vec<Vec<u8>> = compressed_data
        .chunks(config.fragment_size)
        .map(|chunk| chunk.to_vec())
        .collect();
        
    let frames_count = chunks.len() / config.chunks_per_frame;
    
    println!("[BATCH {}] Procesando {} frames (de {} chunks)", 
             batch_index, frames_count, chunks.len());
    
    // 4. Generar QRs
    let start_qr = std::time::Instant::now();
    let chunks_clone = chunks.clone();
    let chunks_per_frame_value = config.chunks_per_frame; // Extraer el valor que necesitamos
    let qrs_future = tokio::task::spawn_blocking(move || {
        generate_qr_codes(&chunks_clone, chunks_per_frame_value) // Usar el valor copiado
    });
    
    let qrs_for_batch = qrs_future.await.expect("Error al generar QRs");
    let qr_time = start_qr.elapsed();
    println!("[DEBUG] Generación de QR para {} frames: {:.2?}", frames_count, qr_time);
    

    
    // 5. Procesar frames en GPU
    process_frames_batch(
        device,
        queue,
        shader_source,
        &qrs_for_batch,
        output_dir,
        current_frame
    ).await;
    
    Ok(frames_count)
}

fn generate_qr_codes(
    chunks: &[Vec<u8>],
    chunks_per_frame: usize
) -> Vec<Vec<GrayImage>> {
    // Agrupar chunks por frames
    let frame_chunks: Vec<Vec<Vec<u8>>> = chunks
        .chunks(chunks_per_frame)
        .map(|c| c.to_vec())
        .collect();
        
    // Generar QRs en paralelo
    frame_chunks
        .par_iter()
        .map(|frame_chunks| {
            frame_chunks.iter()
                .filter_map(|chunk| generate_qrg_image_gray(chunk).ok())
                .collect()
        })
        .collect()
}

fn display_summary(start_frame_index: usize, current_frame: usize, estimated_total_frames: usize) {
    println!("\n[COMPLETADO] Procesamiento finalizado para los frames {} a {}.", 
             start_frame_index, current_frame - 1);
    
    if current_frame < estimated_total_frames {
        println!("\nPara continuar con el siguiente lote, ejecuta:");
        println!("cargo run -- {}", current_frame);
    } else {
        println!("\n¡Procesamiento completo! Se procesaron todos los frames disponibles.");
    }
}

async fn process_frames_batch(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    shader_source: &str,
    frames_qrs: &[Vec<GrayImage>],
    output_dir: &Path,
    start_index: usize,
) {
    create_output_directory(output_dir);

    let (tx, rx) = create_channel().await;
    let save_thread = spawn_save_thread(rx, output_dir.to_path_buf());

    if frames_qrs.is_empty() {
        return;
    }

    let (frame_width, frame_height) = get_frame_dimensions(frames_qrs);

    // Procesar frames en paralelo, limitando a un máximo razonable para no sobrecargar la GPU
    let max_parallel_frames = 6; // Ajusta según tu GPU
    let mut futures = Vec::new();

    for (batch_index, qrs) in frames_qrs.iter().enumerate() {
        let absolute_batch_index = start_index + batch_index;

        if qrs.is_empty() {
            continue;
        }

        // Crear una copia de tx para cada tarea paralela
        let tx_clone = tx.clone();
        let qrs_clone = qrs.clone();
        let device_clone = Arc::clone(device);  // Clonar el Arc, no el device
        let queue_clone = Arc::clone(queue);    // Clonar el Arc, no el queue
        let shader_source_clone = shader_source.to_string(); // Clonar el string
        
        // Crear una tarea para este frame
        let future = tokio::spawn(async move {
            let frame_start_time = std::time::Instant::now();

            let qr_data = prepare_qr_data(&qrs_clone);

            let (qr_buffer, frame_buffer, uniform_buffer) = create_buffers(
                &device_clone, &qr_data, frame_width, frame_height, qrs_clone.len() as u32, &absolute_batch_index
            );

            let (bind_group, compute_pipeline) = create_bind_group_and_pipe_layout(
                &device_clone, &qr_buffer, &frame_buffer, &uniform_buffer, &shader_source_clone, &absolute_batch_index
            );

            let (compute_time, transfer_time, result_staging_buffer) = run_compute_shader(
                &device_clone, &queue_clone, &bind_group, &compute_pipeline, 
                &frame_buffer, frame_width, frame_height, &absolute_batch_index
            );

            let frame = read_frame_data(&device_clone, result_staging_buffer, frame_width, frame_height).await;

            tx_clone.send((frame, absolute_batch_index, compute_time, transfer_time, frame_start_time))
                .await
                .expect("Failed to send frame");
        });

        futures.push(future);

        // Esperar a que algunos frames terminen si hay demasiados en proceso
        if futures.len() >= max_parallel_frames {
            let (_, _, remaining) = futures::future::select_all(futures).await;
            futures = remaining;
        }
    }

    // Esperar a que todos los frames restantes terminen
    for future in futures {
        future.await.expect("Error al procesar frame en paralelo");
    }

    drop(tx);
    save_thread.await.expect("Failed to join save thread");
}

async fn create_channel() -> (
    mpsc::Sender<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>, 
    mpsc::Receiver<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>
) {
    mpsc::channel(500) // Aumentar el tamaño del buffer del canal
}