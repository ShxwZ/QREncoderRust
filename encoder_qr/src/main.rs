use std::fs;
use std::path::Path;
use std::io::{self};
use image::GrayImage;
use image::Luma;
use rayon::prelude::*;
use std::env::*;
use std::sync::mpsc;
use futures::executor::block_on;
use zstd::stream::encode_all;
use tokio::io::AsyncReadExt;
// Custom imports
mod image_utils;
mod qr_utils;
mod gpu_utils;
mod frames_utils;
use gpu_utils::*;
use image_utils::*;
use qr_utils::*;
use frames_utils::*;

fn main() -> io::Result<()> {
    // Obtener argumentos de línea de comandos
    let args: Vec<String> = Args::collect(self::args());
    let start_frame_index = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };
    
    // Configuración de rutas
    let input_file = Path::new("C:\\Users\\Shaw\\Desktop\\QREncoderRust\\resources\\1gb.mp4");
    let output_dir = Path::new("C:\\Users\\Shaw\\Desktop\\QREncoderRust\\qrs");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }
    
    // Cargar el shader una sola vez
    let shader_source = fs::read_to_string("C:\\Users\\Shaw\\Desktop\\QREncoderRust\\encoder_qr\\src\\shader.wgsl")
        .expect("No se pudo leer el archivo WGSL");
    
    // Inicializar GPU
    let (device, queue) = block_on(initialize_wgpu());
    
    // Procesar el archivo por chunks para reducir uso de memoria
    println!("[INFO] Iniciando procesamiento del archivo: {}", input_file.display());
    
    // Configuración de procesamiento
    let fragment_size = 2800; 
    let frames_per_batch = 60;
    let chunks_per_frame = 6;
    let max_frames = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(usize::MAX)
    } else {
        usize::MAX
    };
    
    // Crear un runtime de Tokio optimizado
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)  // Ajusta según tu CPU
        .enable_all()
        .build()
        .unwrap();
    
    runtime.block_on(async {
        // Abrir archivo para streaming en lugar de cargarlo todo en memoria
        let file = tokio::fs::File::open(input_file).await?;
        let file_size = file.metadata().await?.len();
        
        println!("[INFO] Tamaño del archivo: {} bytes", file_size);
        
        // Calcular cuántos frames aproximados hay y procesar por lotes
        let estimated_total_frames = (file_size as f64 / (fragment_size * chunks_per_frame) as f64) as usize;
        println!("[INFO] Número estimado de frames: {}", estimated_total_frames);
        
        // Determinar cuántos frames procesar en esta ejecución
        let end_frame_index = (start_frame_index + max_frames).min(estimated_total_frames);
        println!("[INFO] Procesando frames {} a {}", start_frame_index, end_frame_index - 1);
        
        // Procesar el archivo en streaming
        let mut reader = tokio::io::BufReader::new(file);
        let mut buffer = vec![0u8; fragment_size * chunks_per_frame * frames_per_batch];
        
        // Saltar al punto de inicio si es necesario
        if start_frame_index > 0 {
            use tokio::io::AsyncSeekExt;
            
            let skip_bytes = start_frame_index * 6 * fragment_size;
            reader.seek(std::io::SeekFrom::Start(skip_bytes as u64)).await?;
            println!("[INFO] Saltados {} bytes para comenzar en el frame {}", skip_bytes, start_frame_index);
        }
        
        // Procesar por lotes para eficiencia
        let mut current_frame = start_frame_index;
        let mut batch_index = 0;
        
        while current_frame < end_frame_index {
            batch_index += 1;
            
            // Leer el siguiente lote de datos
            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                println!("[INFO] Final del archivo alcanzado");
                break;
            }
            
            // Comprimir este lote en lugar de comprimir todo el archivo
            let compressed_data = encode_all(&buffer[..bytes_read], 5).unwrap();
            
            // Dividir en chunks para los frames
            let chunks: Vec<Vec<u8>> = compressed_data
                .chunks(fragment_size)
                .map(|chunk| chunk.to_vec())
                .collect();
                
            let frames_count = chunks.len() / chunks_per_frame;
            
            println!("[BATCH {}] Procesando {} frames", batch_index, frames_count);
            
            // Generar QRs para este lote
            let frame_chunks: Vec<Vec<Vec<u8>>> = chunks
                .chunks(chunks_per_frame)
                .map(|c| c.to_vec())
                .collect();
                
            let qrs_for_batch: Vec<Vec<GrayImage>> = frame_chunks
                .par_iter()
                .map(|frame_chunks| {
                    frame_chunks.iter()
                        .filter_map(|chunk| generate_qrg_image_gray(chunk).ok())
                        .collect()
                })
                .collect();
                
            // Procesar frames en GPU
            process_frames_batch(
                &device,
                &queue,
                &shader_source,
                &qrs_for_batch,
                &output_dir,
                current_frame
            ).await;
            
            // Actualizar conteo de frames
            current_frame += frames_count;
            
            println!("[PROGRESO] Completados {}/{} frames", 
                     (current_frame - start_frame_index), 
                     (end_frame_index - start_frame_index));
        }
        
        println!("\n[COMPLETADO] Procesamiento finalizado para los frames {} a {}.", 
                start_frame_index, current_frame - 1);
        
        if current_frame < estimated_total_frames {
            println!("\nPara continuar con el siguiente lote, ejecuta:");
            println!("cargo run -- {}", current_frame);
        }
        
        Ok(()) as io::Result<()>
    })?;
    
    Ok(())
}

async fn process_frames_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_source: &str,
    frames_qrs: &[Vec<GrayImage>],
    output_dir: &Path,
    start_index: usize,
) {
    create_output_directory(output_dir);

    let (tx, rx) = create_channel();
    let save_thread = spawn_save_thread(rx, output_dir.to_path_buf());

    if frames_qrs.is_empty() {
        return;
    }

    let (frame_width, frame_height) = get_frame_dimensions(frames_qrs);

    for (batch_index, qrs) in frames_qrs.iter().enumerate() {
        let absolute_batch_index = start_index + batch_index;

        if qrs.is_empty() {
            continue;
        }

        let frame_start_time = std::time::Instant::now();

        let qr_data = prepare_qr_data(qrs);

        let (qr_buffer, frame_buffer, uniform_buffer) = create_buffers(
            device, &qr_data, frame_width, frame_height, qrs.len() as u32, &absolute_batch_index
        );

        let (bind_group, compute_pipeline) = create_bind_group_and_pipe_layout(device, &qr_buffer, &frame_buffer, &uniform_buffer, shader_source,&absolute_batch_index);


        let (compute_time, transfer_time,result_staging_buffer) = run_compute_shader(
            device, queue, &bind_group, &compute_pipeline, &frame_buffer, frame_width, frame_height, &absolute_batch_index
        );

        let frame = read_frame_data(device,result_staging_buffer, frame_width, frame_height).await;

        tx.send((frame, absolute_batch_index, compute_time, transfer_time, frame_start_time))
            .expect("Failed to send frame");
    }

    drop(tx);
    save_thread.join().expect("Failed to join save thread");
}



fn create_channel() -> (
    mpsc::Sender<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>, 
    mpsc::Receiver<(image::ImageBuffer<Luma<u8>, Vec<u8>>, usize, std::time::Duration, std::time::Duration, std::time::Instant)>
) {
    return  mpsc::channel::<(
        image::ImageBuffer<image::Luma<u8>, Vec<u8>>, // GrayImage
        usize,                                        // índice del frame
        std::time::Duration,                          // tiempo de cómputo
        std::time::Duration,                          // tiempo de transferencia
        std::time::Instant                            // tiempo de inicio del frame
    )>();
}

