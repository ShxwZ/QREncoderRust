
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformParams {
    qr_width: u32,
    qr_height: u32,
    frame_width: u32,
    frame_height: u32,
    qrs_count: u32,
    qr_start_index: u32,
    scaled_width: u32,   // Nuevo: ancho escalado
    scaled_height: u32,  // Nuevo: alto escalado
    scale_factor: u32,   // Nuevo: factor de escala (ejemplo: 2 para 50%)
}

pub async fn initialize_wgpu() -> (wgpu::Device, wgpu::Queue) {
  let instance = wgpu::Instance::default();
  
  // Mostrar todos los adaptadores disponibles
  let adapters = instance.enumerate_adapters(wgpu::Backends::all());
  println!("Adaptadores disponibles:");
  for (i, adapter) in adapters.iter().enumerate() {
      let info = adapter.get_info();
      println!("[{}] {} / {}", i, info.name, info.backend.to_str());
  }
  
  // Solicitar específicamente un adaptador de alto rendimiento (GPU dedicada)
  let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
          power_preference: wgpu::PowerPreference::HighPerformance,
          compatible_surface: None,
          force_fallback_adapter: false,
      })
      .await
      .expect("No se encontró un adaptador de alto rendimiento");
  
  // Mostrar información del adaptador seleccionado
  let info = adapter.get_info();
  println!("Usando adaptador: {} / {}", info.name, info.backend.to_str());
  
  let (device, queue) = adapter
      .request_device(
          &wgpu::DeviceDescriptor {
              label: None,
              required_features: wgpu::Features::empty(),
              required_limits: wgpu::Limits::default(),
              ..Default::default()
          },
          None,
      )
      .await
      .expect("Error al crear el dispositivo");
      
  println!("Dispositivo GPU inicializado correctamente");
  
  (device, queue)
}


pub fn create_bind_group_and_pipe_layout(
    device: &wgpu::Device,
    qr_buffer: &wgpu::Buffer,
    frame_buffer: &wgpu::Buffer,
    uniform_buffer: &wgpu::Buffer,
    shader_source: &str,
    absolute_batch_index: &usize,
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
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
    
    let compute_pipeline = create_compute_pipeline(device, shader_source, bind_group_layout, absolute_batch_index);
    return ( bind_group, compute_pipeline);
}

pub fn create_compute_pipeline(device: &wgpu::Device, shader_source: &str, bind_group_layout: wgpu::BindGroupLayout, absolute_batch_index: &usize) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("Compute Shader {}", absolute_batch_index)),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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
    return compute_pipeline;
}

pub fn run_compute_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group: &wgpu::BindGroup,
    compute_pipeline: &wgpu::ComputePipeline,
    frame_buffer: &wgpu::Buffer,
    frame_width: u32,
    frame_height: u32,
    absolute_batch_index: &usize
) -> (std::time::Duration, std::time::Duration, wgpu::Buffer) {

    // COMPUTACIÓN
    let scale_factor = 2;  // Escalar a la mitad (1/2)
    let scaled_width = frame_width / scale_factor;
    let scaled_height = frame_height / scale_factor;
    let workgroup_count_x = (frame_width + 15) / 32;
    let workgroup_count_y = (frame_height + 7) / 8;
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
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    // Enviar los comandos de computación
    queue.submit(std::iter::once(compute_encoder.finish()));

    let compute_duration = start_time_compute.elapsed();

    // TRANSFERENCIA DE DATOS
    // Crear buffer de lectura y preparar transferencia
    let start_time_transfer = std::time::Instant::now();

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
    
    let transfer_duration = start_time_transfer.elapsed();

    return (compute_duration, transfer_duration,result_staging_buffer);
}

pub fn create_buffers(
    device: &wgpu::Device,
    qr_data: &[u32],
    frame_width: u32,
    frame_height: u32,
    qrs_count: u32,
    absolute_batch_index: &usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {

    let qr_width = frame_width/3;
    let qr_height = frame_height/2;
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
                    qrs_count,
                    qr_start_index: 0,
                    scaled_width,
                    scaled_height,
                    scale_factor,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
    });
    return (qr_buffer, frame_buffer, uniform_buffer);
}

