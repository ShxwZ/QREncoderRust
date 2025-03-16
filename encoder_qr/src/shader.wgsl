// Definición de los buffers
struct QRImages {
    data: array<u32>,
};

@group(0) @binding(0)
var<storage, read> qr_images: QRImages;

struct Frame {
    data: array<u32>,
};

@group(0) @binding(1)
var<storage, read_write> frame: Frame;

struct Params {
    qr_width: u32,
    qr_height: u32,
    frame_width: u32,
    frame_height: u32,
    qrs_count: u32,
    qr_start_index: u32,
    
    // Nuevo: tamaño del frame escalado
    scaled_width: u32,
    scaled_height: u32,
    scale_factor: u32,    // Por ejemplo, 2 para escalar a la mitad (1/2)
};

@group(0) @binding(2)
var<uniform> params: Params;

// Cambiamos a un solo dispatch 2D para todo el frame ESCALADO
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Estas son las coordenadas en el frame escalado
    let scaled_x = id.x;
    let scaled_y = id.y;
    
    // Si estamos fuera de los límites del frame escalado, terminar
    if scaled_x >= params.scaled_width || scaled_y >= params.scaled_height {
        return;
    }
    
    // Calcular las coordenadas originales en el frame sin escalar
    let x = scaled_x * params.scale_factor;
    let y = scaled_y * params.scale_factor;
    
    // Color blanco (RGBA: 255,255,255,255)
    let COLOR_BLANCO = 0xFFFFFFFFu;
    
    // Calcular en qué QR estamos (usando coordenadas originales)
    let qr_x = x / params.qr_width;
    let qr_y = y / params.qr_height;
    
    // Si estamos fuera de la cuadrícula de QRs 3x2, terminar
    if qr_x >= 3u || qr_y >= 2u {
        frame.data[scaled_y * params.scaled_width + scaled_x] = COLOR_BLANCO;
        return;
    }
    
    // Calcular el índice del QR (0-5)
    let qr_index = qr_y * 3u + qr_x;
    
    // Si este QR no existe, terminar
    if qr_index >= params.qrs_count {
        frame.data[scaled_y * params.scaled_width + scaled_x] = COLOR_BLANCO;
        return;
    }
    
    // Calcular la posición relativa dentro del QR actual
    let local_x = x % params.qr_width;
    let local_y = y % params.qr_height;
    
    // Calcular el índice ajustado con el offset
    let adjusted_qr_index = params.qr_start_index + qr_index;
    
    // Calcular el índice del píxel en el QR source con el índice ajustado
    let qr_pixel_index = adjusted_qr_index * (params.qr_width * params.qr_height) + 
                       (local_y * params.qr_width + local_x);
    
    // Copiar el valor del QR al frame ESCALADO
    // Nota que el índice de destino usa las coordenadas escaladas
    frame.data[scaled_y * params.scaled_width + scaled_x] = qr_images.data[qr_pixel_index];
}