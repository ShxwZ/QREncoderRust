use bardecoder;
use image::io::Reader as ImageReader;
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn scan_qr_code(qr_image_path: String, fragmento_original: String) -> Result<(), Box<dyn Error>> {
    // Cargar la imagen del código QR
    let img = ImageReader::open(&qr_image_path)?.decode()?;

    // Crear el decodificador predeterminado
    let decoder = bardecoder::default_decoder();

    // Decodificar el código QR
    let results = decoder.decode(&img);
    for result in results {
        match result {
            Ok(data) => {
                // Decodificar el fragmento original (Base64) para obtener los bytes originales
                // Comparar el contenido decodificado con el fragmento original
                if data.as_str() == fragmento_original {
                    println!(
                        "El contenido del código QR en {} coincide con el fragmento original.",
                        qr_image_path
                    );
                } else {
                    println!(
                        "El contenido del código QR en {} NO coincide con el fragmento original.",
                        qr_image_path
                    );
                    // Opcional: Guardar el contenido decodificado para análisis
                    let mut file = File::create("contenido_decodificado.txt")?;
                    file.write_all(data.as_bytes())?;
                }
            }
            Err(e) => {
                println!(
                    "Error al decodificar el código QR en {}: {}",
                    qr_image_path, e
                );
            }
        }
    }
    Ok(())
}
