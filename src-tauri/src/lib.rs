pub mod dedup;
pub mod edit;
pub mod face;
pub mod library;
pub mod vision;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Ensure the ONNX Runtime DLL is locatable at runtime even when the app
    // isn't launched through cargo (which sets this via .cargo/config.toml).
    // TODO(bundle): resolve relative to the resource dir for a shipped build.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        let dll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("runtime")
            .join("onnxruntime.dll");
        std::env::set_var("ORT_DYLIB_PATH", dll);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            vision::analyze_image,
            face::analyze_face_eyes,
            dedup::find_duplicate_clusters,
            library::analyze_library,
            library::analyze_files,
            library::export_selects,
            library::preview_edit,
            library::large_preview,
            library::read_exif,
            library::list_presets,
            library::save_preset,
            library::delete_preset
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
