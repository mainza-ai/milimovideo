#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|error| error.to_string())
}

#[tauri::command]
fn start_backend(app: tauri::AppHandle) -> Result<String, String> {
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use tauri::Manager;

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(root) = std::env::var("MILIMO_REPO_ROOT") {
        candidates.push(PathBuf::from(root));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.clone());
        for ancestor in cwd.ancestors().skip(1).take(5) {
            candidates.push(ancestor.to_path_buf());
        }
    }
    if let Ok(resource) = app.path().resource_dir() {
        candidates.push(resource.clone());
        for ancestor in resource.ancestors().skip(1).take(4) {
            candidates.push(ancestor.to_path_buf());
        }
    }

    for root in candidates {
        let log_path = root.join("backend.desktop.log");
        let open_log = || {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
        };

        // Prefer a bundled sidecar binary when one is shipped. No binary is
        // currently bundled, but this makes a packaged runtime a drop-in.
        for name in ["milimo-backend", "milimo-backend.exe", "bin/milimo-backend"] {
            let sidecar = root.join(name);
            if sidecar.is_file() {
                let stdout = open_log().map_err(|error| error.to_string())?;
                let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
                Command::new(&sidecar)
                    .current_dir(&root)
                    .stdout(Stdio::from(stdout))
                    .stderr(Stdio::from(stderr))
                    .spawn()
                    .map_err(|error| error.to_string())?;
                return Ok(root.to_string_lossy().to_string());
            }
        }

        let script = root.join("run_backend.sh");
        if script.exists() {
            let stdout = open_log().map_err(|error| error.to_string())?;
            let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
            Command::new("bash")
                .arg(&script)
                .current_dir(&root)
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr))
                .spawn()
                .map_err(|error| error.to_string())?;
            return Ok(root.to_string_lossy().to_string());
        }
    }

    Err("Could not find run_backend.sh. Start the backend manually or set MILIMO_REPO_ROOT.".into())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![save_text_file, start_backend])
        .run(tauri::generate_context!())
        .expect("error while running Milimo Video");
}
