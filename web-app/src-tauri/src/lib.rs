#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|error| error.to_string())
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|error| error.to_string())
}

#[tauri::command]
fn set_badge(window: tauri::Window, label: Option<String>, count: Option<i64>) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        window
            .set_badge_label(label)
            .map_err(|error| error.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = label;
    }
    window
        .set_badge_count(count)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_window_title(window: tauri::Window, title: String) -> Result<(), String> {
    window.set_title(&title).map_err(|error| error.to_string())
}

#[tauri::command]
fn notify(title: String, body: String) -> Result<(), String> {
    // macOS: use Notification Center through osascript. Arguments are passed
    // through argv, so a title/body cannot inject AppleScript.
    #[cfg(target_os = "macos")]
    {
        let script = "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run";
        let status = std::process::Command::new("osascript")
            .arg("-e")
            .arg(script)
            .arg(&title)
            .arg(&body)
            .status()
            .map_err(|error| error.to_string())?;
        if !status.success() {
            return Err(format!("osascript exited with {status}"));
        }
        return Ok(());
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (title, body);
        Err("desktop notifications are not implemented on this platform".into())
    }
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

#[tauri::command]
fn set_tray_tooltip(app: tauri::AppHandle, tooltip: String) -> Result<(), String> {
    if let Some(tray) = app.tray_by_id("main-tray") {
        tray.set_tooltip(Some(&tooltip)).map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            use tauri::menu::{Menu, MenuItem};
            use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
            use tauri::Manager;

            let show = MenuItem::with_id(app, "show", "Show Milimo Video", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            let mut builder = TrayIconBuilder::with_id("main-tray")
                .menu(&menu)
                .tooltip("Milimo Video")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        if let Some(window) = tray.app_handle().get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                builder = builder.icon(icon.clone());
            }
            builder.build(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            save_text_file,
            read_text_file,
            start_backend,
            notify,
            set_window_title,
            set_badge,
            set_tray_tooltip,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Milimo Video");
}
