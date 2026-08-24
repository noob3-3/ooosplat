use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    process::{ProcessManager, ProcessSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Ffmpeg,
    Ffprobe,
    Colmap,
    Brush,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    pub kind: EngineKind,
    pub path: PathBuf,
    pub exists: bool,
    pub can_start: bool,
    pub version: Option<String>,
    pub cpu_only: Option<bool>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct EnginePaths {
    pub root: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub colmap: PathBuf,
    pub brush: PathBuf,
}

impl EnginePaths {
    pub fn from_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            ffmpeg: root.join("ffmpeg").join("ffmpeg.exe"),
            ffprobe: root.join("ffmpeg").join("ffprobe.exe"),
            colmap: root.join("colmap").join("bin").join("colmap.exe"),
            brush: root.join("brush").join("brush_app.exe"),
            root,
        }
    }

    pub fn discover(resource_dir: Option<&Path>) -> Self {
        if let Some(value) = std::env::var_os("OOOSPLAT_ENGINE_DIR") {
            return Self::from_root(value);
        }

        let current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let candidates = [
            resource_dir.map(|path| path.join("engines")),
            Some(current.join("engines")),
            Some(current.join("..").join("engines")),
        ];
        let root = candidates
            .into_iter()
            .flatten()
            .find(|path| path.is_dir())
            .unwrap_or_else(|| current.join("engines"));
        Self::from_root(root)
    }

    pub async fn check_all(&self) -> Vec<EngineStatus> {
        let (ffmpeg, ffprobe, colmap, brush) = tokio::join!(
            check_basic(EngineKind::Ffmpeg, &self.ffmpeg, &["-version"]),
            check_basic(EngineKind::Ffprobe, &self.ffprobe, &["-version"]),
            check_colmap(&self.colmap),
            check_basic(EngineKind::Brush, &self.brush, &["--help"]),
        );
        vec![ffmpeg, ffprobe, colmap, brush]
    }
}

fn missing(kind: EngineKind, path: &Path) -> EngineStatus {
    EngineStatus {
        kind,
        path: path.to_path_buf(),
        exists: false,
        can_start: false,
        version: None,
        cpu_only: None,
        detail: format!("未找到 {}", path.display()),
    }
}

async fn check_basic(kind: EngineKind, path: &Path, args: &[&str]) -> EngineStatus {
    if !path.is_file() {
        return missing(kind, path);
    }
    let manager = ProcessManager::new();
    let result = manager
        .run(ProcessSpec {
            executable: path.to_path_buf(),
            args: args.iter().map(OsString::from).collect(),
            working_directory: path.parent().map(Path::to_path_buf),
            log_path: None,
            observer: None,
        })
        .await;

    match result {
        Ok(output) => {
            let combined = format!("{}\n{}", output.stdout, output.stderr);
            let first_line = combined
                .lines()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_owned());
            EngineStatus {
                kind,
                path: path.to_path_buf(),
                exists: true,
                can_start: output.success,
                version: first_line,
                cpu_only: None,
                detail: if output.success {
                    "引擎可启动".into()
                } else {
                    format!("帮助命令退出码：{:?}", output.exit_code)
                },
            }
        }
        Err(error) => EngineStatus {
            kind,
            path: path.to_path_buf(),
            exists: true,
            can_start: false,
            version: None,
            cpu_only: None,
            detail: error.to_string(),
        },
    }
}

async fn check_colmap(path: &Path) -> EngineStatus {
    if !path.is_file() {
        return missing(EngineKind::Colmap, path);
    }
    let manager = ProcessManager::new();
    let mut help = String::new();
    let mut successful = true;
    for args in [
        vec!["feature_extractor", "-h"],
        vec!["sequential_matcher", "-h"],
        vec!["mapper", "-h"],
    ] {
        match manager
            .run(ProcessSpec {
                executable: path.to_path_buf(),
                args: args.into_iter().map(OsString::from).collect(),
                working_directory: path.parent().map(Path::to_path_buf),
                log_path: None,
                observer: None,
            })
            .await
        {
            Ok(output) => {
                successful &= output.success;
                help.push_str(&output.stdout);
                help.push_str(&output.stderr);
            }
            Err(error) => {
                return EngineStatus {
                    kind: EngineKind::Colmap,
                    path: path.to_path_buf(),
                    exists: true,
                    can_start: false,
                    version: None,
                    cpu_only: None,
                    detail: error.to_string(),
                }
            }
        }
    }

    let lower = help.to_ascii_lowercase();
    let explicit_no_cuda = [
        "cuda: no",
        "cuda support: no",
        "without cuda",
        "no cuda support",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let explicit_cuda = [
        "cuda: yes",
        "cuda support: yes",
        "with cuda",
        "cuda enabled",
        "use_gpu",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let bundled_cuda = path.parent().is_some_and(runtime_contains_cuda);
    let cpu_only = if bundled_cuda || explicit_cuda {
        Some(false)
    } else if explicit_no_cuda {
        Some(true)
    } else {
        None
    };
    let first_line = help
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_owned());
    let detail = match cpu_only {
        Some(true) => "三个必需命令可启动，帮助输出明确报告无 CUDA".into(),
        Some(false) => "检测到 CUDA 支持（帮助输出或运行目录中发现 CUDA 运行时）".into(),
        None => "命令可启动，但帮助输出未明确证明是否支持 CUDA，将由系统 CUDA 检测决定".into(),
    };
    EngineStatus {
        kind: EngineKind::Colmap,
        path: path.to_path_buf(),
        exists: true,
        can_start: successful,
        version: first_line,
        cpu_only,
        detail,
    }
}

fn runtime_contains_cuda(directory: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if path.is_dir() {
            return runtime_contains_cuda(&path);
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        ["cudart", "cublas", "cudnn", "cuda.dll"]
            .iter()
            .any(|needle| name.contains(needle))
    })
}

/// Returns `true` when the bundled COLMAP has CUDA support available,
/// `false` when it should run in CPU-only mode.
/// Errors only if COLMAP is missing or cannot start at all.
///
/// Detection strategy (in order):
/// 1. COLMAP help output contains positive CUDA markers (e.g. "cuda: yes")
///    or CUDA DLLs are found next to the executable → GPU.
/// 2. COLMAP help output contains negative CUDA markers (e.g. "cuda: no") → CPU.
/// 3. Neither: fall back to probing the system for a CUDA runtime via
///    `nvidia-smi` (Windows/Linux) or `nvcc`. If found → GPU, else → CPU.
pub async fn detect_colmap_gpu(paths: &EnginePaths) -> Result<bool> {
    let status = check_colmap(&paths.colmap).await;
    if !status.can_start {
        return Err(crate::error::SplatError::UnsupportedEngine(status.detail));
    }
    match status.cpu_only {
        Some(false) => Ok(true),
        Some(true) => Ok(false),
        // Ambiguous: COLMAP help gave no clear answer. Ask the OS.
        None => Ok(system_has_cuda().await),
    }
}

/// Returns `true` when a CUDA-capable GPU driver is detected on the system.
/// Tries `nvidia-smi` first (fastest, works on both Windows and Linux),
/// then falls back to `nvcc --version` for CUDA toolkit installs.
async fn system_has_cuda() -> bool {
    let manager = ProcessManager::new();

    // Try nvidia-smi – present whenever the NVIDIA driver is installed.
    #[cfg(target_os = "windows")]
    let nvidia_smi = "nvidia-smi.exe";
    #[cfg(not(target_os = "windows"))]
    let nvidia_smi = "nvidia-smi";

    // Resolve through PATH
    if let Some(nvidia_smi_path) = which_in_path(nvidia_smi) {
        if let Ok(out) = manager
            .run(ProcessSpec {
                executable: nvidia_smi_path,
                args: vec![],
                working_directory: None,
                log_path: None,
                observer: None,
            })
            .await
        {
            if out.success {
                return true;
            }
        }
    }

    // Fall back to nvcc
    #[cfg(target_os = "windows")]
    let nvcc = "nvcc.exe";
    #[cfg(not(target_os = "windows"))]
    let nvcc = "nvcc";

    if let Some(nvcc_path) = which_in_path(nvcc) {
        if let Ok(out) = manager
            .run(ProcessSpec {
                executable: nvcc_path,
                args: vec!["--version".into()],
                working_directory: None,
                log_path: None,
                observer: None,
            })
            .await
        {
            if out.success {
                return true;
            }
        }
    }

    false
}

/// Look up `name` in the system `PATH`, returning the full path if found.
fn which_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path_var| {
        std::env::split_paths(&path_var).find_map(|dir| {
            let candidate = dir.join(name);
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}
