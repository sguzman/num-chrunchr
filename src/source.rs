use anyhow::{Context, Result};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

/// Represents either a disk path or inline decimal digits supplied via CLI.
#[derive(Clone, Debug)]
pub enum NumberSource {
    File(PathBuf),
    Inline(String),
}

/// Prepared temporary handle to keep inline sources alive.
pub struct PreparedSource {
    pub path: PathBuf,
    _temp: Option<NamedTempFile>,
}

impl NumberSource {
    /// Human-friendly label used for reporting/resume detection.
    pub fn label(&self) -> String {
        match self {
            NumberSource::File(path) => format!("file:{}", path.display()),
            NumberSource::Inline(text) => {
                let preview: String = text.chars().take(24).collect();
                format!("inline:{}...", preview)
            }
        }
    }

    /// Copy this source into `dest`. This is used to place the working cofactor in the report directory.
    pub fn copy_to(&self, dest: &Path) -> Result<()> {
        if dest.exists() {
            fs::remove_file(dest)
                .with_context(|| format!("remove stale working file {}", dest.display()))?;
        }
        match self {
            NumberSource::File(path) => {
                fs::copy(path, dest)
                    .with_context(|| format!("copy {} -> {}", path.display(), dest.display()))?;
            }
            NumberSource::Inline(text) => {
                fs::write(dest, text)
                    .with_context(|| format!("write inline digits to {}", dest.display()))?;
            }
        }
        Ok(())
    }

    /// Prepare this source as a file path for streaming commands. Inline data is written to a temp file.
    pub fn prepare(&self) -> Result<PreparedSource> {
        match self {
            NumberSource::File(path) => Ok(PreparedSource {
                path: path.clone(),
                _temp: None,
            }),
            NumberSource::Inline(text) => {
                let mut tmp = NamedTempFile::new().context("create inline temp file")?;
                write!(tmp, "{text}").context("write inline digits to temp file")?;
                let path = tmp.path().to_path_buf();
                Ok(PreparedSource {
                    path,
                    _temp: Some(tmp),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::read_to_string;
    use tempfile::tempdir;

    #[test]
    fn inline_source_writes_contents() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("inline.txt");
        let source = NumberSource::Inline("12345".into());
        source.copy_to(&path).unwrap();
        assert_eq!(read_to_string(path).unwrap(), "12345");
    }

    #[test]
    fn file_source_copies_to_destination() {
        let tmp = tempdir().unwrap();
        let origin = tmp.path().join("origin.txt");
        fs::write(&origin, "98765").unwrap();
        let dest = tmp.path().join("dest.txt");
        let source = NumberSource::File(origin.clone());
        source.copy_to(&dest).unwrap();
        assert_eq!(read_to_string(dest).unwrap(), "98765");
    }

    #[test]
    fn prepare_inline_returns_temp_path() {
        let src = NumberSource::Inline("31415".into());
        let prepared = src.prepare().unwrap();
        assert_eq!(
            prepared.path.file_name().unwrap(),
            prepared._temp.as_ref().unwrap().path().file_name().unwrap()
        );
        assert_eq!(read_to_string(prepared.path).unwrap(), "31415");
    }
}
