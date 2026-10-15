use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::extract::extract_path;
use crate::file_ops::{file_edit, sha256_hex};
use crate::lang;
use crate::types::{FileLineEdit, SymbolEditResult};

pub fn symbol_edit(
    file: &Path,
    symbol_name: &str,
    new_source: &str,
    dry_run: bool,
    expected_hash: Option<&str>,
    record_undo: bool,
) -> Result<SymbolEditResult, String> {
    let start = Instant::now();

    if !file.is_file() {
        return Err(format!("not a file: {}", file.display()));
    }

    let original = fs::read_to_string(file)
        .map_err(|e| format!("failed to read {}: {e}", file.display()))?;
    let before_hash = sha256_hex(original.as_bytes());

    if let Some(expected) = expected_hash {
        if expected != before_hash {
            return Err("expected_hash mismatch".to_string());
        }
    }

    let specs = lang::registry();
    let results = extract_path(file, specs)?;

    let extraction = results.into_iter().next()
        .ok_or_else(|| "no extraction result".to_string())?;

    let symbol = extraction.symbols.iter()
        .find(|s| s.name == symbol_name)
        .ok_or_else(|| format!("symbol not found: {symbol_name}"))?;

    let edit = FileLineEdit {
        start_line: symbol.line_start + 1,
        end_line: Some(symbol.line_end + 1),
        text: Some(new_source.to_string()),
    };

    let edit_result = file_edit(
        file,
        "replace",
        Some(vec![edit]),
        None,
        dry_run,
        Some(&before_hash),
        record_undo,
    )?;

    Ok(SymbolEditResult {
        file_path: file.to_string_lossy().to_string(),
        symbol_name: symbol_name.to_string(),
        applied: edit_result.applied,
        dry_run,
        before_hash: edit_result.before_hash,
        after_hash: edit_result.after_hash,
        line_start: symbol.line_start,
        line_end: symbol.line_end,
        duration_ms: start.elapsed().as_millis(),
        edit_id: edit_result.edit_id,
        updated_text: edit_result.updated_text,
    })
}
