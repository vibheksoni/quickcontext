use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::to_string_pretty;
use sha2::{Digest, Sha256};

use crate::types::{
    FileEditResult,
    FileEditRevertResult,
    FileEditUndoRecord,
    FileLineEdit,
    FileReadLine,
    FileReadResult,
};


static EDIT_COUNTER: AtomicU64 = AtomicU64::new(1);


pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}


fn quick_context_dir() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("failed to read current dir: {e}"))?;
    Ok(cwd.join(".quick-context"))
}


fn undo_dir() -> Result<PathBuf, String> {
    Ok(quick_context_dir()?.join("undo"))
}


fn undo_record_path(edit_id: &str) -> Result<PathBuf, String> {
    Ok(undo_dir()?.join(format!("{edit_id}.json")))
}


fn line_starts(content: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, b) in content.as_bytes().iter().enumerate() {
        if *b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}


fn line_to_byte_start(content: &str, starts: &[usize], line: usize) -> usize {
    if line <= 1 {
        return 0;
    }
    let total_lines = starts.len();
    if line > total_lines {
        return content.len();
    }
    starts[line - 1]
}


fn line_to_byte_end_inclusive(content: &str, starts: &[usize], line: usize) -> usize {
    let total_lines = starts.len();
    if line >= total_lines {
        return content.len();
    }
    starts[line]
}


#[derive(Debug, Clone)]
struct PlannedEdit {
    start_byte: usize,
    end_byte: usize,
    replacement: String,
}


fn plan_line_edits(content: &str, edits: &[FileLineEdit], default_text: Option<&str>) -> Result<Vec<PlannedEdit>, String> {
    if edits.is_empty() {
        return Err("at least one edit is required".to_string());
    }

    let starts = line_starts(content);
    let mut planned = Vec::with_capacity(edits.len());

    for edit in edits {
        if edit.start_line == 0 {
            return Err("start_line must be >= 1".to_string());
        }

        let (start, end) = match edit.end_line {
            None => {
                let pos = line_to_byte_end_inclusive(content, &starts, edit.start_line);
                (pos, pos)
            }
            Some(end_line) => {
                if end_line < edit.start_line {
                    return Err("end_line must be >= start_line".to_string());
                }
                let start = line_to_byte_start(content, &starts, edit.start_line);
                let end = line_to_byte_end_inclusive(content, &starts, end_line);
                (start, end)
            }
        };

        let replacement = match (&edit.text, default_text) {
            (Some(text), _) => text.clone(),
            (None, Some(text)) => text.to_string(),
            (None, None) => String::new(),
        };

        planned.push(PlannedEdit {
            start_byte: start,
            end_byte: end,
            replacement,
        });
    }

    planned.sort_by_key(|edit| std::cmp::Reverse(edit.start_byte));

    let mut last_start = content.len() + 1;
    for edit in &planned {
        if edit.end_byte > last_start {
            return Err("overlapping edits are not allowed".to_string());
        }
        last_start = edit.start_byte;
    }

    Ok(planned)
}


fn apply_planned_edits(content: &str, edits: &[PlannedEdit]) -> String {
    let mut output = content.to_string();
    for edit in edits {
        if edit.start_byte <= output.len() && edit.end_byte <= output.len() {
            output.replace_range(edit.start_byte..edit.end_byte, &edit.replacement);
        }
    }
    output
}


pub fn make_edit_id(file_path: &Path) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = EDIT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = Sha256::new();
    hasher.update(file_path.to_string_lossy().as_bytes());
    hasher.update(now.to_le_bytes());
    hasher.update(seq.to_le_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("e_{}_{}", now, &digest[..12])
}


fn save_undo_record(record: &FileEditUndoRecord) -> Result<(), String> {
    let dir = undo_dir()?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create undo dir {}: {e}", dir.display()))?;

    let record_path = undo_record_path(&record.edit_id)?;
    let json = to_string_pretty(record).map_err(|e| format!("failed to serialize undo record: {e}"))?;
    fs::write(&record_path, json).map_err(|e| format!("failed to write undo record {}: {e}", record_path.display()))?;
    Ok(())
}


fn load_undo_record(edit_id: &str) -> Result<FileEditUndoRecord, String> {
    let path = undo_record_path(edit_id)?;
    let content = fs::read_to_string(&path).map_err(|e| format!("failed to read undo record {}: {e}", path.display()))?;
    serde_json::from_str(&content).map_err(|e| format!("failed to parse undo record {}: {e}", path.display()))
}


pub fn file_read(
    file: &Path,
    start_line: Option<usize>,
    end_line: Option<usize>,
    max_bytes: Option<usize>,
) -> Result<FileReadResult, String> {
    let start = Instant::now();
    if !file.is_file() {
        return Err(format!("not a file: {}", file.display()));
    }

    let mut content = fs::read_to_string(file)
        .map_err(|e| format!("failed to read {}: {e}", file.display()))?;

    let mut truncated = false;
    if let Some(limit) = max_bytes {
        if content.len() > limit {
            let mut end = limit;
            while !content.is_char_boundary(end) {
                end -= 1;
            }
            content.truncate(end);
            truncated = true;
        }
    }

    let all_lines: Vec<&str> = content.lines().collect();
    let total_lines = all_lines.len();

    let line_start = start_line.unwrap_or(1).max(1);
    let mut line_end = end_line.unwrap_or(total_lines.max(1));
    if line_end < line_start {
        line_end = line_start;
    }

    let slice_start = line_start.saturating_sub(1).min(total_lines);
    let slice_end = line_end.min(total_lines);

    let mut lines = Vec::new();
    for (idx, text) in all_lines[slice_start..slice_end].iter().enumerate() {
        lines.push(FileReadLine {
            line_number: slice_start + idx + 1,
            text: (*text).to_string(),
        });
    }

    Ok(FileReadResult {
        file_path: file.to_string_lossy().to_string(),
        line_start,
        line_end: slice_end.max(line_start),
        total_lines,
        truncated,
        duration_ms: start.elapsed().as_millis(),
        lines,
    })
}


pub fn file_edit(
    file: &Path,
    mode: &str,
    edits: Option<Vec<FileLineEdit>>,
    text: Option<String>,
    dry_run: bool,
    expected_hash: Option<&str>,
    record_undo: bool,
) -> Result<FileEditResult, String> {
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

    let updated = match mode {
        "append" => {
            let append_text = text.ok_or_else(|| "text is required for append mode".to_string())?;
            format!("{}{}", original, append_text)
        }
        "insert" => {
            let edit_items = edits.ok_or_else(|| "edits are required for insert mode".to_string())?;
            let starts = line_starts(&original);
            let mut planned = Vec::new();
            for edit in &edit_items {
                if edit.start_line == 0 {
                    return Err("start_line must be >= 1".to_string());
                }
                let pos = line_to_byte_end_inclusive(&original, &starts, edit.start_line);
                let replacement = edit
                    .text
                    .clone()
                    .or_else(|| text.clone())
                    .ok_or_else(|| "text is required for insert mode".to_string())?;
                planned.push(PlannedEdit {
                    start_byte: pos,
                    end_byte: pos,
                    replacement,
                });
            }
            planned.sort_by_key(|edit| std::cmp::Reverse(edit.start_byte));
            apply_planned_edits(&original, &planned)
        }
        "replace" => {
            let edit_items = edits.ok_or_else(|| "edits are required for replace mode".to_string())?;
            let planned = plan_line_edits(&original, &edit_items, text.as_deref())?;
            apply_planned_edits(&original, &planned)
        }
        "delete" => {
            let edit_items = edits.ok_or_else(|| "edits are required for delete mode".to_string())?;
            let mut zeroed = edit_items;
            for item in &mut zeroed {
                item.text = Some(String::new());
            }
            let planned = plan_line_edits(&original, &zeroed, Some(""))?;
            apply_planned_edits(&original, &planned)
        }
        "batch" => {
            let edit_items = edits.ok_or_else(|| "edits are required for batch mode".to_string())?;
            let planned = plan_line_edits(&original, &edit_items, text.as_deref())?;
            apply_planned_edits(&original, &planned)
        }
        _ => return Err(format!("unsupported edit mode: {mode}")),
    };

    let after_hash = sha256_hex(updated.as_bytes());

    let mut edit_id = None;
    if record_undo {
        let id = make_edit_id(file);
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let record = FileEditUndoRecord {
            edit_id: id.clone(),
            file_path: file.to_string_lossy().to_string(),
            before_hash: before_hash.clone(),
            after_hash: after_hash.clone(),
            original_text: original.clone(),
            updated_text: updated.clone(),
            created_at_ms,
            reverted: false,
        };
        save_undo_record(&record)?;
        edit_id = Some(id);
    }

    if !dry_run {
        fs::write(file, &updated)
            .map_err(|e| format!("failed to write {}: {e}", file.display()))?;
    }

    Ok(FileEditResult {
        file_path: file.to_string_lossy().to_string(),
        applied: true,
        dry_run,
        before_hash,
        after_hash,
        duration_ms: start.elapsed().as_millis(),
        edit_id,
        updated_text: if dry_run { Some(updated) } else { None },
    })
}


pub fn file_edit_revert(edit_id: &str, dry_run: bool, expected_hash: Option<&str>) -> Result<FileEditRevertResult, String> {
    let start = Instant::now();
    let mut record = load_undo_record(edit_id)?;

    if record.reverted {
        return Err(format!("edit id already reverted: {edit_id}"));
    }

    let file = Path::new(&record.file_path);
    if !file.is_file() {
        return Err(format!("not a file: {}", file.display()));
    }

    let current = fs::read_to_string(file)
        .map_err(|e| format!("failed to read {}: {e}", file.display()))?;
    let before_hash = sha256_hex(current.as_bytes());

    if let Some(expected) = expected_hash {
        if before_hash != expected {
            return Err("expected_hash mismatch before revert".to_string());
        }
    } else if before_hash != record.after_hash {
        return Err("file has changed since edit; refusing revert without expected_hash".to_string());
    }

    if !dry_run {
        fs::write(file, &record.original_text)
            .map_err(|e| format!("failed to write {}: {e}", file.display()))?;
        record.reverted = true;
        save_undo_record(&record)?;
    }

    let after_hash = sha256_hex(record.original_text.as_bytes());

    Ok(FileEditRevertResult {
        file_path: record.file_path,
        reverted: true,
        before_hash,
        after_hash,
        duration_ms: start.elapsed().as_millis(),
        edit_id: edit_id.to_string(),
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_read_truncates_on_char_boundary() {
        let path = std::env::temp_dir().join(format!("qc_file_read_{}.txt", std::process::id()));
        fs::write(&path, "h\u{e9}llo\nw\u{f6}rld\n").unwrap();

        let result = file_read(&path, None, None, Some(2));
        let _ = fs::remove_file(&path);

        let result = result.unwrap();
        assert!(result.truncated);
        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.lines[0].text, "h");
    }
}
