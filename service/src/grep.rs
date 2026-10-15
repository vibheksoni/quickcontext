use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use ignore::WalkBuilder;

use crate::types::{GrepMatch, GrepResult};

const QUICK_IGNORE_FILENAME: &str = ".quick-ignore";

pub fn grep_path(
    query: &str,
    root: &Path,
    respect_gitignore: bool,
    limit: usize,
    before_context: usize,
    after_context: usize,
) -> Result<GrepResult, String> {
    if query.is_empty() {
        return Err("query cannot be empty".to_string());
    }
    if !root.exists() {
        return Err(format!("path does not exist: {}", root.display()));
    }

    let start = Instant::now();
    let effective_limit = limit.max(1);
    let match_count = AtomicUsize::new(0);
    let context = ContextWindow {
        before: before_context,
        after: after_context,
    };

    if root.is_file() {
        let mut matches = Vec::new();
        search_file(root, query, context, &match_count, effective_limit, &mut |m| {
            matches.push(m)
        });
        return Ok(GrepResult {
            matches,
            searched_files: 1,
            duration_ms: start.elapsed().as_millis(),
            truncated: match_count.load(Ordering::Relaxed) > effective_limit,
        });
    }

    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true)
        .add_custom_ignore_filename(QUICK_IGNORE_FILENAME)
        .threads(std::thread::available_parallelism().map_or(4, usize::from));

    if !respect_gitignore {
        builder.git_ignore(false).git_global(false).git_exclude(false);
    }

    let (tx, rx) = mpsc::channel::<GrepMatch>();
    let searched_files = AtomicUsize::new(0);

    builder.build_parallel().run(|| {
        let tx = tx.clone();
        let match_count = &match_count;
        let searched_files = &searched_files;

        Box::new(move |entry| {
            if match_count.load(Ordering::Relaxed) >= effective_limit {
                return ignore::WalkState::Quit;
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => return ignore::WalkState::Continue,
            };

            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                return ignore::WalkState::Continue;
            }

            searched_files.fetch_add(1, Ordering::Relaxed);

            let limit_reached = search_file(
                entry.path(),
                query,
                context,
                match_count,
                effective_limit,
                &mut |m| {
                    let _ = tx.send(m);
                },
            );

            if limit_reached {
                ignore::WalkState::Quit
            } else {
                ignore::WalkState::Continue
            }
        })
    });

    drop(tx);

    let mut matches: Vec<GrepMatch> = rx.into_iter().collect();

    matches.sort_by(|a, b| {
        a.file_path
            .cmp(&b.file_path)
            .then(a.line_number.cmp(&b.line_number))
            .then(a.column_start.cmp(&b.column_start))
    });
    matches.truncate(effective_limit);

    Ok(GrepResult {
        matches,
        searched_files: searched_files.load(Ordering::Relaxed),
        duration_ms: start.elapsed().as_millis(),
        truncated: match_count.load(Ordering::Relaxed) > effective_limit,
    })
}

#[derive(Clone, Copy)]
struct ContextWindow {
    before: usize,
    after: usize,
}

/// Scan one file for literal matches, claiming result slots from the shared counter.
///
/// Returns true when scanning stopped because the global match limit was reached.
fn search_file(
    path: &Path,
    query: &str,
    context: ContextWindow,
    match_count: &AtomicUsize,
    limit: usize,
    emit: &mut dyn FnMut(GrepMatch),
) -> bool {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let mut reader = BufReader::new(file);
    let mut buf: Vec<u8> = Vec::new();
    let mut line_number = 0usize;
    let mut before_lines: VecDeque<String> = VecDeque::with_capacity(context.before);
    let mut pending: Option<(GrepMatch, usize)> = None;
    let mut file_path: Option<String> = None;
    let query_chars = query.chars().count();

    loop {
        if pending.is_none() && match_count.load(Ordering::Relaxed) >= limit {
            return true;
        }

        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        line_number += 1;

        if buf.contains(&0) {
            break;
        }

        let line = String::from_utf8_lossy(&buf);
        let text = line.trim_end_matches(['\r', '\n']);

        if let Some((m, remaining)) = pending.as_mut() {
            m.context_after.push(text.to_string());
            *remaining -= 1;
            if *remaining == 0 {
                if let Some((m, _)) = pending.take() {
                    emit(m);
                }
            }
        }

        if let Some(byte_idx) = line.find(query) {
            if match_count.fetch_add(1, Ordering::Relaxed) < limit {
                if let Some((m, _)) = pending.take() {
                    emit(m);
                }

                let column_start = line[..byte_idx].chars().count() + 1;
                let m = GrepMatch {
                    file_path: file_path
                        .get_or_insert_with(|| path.to_string_lossy().into_owned())
                        .clone(),
                    line_number,
                    column_start,
                    column_end: column_start + query_chars - 1,
                    line: text.to_string(),
                    context_before: before_lines.iter().cloned().collect(),
                    context_after: Vec::new(),
                };

                if context.after == 0 {
                    emit(m);
                } else {
                    pending = Some((m, context.after));
                }
            }
        }

        if context.before > 0 {
            if before_lines.len() == context.before {
                before_lines.pop_front();
            }
            before_lines.push_back(text.to_string());
        }
    }

    if let Some((m, _)) = pending.take() {
        emit(m);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("qc_grep_{}_{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn single_file_respects_limit_with_after_context() {
        let path = temp_file("limit.txt", b"foo 1\nfoo 2\nfoo 3\nbar\n");
        let result = grep_path("foo", &path, true, 1, 0, 2);
        let _ = std::fs::remove_file(&path);

        let result = result.unwrap();
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].line_number, 1);
        assert_eq!(result.matches[0].context_after, vec!["foo 2", "foo 3"]);
        assert!(result.truncated);
    }

    #[test]
    fn single_file_collects_context_lines() {
        let path = temp_file("context.txt", b"a\nb\nneedle here\nc\nd\n");
        let result = grep_path("needle", &path, true, 10, 1, 1);
        let _ = std::fs::remove_file(&path);

        let result = result.unwrap();
        assert_eq!(result.matches.len(), 1);
        let m = &result.matches[0];
        assert_eq!(m.line_number, 3);
        assert_eq!(m.column_start, 1);
        assert_eq!(m.column_end, 6);
        assert_eq!(m.line, "needle here");
        assert_eq!(m.context_before, vec!["b"]);
        assert_eq!(m.context_after, vec!["c"]);
        assert!(!result.truncated);
    }

    #[test]
    fn single_file_keeps_scanning_past_invalid_utf8() {
        let path = temp_file("latin1.txt", b"caf\xe9\nneedle\n");
        let result = grep_path("needle", &path, true, 10, 0, 0);
        let _ = std::fs::remove_file(&path);

        let result = result.unwrap();
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].line_number, 2);
    }
}
