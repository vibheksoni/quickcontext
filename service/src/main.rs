use std::path::Path;
use std::process;

use quickcontext_service::extract::extract_path;
use quickcontext_service::grep::grep_path;
use quickcontext_service::lang;
use quickcontext_service::pattern_match;
use quickcontext_service::protocol_search::{self, ProtocolSearchOptions};
use quickcontext_service::skeleton::{self, SkeletonOptions};
use quickcontext_service::text_search;


#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "extract" => cmd_extract(&args),
        "extract-symbol" => cmd_extract_symbol(&args),
        "grep" => cmd_grep(&args),
        "skeleton" => cmd_skeleton(&args),
        "text-search" => cmd_text_search(&args),
        "protocol-search" => cmd_protocol_search(&args),
        "pattern-search" => cmd_pattern_search(&args),
        "serve" => cmd_serve().await,
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}


fn print_usage() {
    eprintln!("usage: quickcontext-service <command>");
    eprintln!();
    eprintln!("commands:");
    eprintln!("  extract <file_or_dir> [--pretty] [--compact] [--stats] [--no-gitignore]  Extract symbols from source files");
    eprintln!("  extract-symbol <file> <symbol> [--pretty]          Extract specific symbol by name");
    eprintln!("  grep <query> [--path <file_or_dir>] [--no-gitignore] [--limit N] [--pretty]  Fast literal grep");
    eprintln!("  skeleton <dir> [--depth N] [--no-signatures] [--lines] [--collapse N] [--no-gitignore] [--markdown] [--pretty]  Repo skeleton");
    eprintln!("  text-search <query> [--path <dir>] [--no-gitignore] [--limit N] [--intent] [--intent-level N] [--pretty]  BM25 full-text search");
    eprintln!("  protocol-search <query> [--path <dir>] [--no-gitignore] [--limit N] [--context-radius N] [--min-score F] [--include-marker M]... [--exclude-marker M]... [--max-input-fields N] [--max-output-fields N] [--pretty]  Extract protocol request/response contracts");
    eprintln!("  pattern-search <pattern> --lang <language> [--path <dir>] [--no-gitignore] [--limit N] [--pretty]  AST pattern matching");
    eprintln!("  serve                                             Start IPC daemon");
}


fn cmd_extract(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service extract <file_or_dir> [--pretty] [--compact] [--stats] [--no-gitignore]");
        process::exit(1);
    }

    let target = &args[2];
    let pretty = args.iter().any(|a| a == "--pretty");
    let compact = args.iter().any(|a| a == "--compact");
    let stats_only = args.iter().any(|a| a == "--stats");
    let no_gitignore = args.iter().any(|a| a == "--no-gitignore");
    let specs = lang::registry();
    let path = Path::new(target);

    let options = quickcontext_service::extract::ExtractOptions {
        respect_gitignore: !no_gitignore,
    };

    if (compact || stats_only) && path.is_dir() {
        if stats_only {
            let mut sink = std::io::sink();
            let stats = quickcontext_service::extract::extract_compact_streaming(
                path, specs, options, &mut sink,
            );
            let json = if pretty {
                serde_json::to_string_pretty(&stats)
            } else {
                serde_json::to_string(&stats)
            };
            match json {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("error: stats serialization failed: {e}");
                    process::exit(1);
                }
            }
        } else {
            let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
            let stats = quickcontext_service::extract::extract_compact_streaming(
                path, specs, options, &mut stdout,
            );
            drop(stdout);
            let json = if pretty {
                serde_json::to_string_pretty(&stats)
            } else {
                serde_json::to_string(&stats)
            };
            match json {
                Ok(s) => eprintln!("{s}"),
                Err(e) => eprintln!("error: stats serialization failed: {e}"),
            }
        }
        return;
    }

    let results = match extract_path(path, specs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&results)
    } else {
        serde_json::to_string(&results)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


fn cmd_extract_symbol(args: &[String]) {
    if args.len() < 4 {
        eprintln!("usage: quickcontext-service extract-symbol <file> <symbol> [--pretty]");
        eprintln!("  symbol can be 'name' or 'Parent.name' for disambiguation");
        process::exit(1);
    }

    let file = &args[2];
    let symbol_query = &args[3];
    let pretty = args.iter().any(|a| a == "--pretty");
    let specs = lang::registry();
    let path = Path::new(file);

    if !path.is_file() {
        eprintln!("error: not a file: {file}");
        process::exit(1);
    }

    let results = match extract_path(path, specs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let extraction = match results.into_iter().next() {
        Some(r) => r,
        None => {
            eprintln!("error: no extraction result for: {file}");
            process::exit(1);
        }
    };

    let (parent_filter, name_filter) = if let Some(dot_pos) = symbol_query.rfind('.') {
        (Some(&symbol_query[..dot_pos]), &symbol_query[dot_pos + 1..])
    } else {
        (None, symbol_query.as_str())
    };

    let name_lower = name_filter.to_lowercase();
    let parent_lower = parent_filter.map(|p| p.to_lowercase());

    let matched: Vec<_> = extraction
        .symbols
        .into_iter()
        .filter(|s| {
            if s.name.to_lowercase() != name_lower {
                return false;
            }
            if let Some(ref pf) = parent_lower {
                match &s.parent {
                    Some(p) => p.to_lowercase() == *pf,
                    None => false,
                }
            } else {
                true
            }
        })
        .collect();

    if matched.is_empty() {
        eprintln!("error: symbol not found: {symbol_query} in {file}");
        process::exit(1);
    }

    let output = serde_json::json!({
        "file_path": file,
        "language": extraction.language,
        "query": symbol_query,
        "symbols": matched,
        "total_matches": matched.len(),
    });

    let json = if pretty {
        serde_json::to_string_pretty(&output)
    } else {
        serde_json::to_string(&output)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


fn cmd_grep(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service grep <query> [--path <file_or_dir>] [--no-gitignore] [--limit N] [--pretty]");
        process::exit(1);
    }

    let query = args[2].clone();
    let mut path = ".".to_string();
    let mut respect_gitignore = true;
    let mut limit: usize = 200;
    let mut pretty = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --path requires a value");
                    process::exit(1);
                }
                path = args[i].clone();
            }
            "--no-gitignore" => {
                respect_gitignore = false;
            }
            "--limit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --limit requires a value");
                    process::exit(1);
                }
                limit = match args[i].parse::<usize>() {
                    Ok(v) => v.max(1),
                    Err(_) => {
                        eprintln!("error: invalid limit value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--pretty" => {
                pretty = true;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    let target = Path::new(&path);
    let result = match grep_path(&query, target, respect_gitignore, limit, 0, 0) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


fn cmd_skeleton(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service skeleton <dir> [--depth N] [--no-signatures] [--lines] [--collapse N] [--no-gitignore] [--markdown] [--pretty]");
        process::exit(1);
    }

    let target = &args[2];
    let mut options = SkeletonOptions::default();
    let mut markdown = false;
    let mut pretty = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--depth" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --depth requires a value");
                    process::exit(1);
                }
                options.max_depth = match args[i].parse::<usize>() {
                    Ok(v) => v.max(1),
                    Err(_) => {
                        eprintln!("error: invalid depth value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--no-signatures" => {
                options.include_signatures = false;
            }
            "--lines" => {
                options.include_line_numbers = true;
            }
            "--collapse" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --collapse requires a value");
                    process::exit(1);
                }
                options.collapse_threshold = match args[i].parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("error: invalid collapse value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--no-gitignore" => {
                options.respect_gitignore = false;
            }
            "--markdown" => {
                markdown = true;
            }
            "--pretty" => {
                pretty = true;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    let specs = lang::registry();
    let path = Path::new(target);

    let result = match skeleton::build_skeleton(path, specs, &options) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    if markdown {
        let md = skeleton::render_markdown(&result, &options);
        println!("{md}");
        eprintln!(
            "[skeleton] {} files, {} symbols, {} dirs in {}ms",
            result.total_files, result.total_symbols, result.total_directories, result.duration_ms
        );
    } else {
        let json = if pretty {
            serde_json::to_string_pretty(&result)
        } else {
            serde_json::to_string(&result)
        };

        match json {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("error: json serialization failed: {e}");
                process::exit(1);
            }
        }
    }
}


fn cmd_text_search(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service text-search <query> [--path <dir>] [--no-gitignore] [--limit N] [--intent] [--intent-level N] [--pretty]");
        process::exit(1);
    }

    let query = args[2].clone();
    let mut path = ".".to_string();
    let mut respect_gitignore = true;
    let mut limit: usize = 20;
    let mut pretty = false;
    let mut intent_mode = false;
    let mut intent_level: u8 = 2;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --path requires a value");
                    process::exit(1);
                }
                path = args[i].clone();
            }
            "--no-gitignore" => {
                respect_gitignore = false;
            }
            "--limit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --limit requires a value");
                    process::exit(1);
                }
                limit = match args[i].parse::<usize>() {
                    Ok(v) => v.max(1),
                    Err(_) => {
                        eprintln!("error: invalid limit value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--intent" => {
                intent_mode = true;
            }
            "--intent-level" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --intent-level requires a value");
                    process::exit(1);
                }
                intent_level = match args[i].parse::<u8>() {
                    Ok(v) => v.clamp(1, 3),
                    Err(_) => {
                        eprintln!("error: invalid intent level value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--pretty" => {
                pretty = true;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    let target = Path::new(&path);
    let specs = lang::registry();
    let result = match text_search::text_search(
        &query,
        target,
        respect_gitignore,
        limit,
        specs,
        intent_mode,
        intent_level,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


fn cmd_protocol_search(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service protocol-search <query> [--path <dir>] [--no-gitignore] [--limit N] [--context-radius N] [--min-score F] [--include-marker M]... [--exclude-marker M]... [--max-input-fields N] [--max-output-fields N] [--pretty]");
        process::exit(1);
    }

    let query = args[2].clone();
    let mut path = ".".to_string();
    let mut respect_gitignore = true;
    let mut limit: usize = 20;
    let mut pretty = false;
    let mut options = ProtocolSearchOptions::default();

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --path requires a value");
                    process::exit(1);
                }
                path = args[i].clone();
            }
            "--no-gitignore" => {
                respect_gitignore = false;
            }
            "--limit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --limit requires a value");
                    process::exit(1);
                }
                limit = match args[i].parse::<usize>() {
                    Ok(v) => v.max(1),
                    Err(_) => {
                        eprintln!("error: invalid limit value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--context-radius" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --context-radius requires a value");
                    process::exit(1);
                }
                options.context_radius = match args[i].parse::<usize>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        eprintln!("error: invalid context radius value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--min-score" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --min-score requires a value");
                    process::exit(1);
                }
                options.min_score = match args[i].parse::<f64>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        eprintln!("error: invalid min-score value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--include-marker" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --include-marker requires a value");
                    process::exit(1);
                }
                if options.include_markers.is_none() {
                    options.include_markers = Some(Vec::new());
                }
                if let Some(markers) = &mut options.include_markers {
                    markers.push(args[i].clone());
                }
            }
            "--exclude-marker" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --exclude-marker requires a value");
                    process::exit(1);
                }
                if options.exclude_markers.is_none() {
                    options.exclude_markers = Some(Vec::new());
                }
                if let Some(markers) = &mut options.exclude_markers {
                    markers.push(args[i].clone());
                }
            }
            "--max-input-fields" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --max-input-fields requires a value");
                    process::exit(1);
                }
                options.max_input_fields = match args[i].parse::<usize>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        eprintln!("error: invalid max-input-fields value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--max-output-fields" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --max-output-fields requires a value");
                    process::exit(1);
                }
                options.max_output_fields = match args[i].parse::<usize>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        eprintln!("error: invalid max-output-fields value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--pretty" => {
                pretty = true;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    let target = Path::new(&path);
    let specs = lang::registry();
    let result = match protocol_search::protocol_search(
        &query,
        target,
        respect_gitignore,
        limit,
        specs,
        Some(&options),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


fn cmd_pattern_search(args: &[String]) {
    if args.len() < 3 {
        eprintln!("usage: quickcontext-service pattern-search <pattern> --lang <language> [--path <dir>] [--no-gitignore] [--limit N] [--pretty]");
        process::exit(1);
    }

    let pattern = args[2].clone();
    let mut language = String::new();
    let mut path = ".".to_string();
    let mut respect_gitignore = true;
    let mut limit: usize = 50;
    let mut pretty = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --lang requires a value");
                    process::exit(1);
                }
                language = args[i].clone();
            }
            "--path" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --path requires a value");
                    process::exit(1);
                }
                path = args[i].clone();
            }
            "--no-gitignore" => {
                respect_gitignore = false;
            }
            "--limit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --limit requires a value");
                    process::exit(1);
                }
                limit = match args[i].parse::<usize>() {
                    Ok(v) => v.max(1),
                    Err(_) => {
                        eprintln!("error: invalid limit value: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--pretty" => {
                pretty = true;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    if language.is_empty() {
        eprintln!("error: --lang is required");
        process::exit(1);
    }

    let target = Path::new(&path);
    let specs = lang::registry();

    let result = match pattern_match::pattern_search(&pattern, &language, target, respect_gitignore, limit, specs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    };

    match json {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: json serialization failed: {e}");
            process::exit(1);
        }
    }
}


async fn cmd_serve() {
    eprintln!("[quickcontext] starting daemon...");
    if let Err(e) = quickcontext_service::server::run().await {
        eprintln!("[quickcontext] fatal: {e}");
        process::exit(1);
    }
}
