use std::fs;
use std::path::PathBuf;
use std::collections::HashSet;
use serde_json::json;
use walkdir::WalkDir;
use scraper::{Html, Selector, ElementRef};
use regex::Regex;
use lazy_static::lazy_static;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: html_to_json <input_path> <output.json> [options]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let keep_numbers = args.iter().any(|a| a == "--keep-numbers");
    let no_filter = args.iter().any(|a| a == "--no-filter");
    let verbose = args.iter().any(|a| a == "--verbose");
    let resume = args.iter().any(|a| a == "--resume");

    let save_every = args.iter()
        .position(|a| a == "--save-every")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let mut exclude_patterns = Vec::new();
    let mut include_patterns = Vec::new();
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--exclude" && i + 1 < args.len() {
            exclude_patterns.push(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--include" && i + 1 < args.len() {
            include_patterns.push(args[i + 1].clone());
            i += 2;
        } else { i += 1; }
    }

    let min_words = args.iter()
        .position(|a| a == "--min-words")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let max_digit_ratio = args.iter()
        .position(|a| a == "--max-digit-ratio")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.25);
    let min_alpha_ratio = args.iter()
        .position(|a| a == "--min-alpha-ratio")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);

    println!("Scanning {}...", input_path);

    let mut all_conversations = Vec::new();
    let mut processed_files: HashSet<String> = HashSet::new();

    if resume && std::path::Path::new(output_path).exists() {
        if let Ok(existing_json) = fs::read_to_string(output_path) {
            if let Ok(existing_convs) = serde_json::from_str::<Vec<serde_json::Value>>(&existing_json) {
                for conv in &existing_convs {
                    if let Some(name) = conv.get("name").and_then(|n| n.as_str()) {
                        if let Some(filename) = name.strip_prefix("HTML: ") {
                            processed_files.insert(filename.to_string());
                        }
                    }
                }
                all_conversations = existing_convs;
            }
        }
    }

    let all_html_files: Vec<PathBuf> = if fs::metadata(input_path).unwrap().is_file() {
        vec![PathBuf::from(input_path)]
    } else {
        WalkDir::new(input_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "html" | "htm"))
                    .unwrap_or(false)
            })
            .map(|e| e.path().to_path_buf())
            .collect()
    };

    let html_files: Vec<PathBuf> = all_html_files
        .into_iter()
        .filter(|path| {
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let included = should_include_file(filename, &include_patterns);
            let excluded = should_exclude_file(filename, &exclude_patterns);
            included && !excluded
        })
        .collect();

    println!("Processing {} file(s)", html_files.len());
    if html_files.is_empty() {
        eprintln!("No .html files found");
        std::process::exit(1);
    }

    let mut file_count = 0;
    let mut new_files_processed = 0;
    let mut filtered_count = 0;
    let mut total_paragraphs = 0;
    let mut kept_paragraphs = 0;

    let save_progress = |convs: &Vec<serde_json::Value>, path: &str| {
        let json_str = serde_json::to_string_pretty(convs).unwrap();
        fs::write(path, json_str).unwrap();
        println!("  [SAVE] Progress saved ({} conversations)", convs.len());
    };

    for path in &html_files {
        match fs::read_to_string(path) {
            Ok(content) => {
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
                if processed_files.contains(file_name) { continue; }

                let dom = Html::parse_document(&content);
                let title = first_text(&dom, "title").unwrap_or_default();
                let h1 = count_nodes(&dom, "h1");
                let links = count_nodes(&dom, "a[href]");
                let images = count_nodes(&dom, "img");

                let (paragraphs, para_total, para_kept) = if no_filter {
                    let text = gather_text(&dom, &["h1","h2","h3","p","li","blockquote"]);
                    let cleaned = clean_text(&text, keep_numbers);
                    (vec![cleaned], 1, 1)
                } else {
                    gather_filtered_text(&dom, &["h1","h2","h3","p","li","blockquote"],
                        keep_numbers, min_words, max_digit_ratio, min_alpha_ratio)
                };

                total_paragraphs += para_total;
                kept_paragraphs += para_kept;
                let final_text = paragraphs.join(" ");

                let lower = final_text.to_lowercase();
                let trash_markers = [
                    "gnu free documentation license","wikimedia foundation",
                    "all text is available under","wikiproject",
                    "this page was last modified","talk page"
                ];
                if trash_markers.iter().any(|m| lower.contains(m)) { filtered_count+=1; file_count+=1; continue; }
                if final_text.split_whitespace().count() < min_words { filtered_count+=1; file_count+=1; continue; }

                if !final_text.is_empty() {
                    all_conversations.push(json!({
                        "uuid": format!("html-{}", file_count),
                        "name": format!("HTML: {}", file_name),
                        "summary": format!("Raw text from {}", file_name),
                        "meta": { 
                            "title": title, "h1": h1, "links": links, "images": images,
                            "paragraphs_kept": para_kept, "paragraphs_total": para_total
                        },
                        "chat_messages": [{
                            "uuid": format!("msg-{}", file_count),
                            "text": final_text,
                            "sender": "assistant"
                        }]
                    }));
                    new_files_processed += 1;
                    if new_files_processed % save_every == 0 { save_progress(&all_conversations, output_path); }
                } else { filtered_count += 1; }

                file_count += 1;
            }
            Err(e) => eprintln!("  Failed to read {:?}: {}", path, e),
        }
    }

    println!("\nSummary: processed {}, kept {} / {} paragraphs ({:.1}%)",
        file_count, kept_paragraphs, total_paragraphs,
        if total_paragraphs>0 {(kept_paragraphs as f32/total_paragraphs as f32)*100.0}else{0.0});
    let json_str = serde_json::to_string_pretty(&all_conversations).unwrap();
    fs::write(output_path, json_str).unwrap();
    println!("Conversion complete!");
}

// --- helpers ---

fn should_include_file(filename: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() { return true; }
    for pattern in patterns {
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            let mut search_from = 0; let mut all_found = true;
            for part in &parts {
                if part.is_empty() { continue; }
                if let Some(pos) = filename[search_from..].find(part) {
                    search_from += pos + part.len();
                } else { all_found = false; break; }
            }
            if all_found { return true; }
        } else if filename.contains(pattern) { return true; }
    }
    false
}

fn should_exclude_file(filename: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() { return false; }
    for pattern in patterns {
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            let mut search_from = 0; let mut all_found = true;
            for part in &parts {
                if part.is_empty() { continue; }
                if let Some(pos) = filename[search_from..].find(part) {
                    search_from += pos + part.len();
                } else { all_found = false; break; }
            }
            if all_found { return true; }
        } else if filename.contains(pattern) { return true; }
    }
    false
}

fn count_nodes(dom: &Html, css: &str) -> usize {
    Selector::parse(css).ok().map(|sel| dom.select(&sel).count()).unwrap_or(0)
}

fn first_text(dom: &Html, css: &str) -> Option<String> {
    let sel = Selector::parse(css).ok()?;
    let node = dom.select(&sel).next()?;
    Some(node.text().collect::<Vec<_>>().join(" ").trim().to_string())
}

fn should_skip_element(element: ElementRef) -> bool {
    let skip_classes = [
        "mw-editsection","navbox","metadata","navigation","toc","references","reflist","catlinks",
        "printfooter","siteSub","jump-to-nav","mw-jump-link","noprint","sidebar","infobox",
        "thumbcaption","magnify","reference","cite","navigation-not-searchable",
        "mw-footer","mw-indicators","authority-control","sistersitebox","hatnote",
        "shortdescription","ambox","licensetpl","mw-empty-elt"
    ];
    let skip_ids = [
        "toc","siteSub","contentSub","jump-to-nav","mw-navigation","footer","catlinks","references",
        "mw-head","mw-panel","mw-page-base","mw-data-after-content","mw-data-before-content",
        "coordinates","further-reading","external-links","see-also","mw-normal-catlinks","mw-hidden-catlinks"
    ];
    if let Some(class_attr) = element.value().attr("class") {
        if skip_classes.iter().any(|c| class_attr.contains(c)) { return true; }
    }
    if let Some(id_attr) = element.value().attr("id") {
        if skip_ids.iter().any(|c| id_attr.contains(c)) { return true; }
    }
    if let Some(role) = element.value().attr("role") {
        if matches!(role, "navigation" | "note" | "banner") { return true; }
    }
    if let Some(aria) = element.value().attr("aria-label") {
        let a = aria.to_lowercase();
        if a.contains("navigation") || a.contains("breadcrumb") || a.contains("toc") { return true; }
    }
    false
}

fn is_gibberish(text: &str, min_words: usize, max_digit_ratio: f32, min_alpha_ratio: f32) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 10 { return true; }
    let word_count = trimmed.split_whitespace().count();
    if word_count < min_words { return true; }
    let total_chars: usize = trimmed.chars().filter(|c| !c.is_whitespace()).count();
    if total_chars == 0 { return true; }
    let digit_count: usize = trimmed.chars().filter(|c| c.is_numeric()).count();
    let alpha_count: usize = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let digit_ratio = digit_count as f32 / total_chars as f32;
    let alpha_ratio = alpha_count as f32 / total_chars as f32;
    if digit_ratio > max_digit_ratio { return true; }
    if alpha_ratio < min_alpha_ratio { return true; }
    if trimmed.chars().all(|c| c.is_numeric() || c.is_whitespace() || "().-".contains(c)) { return true; }
    let caps_count: usize = trimmed.chars().filter(|c| c.is_uppercase()).count();
    if caps_count > total_chars / 2 && digit_count > total_chars / 3 { return true; }
    let lower = trimmed.to_lowercase();
    if lower.ends_with(" rhs") || lower.ends_with(" lhs") { return true; }
    let eqs = trimmed.matches('=').count();
    if eqs >= 3 && alpha_count < total_chars / 3 { return true; }
    false
}

fn gather_text(dom: &Html, tags: &[&str]) -> String {
    let mut out = String::new();
    for t in tags {
        if let Ok(sel) = Selector::parse(t) {
            for el in dom.select(&sel) {
                if should_skip_element(el) { continue; }
                let seg = el.text().collect::<Vec<_>>().join(" ");
                if !seg.trim().is_empty() {
                    out.push_str(seg.trim());
                    out.push('\n');
                }
            }
        }
    }
    out
}

fn gather_filtered_text(
    dom: &Html, tags: &[&str],
    keep_numbers: bool, min_words: usize,
    max_digit_ratio: f32, min_alpha_ratio: f32
) -> (Vec<String>, usize, usize) {
    let mut paragraphs = Vec::new(); let mut total = 0; let mut kept = 0;
    for t in tags {
        if let Ok(sel) = Selector::parse(t) {
            for el in dom.select(&sel) {
                if should_skip_element(el) { continue; }
                let raw_text = el.text().collect::<Vec<_>>().join(" ");
                let trimmed = raw_text.trim();
                if trimmed.is_empty() { continue; }
                total += 1;
                if is_gibberish(trimmed, min_words, max_digit_ratio, min_alpha_ratio) { continue; }
                let cleaned = clean_text(trimmed, keep_numbers);
                if !cleaned.is_empty() && cleaned.split_whitespace().count() >= min_words {
                    paragraphs.push(cleaned); kept += 1;
                }
            }
        }
    }
    (paragraphs, total, kept)
}

fn clean_text(text: &str, keep_numbers: bool) -> String {
    lazy_static! {
        static ref BRACKETS: Regex = Regex::new(r"\[[0-9]+\]").unwrap();
        static ref CITE_NEED: Regex = Regex::new(r"\s*\(citation needed\)\s*|\s*\[citation needed\]\s*").unwrap();
        static ref WS_MULTI: Regex = Regex::new(r"\s+").unwrap();
    }
        let mut s = text.replace('\u{00A0}', " ");
        s = BRACKETS.replace_all(&s, "").into_owned();
        s = CITE_NEED.replace_all(&s, "").into_owned();
        let s: String = s.chars().map(|c| match c {
            '–'|'—' => '-', '“'|'”'|'’' => '\'', _ => c
        }).collect();
        let s = WS_MULTI.replace_all(&s, " ").to_string();
        s.chars()
            .filter(|c| {
                c.is_alphabetic() || c.is_whitespace() ||
                (keep_numbers && c.is_numeric()) ||
                ['.',';',',','!','?','-','\'',':'].contains(c)
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
}
