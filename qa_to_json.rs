use std::fs;
use std::path::PathBuf;
use serde_json::json;
use walkdir::WalkDir;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: corpus_generator <input_path> <output.json> [--keep-numbers] [--raw-lines]");
        eprintln!("  input_path: file or directory to process");
        eprintln!("  output.json: where to write the conversation JSON");
        eprintln!("  --keep-numbers: optional flag to preserve numbers in text");
        eprintln!("  --raw-lines: treat each non-empty line as separate message");
        eprintln!("");
        eprintln!("Automatically detects:");
        eprintln!("  - Q&A format (Q: ... A: ... lines)");
        eprintln!("  - Raw text files (entire file as one message)");
        eprintln!("  - Raw lines mode (each line as separate message)");
        eprintln!("  - Single file or recursive directory scan");
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = &args[2];
    let keep_numbers = args.iter().any(|a| a == "--keep-numbers");
    let raw_lines = args.iter().any(|a| a == "--raw-lines");
    
    println!("Scanning {}...", input_path);
    if raw_lines {
        println!("Mode: RAW LINES (each line = separate message)");
    }
    
    // Collect all .txt files (either single file or recursive scan)
    let txt_files: Vec<PathBuf> = if fs::metadata(input_path).unwrap().is_file() {
        vec![PathBuf::from(input_path)]
    } else {
        WalkDir::new(input_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("txt"))
            .map(|e| e.path().to_path_buf())
            .collect()
    };
    
    println!("Found {} .txt file(s)", txt_files.len());
    
    if txt_files.is_empty() {
        eprintln!("No .txt files found");
        std::process::exit(1);
    }
    
    // Process all files
    let mut all_conversations = Vec::new();
    let mut file_count = 0;
    let mut qa_count = 0;
    let mut raw_count = 0;
    let mut line_count = 0;
    
    for path in &txt_files {
        match fs::read_to_string(path) {
            Ok(content) => {
                let file_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                
                // Try to detect format
                if is_qa_format(&content) && !raw_lines {
                    // Q&A format
                    let messages = parse_qa_format(&content, keep_numbers);
                    if !messages.is_empty() {
                        all_conversations.push(json!({
                            "uuid": format!("qa-{}", file_count),
                            "name": format!("Q&A: {}", file_name),
                            "summary": format!("Converted Q&A from {}", file_name),
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "chat_messages": messages
                        }));
                        qa_count += 1;
                        println!("  Q&A format: {} ({} pairs)", file_name, messages.len() / 2);
                    }
                } else if raw_lines {
                    // Raw lines mode - each line is a separate message
                    let messages = parse_raw_lines(&content, keep_numbers);
                    if !messages.is_empty() {
                        all_conversations.push(json!({
                            "uuid": format!("lines-{}", file_count),
                            "name": format!("Lines: {}", file_name),
                            "summary": format!("Line-by-line from {}", file_name),
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "chat_messages": messages
                        }));
                        line_count += 1;
                        println!("  Raw lines: {} ({} lines)", file_name, messages.len());
                    }
                } else {
                    // Raw text format (entire file as one message)
                    let cleaned = clean_text(&content, keep_numbers);
                    if !cleaned.is_empty() {
                        all_conversations.push(json!({
                            "uuid": format!("text-{}", file_count),
                            "name": format!("Text: {}", file_name),
                            "summary": format!("Raw text from {}", file_name),
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "chat_messages": [{
                                "uuid": format!("msg-{}", file_count),
                                "text": cleaned,
                                "sender": "assistant",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-01T00:00:00Z",
                            }]
                        }));
                        raw_count += 1;
                        println!("  Raw text: {} ({} chars)", file_name, cleaned.len());
                    }
                }
                
                file_count += 1;
                if file_count % 100 == 0 {
                    println!("  Progress: {} files processed...", file_count);
                }
            }
            Err(e) => {
                eprintln!("  Failed to read {:?}: {}", path, e);
            }
        }
    }
    
    println!("\nProcessing Summary:");
    println!("   Total files: {}", file_count);
    println!("   Q&A format: {}", qa_count);
    println!("   Raw text: {}", raw_count);
    println!("   Raw lines: {}", line_count);
    println!("   Conversations: {}", all_conversations.len());
    
    // Write JSON
    println!("\nWriting to {}...", output_path);
    let json_str = serde_json::to_string_pretty(&all_conversations)
        .expect("Failed to serialize");
    fs::write(output_path, json_str).expect("Failed to write output file");
    
    println!("Conversion complete!");
    println!("   Output: {}", output_path);
}

fn is_qa_format(text: &str) -> bool {
    // Check if file looks like Q&A format
    let lines: Vec<&str> = text.lines().collect();
    let q_lines = lines.iter().filter(|l| l.trim().starts_with("Q:")).count();
    let a_lines = lines.iter().filter(|l| l.trim().starts_with("A:")).count();
    
    // If we have at least 2 Q&A pairs and they're balanced, it's Q&A format
    q_lines >= 2 && a_lines >= 2 && (q_lines as i32 - a_lines as i32).abs() <= 1
}

fn parse_qa_format(text: &str, keep_numbers: bool) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    let mut msg_id = 0;
    
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        if let Some(question) = line.strip_prefix("Q:").or_else(|| line.strip_prefix("q:")) {
            // Extract question (before any NEXT: or |NEXT: markers)
            let clean_q = question
                .split("|NEXT:")
                .next()
                .unwrap_or(question)
                .split("NEXT:")
                .next()
                .unwrap_or(question)
                .trim();
            
            let filtered = clean_text(clean_q, keep_numbers);
            if !filtered.is_empty() {
                messages.push(json!({
                    "uuid": format!("q-{}", msg_id),
                    "text": filtered,
                    "sender": "human",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }));
                msg_id += 1;
            }
        } else if let Some(answer) = line.strip_prefix("A:").or_else(|| line.strip_prefix("a:")) {
            // Extract answer (before any NEXT: markers)
            let clean_a = answer
                .split("|NEXT:")
                .next()
                .unwrap_or(answer)
                .split("NEXT:")
                .next()
                .unwrap_or(answer)
                .trim();
            
            let filtered = clean_text(clean_a, keep_numbers);
            if !filtered.is_empty() {
                messages.push(json!({
                    "uuid": format!("a-{}", msg_id),
                    "text": filtered,
                    "sender": "assistant",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }));
                msg_id += 1;
            }
        }
    }
    
    messages
}

fn parse_raw_lines(text: &str, keep_numbers: bool) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    let mut msg_id = 0;
    
    for line in text.lines() {
        let line = line.trim();
        
        // Skip empty lines
        if line.is_empty() {
            continue;
        }
        
        let cleaned = clean_text(line, keep_numbers);
        if !cleaned.is_empty() {
            messages.push(json!({
                "uuid": format!("line-{}", msg_id),
                "text": cleaned,
                "sender": "assistant",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }));
            msg_id += 1;
        }
    }
    
    messages
}

fn clean_text(text: &str, keep_numbers: bool) -> String {
    text.chars()
        .filter(|c| {
            // Keep: letters, spaces, and basic sentence punctuation
            c.is_alphabetic() || 
            c.is_whitespace() || 
            (keep_numbers && c.is_numeric()) ||
            *c == '.' || 
            *c == ',' || 
            *c == '!' || 
            *c == '?' || 
            *c == ';' || 
            *c == ':' || 
            *c == '\'' ||  // apostrophes for contractions
            *c == '-'      // hyphens for compound words
        })
        .collect::<String>()
        // Collapse multiple spaces
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_qa_format() {
        let qa_text = "Q: What is this?\nA: This is a test.\nQ: Another?\nA: Yes.";
        assert!(is_qa_format(qa_text));
        
        let raw_text = "This is just some raw text without Q&A structure.";
        assert!(!is_qa_format(raw_text));
    }
    
    #[test]
    fn test_clean_text() {
        assert_eq!(
            clean_text("Hello123 world456!", false),
            "Hello world!"
        );
        
        assert_eq!(
            clean_text("Hello123 world456!", true),
            "Hello123 world456!"
        );
        
        assert_eq!(
            clean_text("Test @ #symbols $%^ removal", false),
            "Test symbols removal"
        );
        
        assert_eq!(
            clean_text("Keep hyphen-words and don't remove apostrophes.", false),
            "Keep hyphen-words and don't remove apostrophes."
        );
    }
    
    #[test]
    fn test_parse_qa_format() {
        let text = "Q: What is life?\nA: Life is complex.\nQ: Really?\nA: Yes!";
        let messages = parse_qa_format(text, false);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["sender"], "human");
        assert_eq!(messages[1]["sender"], "assistant");
    }
    
    #[test]
    fn test_parse_raw_lines() {
        let text = "First line of text.\nSecond line here.\n\nFourth line (third was empty).";
        let messages = parse_raw_lines(text, false);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["text"], "First line of text.");
        assert_eq!(messages[1]["text"], "Second line here.");
        assert_eq!(messages[2]["text"], "Fourth line third was empty.");
    }
}