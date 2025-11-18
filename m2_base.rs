use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use serde_json::Value;

// ============================================================================
// DATA STRUCTURES - Disk Format (What we save/load from disk)
// ============================================================================

/// Represents a single conversation loaded from JSON
#[derive(Debug, Clone)]
struct DiskCorpus {
    uuid: String,
    name: String,
    messages: Vec<DiskMessage>,
}

#[derive(Debug, Clone)]
struct DiskMessage {
    uuid: String,
    text: String,
    sender: String, // "human" or "assistant"
}

/// N-gram stored on disk with contexts
#[derive(Debug, Clone)]
struct DiskNGram {
    tokens: Vec<String>,      // The n-gram tokens
    count: usize,              // How many times it appears
    contexts: Vec<String>,     // Example contexts (first 5)
}

// ============================================================================
// RAM STRUCTURES - In-memory working data
// ============================================================================

/// N-gram trie node for efficient RAM storage
#[derive(Default, Debug)]
struct NgramTrieNode {
    continuations: HashMap<String, usize>,  // next word -> count
    children: HashMap<String, Box<NgramTrieNode>>,
    contexts: Vec<String>,  // Store up to 5 example contexts
}

impl NgramTrieNode {
    fn new() -> Self {
        NgramTrieNode {
            continuations: HashMap::new(),
            children: HashMap::new(),
            contexts: Vec::new(),
        }
    }

    fn insert(&mut self, context: &[String], next_word: String, full_context: &str) {
        if context.is_empty() {
            // Base case: add to continuations
            *self.continuations.entry(next_word).or_insert(0) += 1;

            // Store context example (limit to 5)
            if self.contexts.len() < 5 {
                self.contexts.push(full_context.to_string());
            }
        } else {
            // Recursive case: navigate deeper
            let first = &context[0];
            let child = self.children
                .entry(first.clone())
                .or_insert_with(|| Box::new(NgramTrieNode::new()));
            child.insert(&context[1..], next_word, full_context);
        }
    }

    fn get_continuations(&self, context: &[String]) -> Option<&HashMap<String, usize>> {
        if context.is_empty() {
            Some(&self.continuations)
        } else {
            self.children
                .get(&context[0])
                .and_then(|child| child.get_continuations(&context[1..]))
        }
    }

    fn get_contexts(&self, context: &[String]) -> Option<&Vec<String>> {
        if context.is_empty() {
            Some(&self.contexts)
        } else {
            self.children
                .get(&context[0])
                .and_then(|child| child.get_contexts(&context[1..]))
        }
    }
}

/// Main trie structure for n-grams
struct NgramTrie {
    root: NgramTrieNode,
    max_order: usize,
}

impl NgramTrie {
    fn new(max_order: usize) -> Self {
        NgramTrie {
            root: NgramTrieNode::new(),
            max_order,
        }
    }

    fn insert_sequence(&mut self, tokens: &[String]) {
        if tokens.len() < 2 {
            return;
        }

        // Build all n-grams up to max_order from this sequence
        for i in 0..tokens.len().saturating_sub(1) {
            for order in 2..=self.max_order.min(tokens.len() - i) {
                if i + order > tokens.len() {
                    break;
                }

                let context = &tokens[i..i + order - 1];
                let next = tokens[i + order - 1].clone();

                // Skip if boundary marker
                if context.iter().any(|t| t == "<MSG>") || next == "<MSG>" {
                    continue;
                }

                // Create context string for storage
                let full_context = tokens[i.saturating_sub(3)..=(i + order - 1).min(tokens.len() - 1)]
                    .join(" ");

                self.root.insert(context, next, &full_context);
            }
        }
    }

    fn get_candidates(&self, context: &[String], max_results: usize) -> Vec<(String, f32, usize, usize)> {
        let mut all_candidates = Vec::new();

        // Try all context lengths from longest to shortest
        for order in (2..=self.max_order.min(context.len() + 1)).rev() {
            if context.len() < order - 1 {
                continue;
            }

            let ctx = &context[context.len().saturating_sub(order - 1)..];

            if let Some(continuations) = self.root.get_continuations(ctx) {
                let total: usize = continuations.values().sum();
                if total >= 2 {
                    for (word, &count) in continuations {
                        let prob = count as f32 / total as f32;
                        let boosted = prob * match order {
                            7 => 4.0,
                            6 => 3.0,
                            5 => 2.5,
                            4 => 2.0,
                            3 => 1.5,
                            _ => 1.0,
                        };
                        all_candidates.push((word.clone(), boosted, order, count));
                    }
                }
            }
        }

        // Deduplicate (keep highest score for each word)
        let mut seen: HashMap<String, (f32, usize, usize)> = HashMap::new();
        for (word, score, order, count) in all_candidates {
            seen.entry(word.clone())
                .and_modify(|e| {
                    if score > e.0 {
                        *e = (score, order, count);
                    }
                })
                .or_insert((score, order, count));
        }

        let mut result: Vec<_> = seen
            .into_iter()
            .map(|(word, (score, order, count))| (word, score, order, count))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(max_results);

        result
    }
}

// ============================================================================
// RAM CACHE & LIMITS
// ============================================================================

struct RAMCache {
    ngram_trie: NgramTrie,
    total_tokens_processed: usize,
    total_messages_processed: usize,
    corpus_names: Vec<String>,

    // Limits
    max_ram_mb: usize,
    current_ram_mb: usize,
}

impl RAMCache {
    fn new(max_ram_mb: usize, max_ngram_order: usize) -> Self {
        RAMCache {
            ngram_trie: NgramTrie::new(max_ngram_order),
            total_tokens_processed: 0,
            total_messages_processed: 0,
            corpus_names: Vec::new(),
            max_ram_mb,
            current_ram_mb: 0,
        }
    }

    fn estimate_ram_usage(&self) -> usize {
        // Rough estimate: each node ~200 bytes, each token ~50 bytes
        let nodes_estimate = self.total_tokens_processed / 10;
        let mb = (nodes_estimate * 200) / (1024 * 1024);
        mb.max(1)
    }

    fn can_load_more(&self) -> bool {
        self.current_ram_mb < self.max_ram_mb
    }

    fn print_stats(&self) {
        println!("\n=== RAM CACHE STATS ===");
        println!("Tokens processed: {}", self.total_tokens_processed);
        println!("Messages processed: {}", self.total_messages_processed);
        println!("Corpora loaded: {}", self.corpus_names.len());
        println!("RAM usage: {} / {} MB", self.current_ram_mb, self.max_ram_mb);
        println!("Max n-gram order: {}", self.ngram_trie.max_order);
        println!("=====================\n");
    }
}

// ============================================================================
// CORPUS LOADING - Extracted from main1.rs
// ============================================================================

/// Load a conversation JSON file (supports both formats)
fn load_conversation_file(path: &str) -> Result<Vec<DiskCorpus>, String> {
    println!("\n[LOAD] Loading file: {}", path);

    let json_text = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let data: Value = serde_json::from_str(&json_text)
        .map_err(|e| format!("Invalid JSON: {}", e))?;

    let mut corpora = Vec::new();

    // Handle array of conversations
    if let Some(conversations) = data.as_array() {
        for (idx, conv) in conversations.iter().enumerate() {
            let mut messages = Vec::new();

            // Format 1: ChatGPT export with "mapping"
            if let Some(mapping) = conv.get("mapping") {
                if let Some(nodes) = mapping.as_object() {
                    for (_node_id, node) in nodes {
                        if let Some(message) = node.get("message") {
                            if let Some(content) = message.get("content") {
                                if let Some(parts) = content.get("parts") {
                                    if let Some(parts_arr) = parts.as_array() {
                                        for part in parts_arr {
                                            if let Some(text) = part.as_str() {
                                                if !text.is_empty() {
                                                    messages.push(DiskMessage {
                                                        uuid: format!("msg-{}-{}", idx, messages.len()),
                                                        text: text.to_string(),
                                                        sender: "unknown".to_string(),
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Format 2: qa_to_json.rs format with "chat_messages"
            else if let Some(chat_messages) = conv.get("chat_messages") {
                if let Some(msgs) = chat_messages.as_array() {
                    for msg in msgs {
                        if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
                            let uuid = msg.get("uuid")
                                .and_then(|u| u.as_str())
                                .unwrap_or(&format!("msg-{}", messages.len()))
                                .to_string();

                            let sender = msg.get("sender")
                                .and_then(|s| s.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            messages.push(DiskMessage {
                                uuid,
                                text: text.to_string(),
                                sender,
                            });
                        }
                    }
                }
            }

            if !messages.is_empty() {
                let uuid = conv.get("uuid")
                    .and_then(|u| u.as_str())
                    .unwrap_or(&format!("corpus-{}", idx))
                    .to_string();

                let name = conv.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("Unnamed Corpus")
                    .to_string();

                corpora.push(DiskCorpus {
                    uuid,
                    name,
                    messages,
                });
            }
        }
    }

    println!("[EXTRACT] Extracted {} corpora", corpora.len());
    Ok(corpora)
}

/// Tokenize text (extracted from main1.rs)
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .flat_map(|word| {
            if word == "<msg>" {
                return vec!["<MSG>".to_string()];
            }

            let clean: String = word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect();

            let punct: String = word.chars()
                .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
                .collect();

            if punct.is_empty() {
                vec![clean]
            } else if clean.is_empty() {
                vec![punct]
            } else {
                vec![clean, punct]
            }
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Process corpora into RAM cache
fn process_corpora_into_ram(corpora: &[DiskCorpus], cache: &mut RAMCache) {
    for corpus in corpora {
        println!("\n[PROCESS] Corpus: {} ({} messages)", corpus.name, corpus.messages.len());

        if !cache.can_load_more() {
            println!("[RAM] Limit reached, skipping remaining corpora");
            break;
        }

        cache.corpus_names.push(corpus.name.clone());

        for msg in &corpus.messages {
            let tokens = tokenize(&msg.text);
            cache.total_tokens_processed += tokens.len();
            cache.total_messages_processed += 1;

            // Build n-grams
            cache.ngram_trie.insert_sequence(&tokens);
        }

        // Update RAM estimate
        cache.current_ram_mb = cache.estimate_ram_usage();

        println!("[RAM] Current usage: {} MB", cache.current_ram_mb);
    }
}

// ============================================================================
// PHASE 2: INTERACTIVE QUERY SYSTEM
// ============================================================================

fn interactive_query_loop(cache: &RAMCache) {
    println!("\n========================================");
    println!("  INTERACTIVE QUERY MODE");
    println!("========================================");
    println!("Commands:");
    println!("  <word> <word> ...  - Query with context");
    println!("  :show <num>        - Show contexts for result #<num>");
    println!("  :stats             - Show cache statistics");
    println!("  :help              - Show this help");
    println!("  :quit              - Exit");
    println!("========================================\n");

    let mut last_query: Vec<String> = Vec::new();
    let mut last_results: Vec<(String, f32, usize, usize)> = Vec::new();

    loop {
        print!("m2> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with(':') {
            if input == ":quit" || input == ":q" || input == ":exit" {
                println!("Goodbye!");
                break;
            } else if input == ":stats" {
                cache.print_stats();
                continue;
            } else if input == ":help" || input == ":h" {
                println!("\nCommands:");
                println!("  <words>      - Query next-word predictions");
                println!("  :show <num>  - Show context examples for result #<num>");
                println!("  :stats       - Show cache statistics");
                println!("  :help        - Show this help");
                println!("  :quit        - Exit\n");
                println!("Examples:");
                println!("  m2> hello");
                println!("  m2> :show 1");
                println!("  m2> how are you");
                println!("  m2> machine learning is\n");
                continue;
            } else if input.starts_with(":show ") {
                let parts: Vec<&str> = input.split_whitespace().collect();
                if parts.len() != 2 {
                    println!("Usage: :show <number>");
                    continue;
                }

                if last_results.is_empty() {
                    println!("  ❌ No query results yet. Run a query first.\n");
                    continue;
                }

                if let Ok(idx) = parts[1].parse::<usize>() {
                    if idx < 1 || idx > last_results.len() {
                        println!("  ❌ Result #{} not found. Valid range: 1-{}\n", idx, last_results.len());
                        continue;
                    }

                    let (word, _, _, _) = &last_results[idx - 1];
                    let full_context = [last_query.clone(), vec![word.clone()]].concat();

                    // Find contexts in the trie
                    if let Some(contexts) = cache.ngram_trie.root.get_contexts(&full_context[..full_context.len()-1]) {
                        println!("\n  📝 Context examples for: {} → {}", last_query.join(" "), word);
                        println!("  Found {} examples:\n", contexts.len());

                        for (i, ctx) in contexts.iter().enumerate() {
                            println!("  {}. \"{}\"", i + 1, ctx);
                        }
                        println!();
                    } else {
                        println!("  ℹ️  No context examples stored for this n-gram.\n");
                    }
                } else {
                    println!("  ❌ Invalid number: {}\n", parts[1]);
                }
                continue;
            } else {
                println!("Unknown command: {}. Type :help for help.", input);
                continue;
            }
        }

        // Parse query into context words
        let context: Vec<String> = input
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        if context.is_empty() {
            continue;
        }

        // Query the cache
        let candidates = cache.ngram_trie.get_candidates(&context, 20);

        if candidates.is_empty() {
            println!("  ❌ No predictions found for: {:?}\n", context);
            continue;
        }

        // Save query results
        last_query = context.clone();
        last_results = candidates.clone();

        // Display results
        println!("\n  Query: {}", context.join(" "));
        println!("  {} predictions found:\n", candidates.len());

        for (i, (word, score, order, count)) in candidates.iter().enumerate() {
            let bar_length = (score * 30.0) as usize;
            let bar: String = "█".repeat(bar_length);

            println!(
                "  {:2}. {:15} [{}-gram] {:30} {:.3} (×{})",
                i + 1,
                word,
                order,
                bar,
                score,
                count
            );
        }

        // Show example contexts for top result
        if let Some((top_word, _, _, _)) = candidates.first() {
            let full_context = [context.clone(), vec![top_word.clone()]].concat();
            println!("\n  💡 Try: {} or :show 1 for examples", full_context.join(" "));
        }

        println!();
    }
}

// ============================================================================
// MAIN - Foundation with Real Data + Interactive Mode
// ============================================================================

fn main() {
    println!("========================================");
    println!("  M2 BASE - Disk/RAM Foundation");
    println!("========================================");
    println!("✅ Real corpus loading (not fake data)");
    println!("✅ N-gram extraction from messages");
    println!("✅ RAM limits and stats");
    println!("✅ Trie-based storage");
    println!("✅ Interactive query mode (Phase 2)");
    println!("========================================\n");

    // Initialize RAM cache (100 MB limit, 7-gram max)
    let mut cache = RAMCache::new(100, 7);

    // Try to load a test file if available
    let test_files = vec![
        "test_conversations.json",
        "conversations.json",
        "../test_conversations.json",
    ];

    let mut loaded = false;
    for file in &test_files {
        if std::path::Path::new(file).exists() {
            match load_conversation_file(file) {
                Ok(corpora) => {
                    println!("[SUCCESS] Loaded {} corpora from {}", corpora.len(), file);
                    process_corpora_into_ram(&corpora, &mut cache);
                    loaded = true;
                    break;
                }
                Err(e) => {
                    println!("[ERROR] Failed to load {}: {}", file, e);
                }
            }
        }
    }

    if !loaded {
        println!("[INFO] No test files found. Create one of:");
        for file in &test_files {
            println!("  - {}", file);
        }
        println!("\nUsing JSON format:");
        println!(r#"[{{
  "uuid": "corpus-1",
  "name": "Test Corpus",
  "chat_messages": [
    {{
      "uuid": "msg-1",
      "text": "Hello world",
      "sender": "human"
    }},
    {{
      "uuid": "msg-2",
      "text": "Hello! How can I help?",
      "sender": "assistant"
    }}
  ]
}}]"#);
        println!("\nExiting (no data to query).");
        return;
    }

    // Print stats
    cache.print_stats();

    // Enter interactive mode
    interactive_query_loop(&cache);

    println!("\n========================================");
    println!("  Session Ended");
    println!("========================================");
}
