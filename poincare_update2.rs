use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use serde_json::Value;
use macroquad::prelude::*;
use rfd::FileDialog;
use std::path::PathBuf;

const GREY_BG: Color = Color::new(0.15, 0.15, 0.15, 1.0);
const BLACK: Color = Color::new(0.0, 0.0, 0.0, 1.0);
const GREEN: Color = Color::new(0.0, 1.0, 0.0, 1.0);
const CYAN: Color = Color::new(0.0, 1.0, 1.0, 1.0);
const YELLOW: Color = Color::new(1.0, 1.0, 0.0, 1.0);
const ORANGE: Color = Color::new(1.0, 0.5, 0.0, 1.0);
const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);
const MAGENTA: Color = Color::new(1.0, 0.0, 1.0, 1.0);
const RED: Color = Color::new(1.0, 0.0, 0.0, 1.0);
const BLUE: Color = Color::new(0.3, 0.3, 1.0, 1.0);

// ============================================================================
// HYPERBOLIC GEOMETRY - POINCARÉ DISC
// ============================================================================

struct PoincareDisc {
    radius: f32,
    inner_radius: f32,
}

impl PoincareDisc {
    fn new(radius: f32, inner_radius: f32) -> Self {
        PoincareDisc { radius, inner_radius }
    }
    
    fn freq_to_radius(&self, freq_rank: f32) -> f32 {
        let max_r = 0.95;
        let compression = 2.5;
        
        let normalized = max_r * (1.0 - (-compression * freq_rank).exp());
        
        self.inner_radius + normalized * (1.0 - self.inner_radius)
    }
    
    fn radius_to_size(&self, r: f32) -> f32 {
        let boundary_factor = 1.0 - r;
        
        (boundary_factor.powf(0.5) * 5.0 + 0.5).max(0.3)
    }
    
    fn hyperbolic_distance(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
        let r1_sq = x1 * x1 + y1 * y1;
        let r2_sq = x2 * x2 + y2 * y2;
        
        let numerator = (x1 - x2).powi(2) + (y1 - y2).powi(2);
        let denominator = (1.0 - r1_sq) * (1.0 - r2_sq);
        
        if denominator <= 0.0 {
            return 100.0;
        }
        
        (1.0 + 2.0 * numerator / denominator).acosh().max(0.0)
    }
    
    fn recenter(&self, px: f32, py: f32, points: &mut [(f32, f32)]) {
        let p_norm_sq = px * px + py * py;
        
        if p_norm_sq < 0.001 {
            return;
        }
        
        for (x, y) in points.iter_mut() {
            let old_x = *x;
            let old_y = *y;
            
            let numerator_x = old_x - px;
            let numerator_y = old_y - py;
            let denominator = 1.0 - (px * old_x + py * old_y);
            
            if denominator.abs() < 0.001 {
                continue;
            }
            
            *x = numerator_x / denominator;
            *y = numerator_y / denominator;
        }
    }
}

// ============================================================================
// LAYER 1: DATA STRUCTURES
// ============================================================================

#[derive(Clone)]
struct Word {
    text: String,
    x: f32,
    y: f32,
    freq: usize,
    freq_rank: f32,
    base_size: f32,
    activation: f32,
    layer: usize,
    in_context: bool,
    embedding: Option<Vec<f32>>,
    render_priority: f32,
    cached_color: Option<(Color, Color)>,
    max_ngram_order: usize,
    ngram_confidence: f32,
    avg_ngram_order: f32,
}

#[derive(Clone)]
struct SemanticCell {
    center_word: String,
    corners: Vec<(f32, f32)>,
    color: Color,
    activation: f32,
}

struct ExpandedWord {
    text: String,
    x: f32,
    y: f32,
    size: f32,
    ngram_order: usize,
    coherence: f32,
}

struct BridgeWord {
    text: String,
    x: f32,
    y: f32,
    size: f32,
    ngram_order: usize,
    coherence: f32,
    from_word: String,
    to_word: String,
}

struct EmbeddingSpace {
    embeddings: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl EmbeddingSpace {
    fn new(dim: usize) -> Self {
        EmbeddingSpace {
            embeddings: HashMap::new(),
            dim,
        }
    }
    
    fn load_glove(&mut self, path: &str) -> Result<usize, String> {
        println!(" Loading embeddings from {}...", path);
        
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut count = 0;
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.len() < 2 {
                continue;
            }
            
            let word = parts[0].to_string();
            let values: Result<Vec<f32>, _> = parts[1..]
                .iter()
                .map(|s| s.parse::<f32>())
                .collect();
            
            if let Ok(vec) = values {
                if vec.len() == self.dim {
                    self.embeddings.insert(word, vec);
                    count += 1;
                }
            }
        }
        
        println!(" Loaded {} embeddings (dim={})", count, self.dim);
        Ok(count)
    }
    
    fn get_semantic_center(&self, context: &[String]) -> Vec<f32> {
        let mut center = vec![0.0; self.dim];
        let mut count = 0;
        
        for word in context.iter().rev().take(5) {
            if let Some(emb) = self.embeddings.get(word) {
                for (i, val) in emb.iter().enumerate() {
                    center[i] += val;
                }
                count += 1;
            }
        }
        
        if count > 0 {
            for val in &mut center {
                *val /= count as f32;
            }
        }
        
        center
    }
}

struct PunctuationModel {
    word_to_punct: HashMap<String, HashMap<char, usize>>,
}

impl PunctuationModel {
    fn new() -> Self {
        PunctuationModel {
            word_to_punct: HashMap::new(),
        }
    }
    
    fn train(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let end_markers = vec!['.', '!', '?'];
        
        for word in words {
            if let Some(last_char) = word.chars().last() {
                if end_markers.contains(&last_char) {
                    let clean_word = word.trim_end_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                    
                    *self.word_to_punct
                        .entry(clean_word)
                        .or_insert_with(HashMap::new)
                        .entry(last_char)
                        .or_insert(0) += 1;
                }
            }
        }
    }
    
    fn suggest_punctuation(&self, word: &str, tokens_since: usize) -> Option<char> {
        if tokens_since > 20 {
            return Some('.');
        }
        
        if let Some(punct_counts) = self.word_to_punct.get(word) {
            let total: usize = punct_counts.values().sum();
            if total > 5 && tokens_since > 8 {
                if let Some((&punct, &count)) = punct_counts.iter().max_by_key(|(_, &c)| c) {
                    if count as f32 / total as f32 > 0.3 {
                        return Some(punct);
                    }
                }
            }
        }
        
        None
    }
}

#[derive(Clone)]
struct FileLoadInfo {
    filename: String,
    ngram_level: usize,
    load_count: usize,
}

// ============================================================================
// DISK-BACKED N-GRAM STORAGE
// ============================================================================

struct DiskBatchManager {
    batch_dir: PathBuf,
    current_batch: HashMap<String, HashMap<String, usize>>,
    batch_counter: usize,
    max_batch_size: usize,
    batch_files: Vec<PathBuf>,
}

impl DiskBatchManager {
    fn new(max_batch_size: usize) -> Self {
        let batch_dir = std::env::temp_dir().join("wordmap_batches");
        fs::create_dir_all(&batch_dir).ok();
        
        DiskBatchManager {
            batch_dir,
            current_batch: HashMap::new(),
            batch_counter: 0,
            max_batch_size,
            batch_files: Vec::new(),
        }
    }
    
    fn add_ngram(&mut self, context: &str, next_word: String) {
        self.current_batch
            .entry(context.to_string())
            .or_insert_with(HashMap::new)
            .entry(next_word)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        
        if self.current_batch.len() >= self.max_batch_size {
            self.flush_batch();
        }
    }
    
    fn flush_batch(&mut self) {
        if self.current_batch.is_empty() {
            return;
        }
        
        let batch_path = self.batch_dir.join(format!("batch_{}.txt", self.batch_counter));
        
        if let Ok(file) = fs::File::create(&batch_path) {
            let mut writer = BufWriter::new(file);
            
            for (context, continuations) in &self.current_batch {
                for (word, count) in continuations {
                    writeln!(writer, "{}\t{}\t{}", context, word, count).ok();
                }
            }
            
            self.batch_files.push(batch_path);
            self.batch_counter += 1;
        }
        
        self.current_batch.clear();
        println!("  [DISK] Flushed batch {} ({} files total)", self.batch_counter, self.batch_files.len());
    }
    
    fn get_candidates(&self, context: &str, max_results: usize) -> Vec<(String, usize)> {
        let mut all_candidates: HashMap<String, usize> = HashMap::new();
        
        if let Some(continuations) = self.current_batch.get(context) {
            for (word, count) in continuations {
                *all_candidates.entry(word.clone()).or_insert(0) += count;
            }
        }
        
        for batch_file in &self.batch_files {
            if let Ok(file) = fs::File::open(batch_file) {
                let reader = BufReader::new(file);
                
                for line in reader.lines().flatten() {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() == 3 && parts[0] == context {
                        if let Ok(count) = parts[2].parse::<usize>() {
                            *all_candidates.entry(parts[1].to_string()).or_insert(0) += count;
                        }
                    }
                }
            }
        }
        
        let mut result: Vec<_> = all_candidates.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.truncate(max_results);
        result
    }
    
    fn cleanup(&self) {
        for batch_file in &self.batch_files {
            fs::remove_file(batch_file).ok();
        }
        fs::remove_dir(&self.batch_dir).ok();
    }
}

impl Drop for DiskBatchManager {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ============================================================================
// OPTIMIZED N-GRAM TRIE STRUCTURE (HYBRID: Memory + Disk)
// ============================================================================

#[derive(Default)]
struct NgramTrieNode {
    continuations: HashMap<String, usize>,
    children: HashMap<String, Box<NgramTrieNode>>,
}

impl NgramTrieNode {
    fn new() -> Self {
        NgramTrieNode {
            continuations: HashMap::new(),
            children: HashMap::new(),
        }
    }
    
    fn insert(&mut self, context: &[String], next_word: String) {
        if context.is_empty() {
            *self.continuations.entry(next_word).or_insert(0) += 1;
        } else {
            let first = &context[0];
            let child = self.children.entry(first.clone())
                .or_insert_with(|| Box::new(NgramTrieNode::new()));
            child.insert(&context[1..], next_word);
        }
    }
    
    fn get_continuations(&self, context: &[String]) -> Option<&HashMap<String, usize>> {
        if context.is_empty() {
            Some(&self.continuations)
        } else {
            self.children.get(&context[0])
                .and_then(|child| child.get_continuations(&context[1..]))
        }
    }
}

struct NgramTrie {
    root: NgramTrieNode,
    max_order: usize,
    disk_manager: DiskBatchManager,
    use_disk: bool,
    memory_limit: usize,
    current_memory_usage: usize,
}

impl NgramTrie {
    fn new(max_order: usize) -> Self {
        NgramTrie {
            root: NgramTrieNode::new(),
            max_order,
            disk_manager: DiskBatchManager::new(50000),
            use_disk: false,
            memory_limit: 100_000_000,
            current_memory_usage: 0,
        }
    }
    
    fn enable_disk_mode(&mut self) {
        self.use_disk = true;
        println!("[DISK] Disk-backed n-gram storage enabled");
    }
    
    fn insert(&mut self, tokens: &[String]) {
        if tokens.len() < 2 {
            return;
        }
        
        for i in 0..tokens.len().saturating_sub(1) {
            for order in 2..=self.max_order.min(tokens.len() - i) {
                if i + order > tokens.len() {
                    break;
                }
                
                let context = &tokens[i..i + order - 1];
                let next = tokens[i + order - 1].clone();
                
                if context.iter().any(|t| t == "<MSG>") || next == "<MSG>" {
                    continue;
                }
                
                if self.use_disk {
                    let context_key = context.join("||");
                    self.disk_manager.add_ngram(&context_key, next);
                } else {
                    self.root.insert(context, next);
                    self.current_memory_usage += context.len() * 20;
                    
                    if self.current_memory_usage > self.memory_limit {
                        println!("[DISK] Memory limit reached, switching to disk storage");
                        self.enable_disk_mode();
                        self.disk_manager.flush_batch();
                    }
                }
            }
        }
    }
    
    fn get_candidates(&self, context: &[String], max_results: usize) -> Vec<(String, f32, usize, usize)> {
        let mut all_candidates = Vec::new();
        
        if !self.use_disk {
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
                            let boosted = prob * Self::get_ngram_boost(order);
                            all_candidates.push((word.clone(), boosted, order, count));
                        }
                    }
                }
            }
        } else {
            for order in (2..=self.max_order.min(context.len() + 1)).rev() {
                if context.len() < order - 1 {
                    continue;
                }
                
                let ctx = &context[context.len().saturating_sub(order - 1)..];
                let context_key = ctx.join("||");
                
                let disk_candidates = self.disk_manager.get_candidates(&context_key, max_results * 2);
                
                if !disk_candidates.is_empty() {
                    let total: usize = disk_candidates.iter().map(|(_, c)| c).sum();
                    if total >= 2 {
                        for (word, count) in disk_candidates {
                            let prob = count as f32 / total as f32;
                            let boosted = prob * Self::get_ngram_boost(order);
                            all_candidates.push((word, boosted, order, count));
                        }
                    }
                }
            }
        }
        
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
        
        let mut result: Vec<_> = seen.into_iter()
            .map(|(word, (score, order, count))| (word, score, order, count))
            .collect();
        
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(max_results);
        
        result
    }
    
    fn get_ngram_boost(order: usize) -> f32 {
        match order {
            12 => 7.0,
            11 => 6.5,
            10 => 6.0,
            9 => 5.5,
            8 => 5.0,
            7 => 4.0,
            6 => 3.0,
            5 => 2.5,
            4 => 2.0,
            3 => 1.5,
            _ => 0.5,
        }
    }
}

// ============================================================================
// LAYER 2: SCORING SYSTEM
// ============================================================================

#[derive(Clone)]
struct ScoredCandidate {
    word: String,
    fluency: f32,
    coherence: f32,
    diversity: f32,
    confidence: f32,
    total: f32,
    ngram_order: usize,
    is_from_dict: bool,
}

struct ScoringWeights {
    fluency: f32,
    coherence: f32,
    diversity: f32,
    confidence: f32,
}

impl ScoringWeights {
    fn balanced() -> Self {
        ScoringWeights {
            fluency: 0.4,
            coherence: 0.3,
            diversity: 0.2,
            confidence: 0.1,
        }
    }
}
// ============================================================================
// MAIN WORDMAP - HYPERBOLIC VISUALIZATION
// ============================================================================

struct WordMap {
    words: Vec<Word>,
    expanded_words: Vec<ExpandedWord>,
    bridge_words: Vec<BridgeWord>,
    semantic_cells: Vec<SemanticCell>,
    
    ngram_trie: NgramTrie,
    embeddings: EmbeddingSpace,
    
    dict_corpus_loaded: bool,
    punct_model: PunctuationModel,
    weights: ScoringWeights,
    
    poincare: PoincareDisc,
    focus_word: Option<String>,
    render_threshold: f32,
    
    selected: Option<String>,
    context_path: Vec<String>,
    files_loaded: usize,
    camera_x: f32,
    camera_y: f32,
    camera_zoom: f32,
    dragging: bool,
    last_mouse: (f32, f32),
    query_text: String,
    text_input_active: bool,
    semantic_mode: bool,
    
    activation_by_word: HashMap<String, f32>,
    any_active: bool,
    redraw_needed: bool,
    last_camera: (f32, f32, f32),
    frame_counter: u32,

    max_ngram_order: usize,
    config_mode: bool,
    config_input: String,
    
    generating: bool,
    best_path: Vec<String>,
    playback_index: usize,
    generation_timer: f32,
    generation_delay: f32,
    generated_output: String,
    exploring: bool,
    ngram_orders_used: Vec<usize>,
    generation_steps: Vec<GenerationStep>,
    use_embeddings: bool,
    
    loaded_files: HashMap<String, FileLoadInfo>,
    target_chunk_tokens: usize,
}

#[derive(Clone)]
struct GenerationStep {
    token: String,
    fluency: f32,
    coherence: f32,
    diversity: f32,
    confidence: f32,
    total: f32,
    ngram_order: usize,
    entropy: f32,
    structural_entropy: f32,
    from_dict: bool,
}

impl WordMap {
    fn new() -> Self {
        WordMap {
            words: Vec::new(),
            expanded_words: Vec::new(),
            bridge_words: Vec::new(),
            semantic_cells: Vec::new(),
            ngram_trie: NgramTrie::new(12),
            embeddings: EmbeddingSpace::new(200),
            dict_corpus_loaded: false,
            punct_model: PunctuationModel::new(),
            weights: ScoringWeights::balanced(),
            poincare: PoincareDisc::new(1500.0, 0.135),
            focus_word: None,
            render_threshold: 25000.0,
            selected: None,
            context_path: Vec::new(),
            files_loaded: 0,
            camera_x: 0.0,
            camera_y: 0.0,
            camera_zoom: 1.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
            query_text: String::new(),
            text_input_active: false,
            semantic_mode: false,
            max_ngram_order: 12,
            config_mode: false,
            config_input: String::from("12"),
            generating: false,
            best_path: Vec::new(),
            playback_index: 0,
            generation_timer: 0.0,
            generation_delay: 0.13, 
            generated_output: String::new(),
            exploring: false,
            ngram_orders_used: Vec::new(),
            generation_steps: Vec::new(),
            use_embeddings: false,
            loaded_files: HashMap::new(),
            target_chunk_tokens: 5000,
            activation_by_word: HashMap::new(),
            any_active: false,
            redraw_needed: true,
            last_camera: (0.0, 0.0, 1.0),
            frame_counter: 0,
        }
    }
    
    fn load_embeddings(&mut self, path: &str) {
        match self.embeddings.load_glove(path) {
            Ok(_count) => {
                self.use_embeddings = true;
                println!("[***] DUAL TOPOLOGY ENABLED! Using embeddings + n-grams");
                
                for word in &mut self.words {
                    if let Some(emb) = self.embeddings.embeddings.get(&word.text) {
                        word.embedding = Some(emb.clone());
                    }
                }
                
                self.compute_bridge_words();
            }
            Err(e) => {
                println!(" Failed to load embeddings: {}", e);
                println!("  Continuing with n-grams only");
                self.use_embeddings = false;
            }
        }
    }
    
    fn load_file(&mut self, path: &str) {
        println!("\n[LOAD] Loading file: {}", path);
        self.redraw_needed = true;
        let is_dict = path.to_lowercase().contains("dict") || 
                      path.to_lowercase().contains("definition");
        
        let filename = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .to_string();
        
        let file_key = format!("{}@{}", filename, self.max_ngram_order);
        if self.loaded_files.contains_key(&file_key) {
            println!("[SKIP] File already loaded at {}-gram level: {}", self.max_ngram_order, filename);
            return;
        }
        
        let json_text = fs::read_to_string(path).expect("Failed to read file");
        let data: Value = serde_json::from_str(&json_text).expect("Invalid JSON");
        
        let existing_vocab: std::collections::HashSet<_> = 
            self.words.iter().map(|w| w.text.clone()).collect();
        
        println!("[EXTRACT] Extracting text from JSON...");
        let mut all_text = String::new();
        let mut raw_text = String::new();
        
        if let Some(conversations) = data.as_array() {
            for conv in conversations {
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
                                                        raw_text.push_str(text);
                                                        raw_text.push_str(" <MSG> ");
                                                        all_text.push_str(text);
                                                        all_text.push_str(" <MSG> ");
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
                else if let Some(messages) = conv.get("chat_messages") {
                    if let Some(msgs) = messages.as_array() {
                        for msg in msgs {
                            if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
                                raw_text.push_str(text);
                                raw_text.push_str(" <MSG> ");
                                all_text.push_str(text);
                                all_text.push_str(" <MSG> ");
                            }
                        }
                    }
                }
            }
        }
        
        println!("[PUNCT] Training punctuation model...");
        self.punct_model.train(&raw_text);
        
        println!("[TOKEN] Tokenizing...");
        let tokens: Vec<String> = all_text
            .to_lowercase()
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
            .collect();
        
        let total_tokens = tokens.len();
        println!("[STATS] File contains {} total tokens", total_tokens);
        
        let num_chunks = ((total_tokens as f32 / self.target_chunk_tokens as f32).ceil() as usize).max(1);
        let tokens_per_chunk = (total_tokens + num_chunks - 1) / num_chunks;
        
        println!("[BATCH] Processing n-grams: {} chunks of ~{} tokens each", 
            num_chunks, tokens_per_chunk);
        
        if total_tokens > 1_000_000 {
            println!("[DISK] Large file detected, enabling disk-backed storage");
            self.ngram_trie.enable_disk_mode();
        }
        
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * tokens_per_chunk;
            let end = ((chunk_idx + 1) * tokens_per_chunk).min(total_tokens);
            let chunk_tokens = &tokens[start..end];
            
            println!("[CHUNK {}/{}] Building n-grams (tokens {}-{})...", 
                chunk_idx + 1, num_chunks, start, end);
            self.redraw_needed = true;
            self.ngram_trie.insert(chunk_tokens);
            
            if self.ngram_trie.use_disk && (chunk_idx + 1) % 5 == 0 {
                self.ngram_trie.disk_manager.flush_batch();
            }
        }
        
        if self.ngram_trie.use_disk {
            self.ngram_trie.disk_manager.flush_batch();
            println!("[DISK] All batches flushed to disk");
        }
        
        self.loaded_files.insert(file_key.clone(), FileLoadInfo {
            filename: filename.clone(),
            ngram_level: self.max_ngram_order,
            load_count: 1,
        });
        
        self.files_loaded += 1;
        
        let mut new_freq: HashMap<String, usize> = HashMap::new();
        for word in &tokens {
            *new_freq.entry(word.clone()).or_insert(0) += 1;
        }
        
        let unique_in_file = new_freq.len();
        println!("[VOCAB] File contains {} unique words (from {} total tokens)", 
            unique_in_file, total_tokens);
        
        let mut sorted_words: Vec<_> = new_freq.iter()
            .filter(|(w, _)| !existing_vocab.contains(*w))
            .collect();
        sorted_words.sort_by(|a, b| b.1.cmp(a.1));
        sorted_words.truncate(250000);
        
        let new_words_to_add = sorted_words.len();
        let already_known = unique_in_file - new_words_to_add;
        
        println!("[VOCAB] {} new words to add | {} already in vocabulary", 
            new_words_to_add, already_known);
        
        if !sorted_words.is_empty() {
            let total_new_words = sorted_words.len();
            let words_before = self.words.len();
            
            for (rank, (word, freq)) in sorted_words.iter().enumerate() {
                let freq_rank = rank as f32 / total_new_words as f32;
                let r = self.poincare.freq_to_radius(freq_rank);
                let angle = (rank as f32 * 2.4) % (2.0 * std::f32::consts::PI);
                let x = r * angle.cos();
                let y = r * angle.sin();
                let size = self.poincare.radius_to_size(r) * 0.5;
                let render_priority = (1.0 - freq_rank) * 0.7 + (1.0 - r) * 0.3;
                
                self.words.push(Word {
                    text: (*word).clone(),
                    x, y,
                    freq: **freq,
                    freq_rank,
                    base_size: size,
                    activation: 0.0,
                    layer: self.files_loaded - 1,
                    in_context: false,
                    embedding: None,
                    render_priority,
                    cached_color: None,
                    max_ngram_order: 2,
                    ngram_confidence: 0.0,
                    avg_ngram_order: 2.0,
                });
            }
            
            println!("[VOCAB] Vocabulary: {} → {} words (+{} from this file)", 
                words_before, self.words.len(), sorted_words.len());
        }
        
        self.compute_ngram_confidence();
        
        println!("[DONE] File #{} loaded | Total vocabulary: {} unique words", 
            self.files_loaded, self.words.len());
        
        if is_dict {
            self.dict_corpus_loaded = true;
            println!("[INFO] Dictionary corpus detected");
        }
        
        if self.use_embeddings {
            self.compute_bridge_words();
        }
    }
    
    fn compute_ngram_confidence(&mut self) {
        println!("[NGRAM] Computing average n-gram confidence for {} words...", self.words.len());
        
        for word in &mut self.words {
            let context = vec![word.text.clone()];
            let candidates = self.ngram_trie.get_candidates(&context, 50);
            
            if !candidates.is_empty() {
                let total_order: usize = candidates.iter()
                    .map(|(_, _, order, count)| order * count)
                    .sum();
                
                let total_count: usize = candidates.iter()
                    .map(|(_, _, _, count)| count)
                    .sum();
                
                let avg_order = if total_count > 0 {
                    total_order as f32 / total_count as f32
                } else {
                    2.0
                };
                
                let max_order = candidates.iter()
                    .map(|(_, _, order, _)| order)
                    .max()
                    .copied()
                    .unwrap_or(2);
                
                word.avg_ngram_order = avg_order;
                word.max_ngram_order = max_order;
                word.ngram_confidence = Self::ngram_order_to_confidence(avg_order as usize);
            }
        }
        
        println!("[NGRAM] Average confidence computed");
    }
    
    fn ngram_order_to_confidence(order: usize) -> f32 {
        match order {
            12 => 1.00, 11 => 0.95, 10 => 0.90, 9 => 0.85, 8 => 0.80,
            7 => 0.75, 6 => 0.65, 5 => 0.55, 4 => 0.45, 3 => 0.30,
            2 => 0.05, _ => 0.0,
        }
    }
    
    fn confidence_to_hsv_color(confidence: f32, position: (f32, f32)) -> Color {
        let angle = position.1.atan2(position.0);
        let hue = ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)) * 360.0;
        let saturation = 0.2 + confidence * 0.7;
        let value = 0.4 + confidence * 0.5;
        Self::hsv_to_rgb(hue, saturation, value)
    }
    
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h < 60.0 { (c, x, 0.0) }
        else if h < 120.0 { (x, c, 0.0) }
        else if h < 180.0 { (0.0, c, x) }
        else if h < 240.0 { (0.0, x, c) }
        else if h < 300.0 { (x, 0.0, c) }
        else { (c, 0.0, x) };
        
        Color::new(r + m, g + m, b + m, 1.0)
    }
    
    fn ngram_order_to_color(order: usize) -> Color {
        match order {
            12 => Color::new(1.0, 0.0, 1.0, 1.0),
            11 => Color::new(1.0, 0.2, 0.8, 1.0),
            10 => Color::new(1.0, 0.0, 0.5, 1.0),
            9 => Color::new(1.0, 0.2, 0.2, 1.0),
            8 => Color::new(1.0, 0.4, 0.0, 1.0),
            7 => Color::new(1.0, 0.6, 0.0, 1.0),
            6 => Color::new(1.0, 1.0, 0.0, 1.0),
            5 => Color::new(0.5, 1.0, 0.0, 1.0),
            4 => Color::new(0.0, 1.0, 0.5, 1.0),
            3 => Color::new(0.0, 1.0, 1.0, 1.0),
            2 => Color::new(0.3, 0.5, 0.7, 1.0),
            _ => Color::new(0.5, 0.5, 0.5, 1.0),
        }
    }
    
    fn recenter_on_word(&mut self, word_text: &str) {
        if let Some(word) = self.words.iter().find(|w| w.text == word_text) {
            let px = word.x;
            let py = word.y;
            
            let mut points: Vec<(f32, f32)> = self.words.iter()
                .map(|w| (w.x, w.y))
                .collect();
            
            self.poincare.recenter(px, py, &mut points);
            
            for (word, (new_x, new_y)) in self.words.iter_mut().zip(points.iter()) {
                word.x = *new_x;
                word.y = *new_y;
                let new_r = (word.x * word.x + word.y * word.y).sqrt();
                word.base_size = self.poincare.radius_to_size(new_r) * 0.5;
                word.render_priority = (1.0 - word.freq_rank) * 0.7 + (1.0 - new_r) * 0.3;
            }
            
            self.focus_word = Some(word_text.to_string());
            self.camera_x = 0.0;
            self.camera_y = 0.0;
            println!("[HYPERBOLIC] Recentered on: {}", word_text);
        }
    }
    
    fn compute_semantic_cells(&mut self) {
        self.semantic_cells.clear();
        let mut renderable_words: Vec<_> = self.words.iter().collect();
        
        if renderable_words.is_empty() {
            return;
        }
        
        renderable_words.sort_by(|a, b| b.render_priority.partial_cmp(&a.render_priority).unwrap());
        let max_cells = self.render_threshold as usize;
        renderable_words.truncate(max_cells);
        
        println!("[CELLS] Computing cells for {} words (top {})", 
            renderable_words.len(), max_cells);
        
        let word_index: HashMap<&str, &Word> = self.words.iter()
            .map(|w| (w.text.as_str(), w))
            .collect();
        
        for word in &renderable_words {
            let context = vec![word.text.clone()];
            let neighbors = self.ngram_trie.get_candidates(&context, 6);
            
            if neighbors.is_empty() {
                continue;
            }
            
            let mut corners = Vec::new();
            for (neighbor_text, _, _, _) in neighbors.iter().take(6) {
                if let Some(&neighbor_word) = word_index.get(neighbor_text.as_str()) {
                    let mid_x = (word.x + neighbor_word.x) / 2.0;
                    let mid_y = (word.y + neighbor_word.y) / 2.0;
                    corners.push((mid_x, mid_y));
                }
            }
            
            if corners.len() < 3 {
                continue;
            }
            
            let cx = word.x;
            let cy = word.y;
            corners.sort_by(|a, b| {
                let angle_a = (a.1 - cy).atan2(a.0 - cx);
                let angle_b = (b.1 - cy).atan2(b.0 - cx);
                angle_a.partial_cmp(&angle_b).unwrap()
            });
            
            let hue = (word.freq_rank * 360.0) % 360.0;
            let sat = 0.6;
            let val = 0.3 + word.activation * 0.4;
            let color = Self::hsv_to_rgb(hue, sat, val);
            
            let connection_count = neighbors.len();
            let base_alpha = if connection_count > 0 {
                (0.90 / connection_count as f32).max(0.08)
            } else {
                0.5
            };
            
            let transparent_color = Color::new(color.r, color.g, color.b, base_alpha);
            
            self.semantic_cells.push(SemanticCell {
                center_word: word.text.clone(),
                corners,
                color: transparent_color,
                activation: word.activation,
            });
        }
        
        println!("[CELLS] Created {} semantic cells", self.semantic_cells.len());
    }

    fn generate_dual_topology(&mut self, seed: Vec<String>, max_tokens: usize) -> Vec<String> {
        println!("\n");
        if self.use_embeddings {
            println!("   [DUAL] TOPOLOGY GENERATION");
        } else {
            println!("   [NGRAM] GENERATION");
        }
        println!("Seed: {:?}\n", seed);
        
        let mut output = seed.clone();
        self.generation_steps.clear();
        self.ngram_orders_used.clear();
        
        let min_tokens = 150;
        let max_entropy = 3.5;
        let min_confidence = 0.10;
        let min_coherence = 0.16;
        
        for step in 0..max_tokens {
            println!(" Step {}", step);
            let context_display: Vec<_> = output.iter().rev().take(5).rev().cloned().collect();
            println!(" Context: {:?}", context_display);
            
            let ngram_candidates = self.ngram_trie.get_candidates(&output, 150);
            
            if ngram_candidates.is_empty() {
                println!("  No candidates\n");
                break;
            }
            
            let scored = if self.use_embeddings {
                let semantic_center = self.embeddings.get_semantic_center(&output);
                self.score_dual_topology(&ngram_candidates, &semantic_center, &output)
            } else {
                self.score_ngram_only(&ngram_candidates, &output)
            };
            
            if scored.is_empty() {
                break;
            }
            
            println!(" Top 3:");
            for (i, c) in scored.iter().take(3).enumerate() {
                if self.use_embeddings {
                    println!("   {}. '{}' [{}g] {:.3} (F:{:.2} C:{:.2} D:{:.2})",
                        i + 1, c.word, c.ngram_order, c.total,
                        c.fluency, c.coherence, c.diversity);
                } else {
                    println!("   {}. '{}' [{}g] {:.3} (F:{:.2} D:{:.2})",
                        i + 1, c.word, c.ngram_order, c.total,
                        c.fluency, c.diversity);
                }
            }
            
            let best = &scored[0];
            println!("  Selected: '{}'", best.word);
            
            let entropy = Self::compute_entropy_with_candidates(&scored);
            let structural_entropy = if self.generation_steps.len() >= 20 {
                self.compute_structural_entropy(&self.generation_steps)
            } else {
                1.0
            };
            
            let gen_step = GenerationStep {
                token: best.word.clone(),
                fluency: best.fluency,
                coherence: best.coherence,
                diversity: best.diversity,
                confidence: best.confidence,
                total: best.total,
                ngram_order: best.ngram_order,
                entropy,
                structural_entropy,
                from_dict: best.is_from_dict,
            };
            self.generation_steps.push(gen_step);
            self.ngram_orders_used.push(best.ngram_order);
            
            output.push(best.word.clone());
            
            if output.len() >= seed.len() + min_tokens {
                if let Some(reason) = self.should_stop(max_entropy, min_confidence, min_coherence) {
                    println!("\n  STOPPING: {}", reason);
                    println!("\n");
                    self.print_summary();
                    return output;
                }
            }
            
            println!("\n");
        }
        
        println!("\nGENERATION COMPLETE\n");
        self.print_summary();
        output
    }
    
    fn score_dual_topology(
        &self,
        candidates: &[(String, f32, usize, usize)],
        semantic_center: &[f32],
        recent: &[String],
    ) -> Vec<ScoredCandidate> {
        let mut scored = Vec::new();
        
        for (word, fluency, order, _count) in candidates {
            let coherence = if let Some(emb) = self.embeddings.embeddings.get(word) {
                let sim = Self::cosine_similarity(semantic_center, emb);
                (sim + 1.0) / 2.0
            } else {
                0.5
            };
            
            let diversity = self.compute_diversity(word, recent);
            let confidence = Self::ngram_order_to_confidence(*order);
            
            let total = 
                fluency * self.weights.fluency +
                coherence * self.weights.coherence +
                diversity * self.weights.diversity +
                confidence * self.weights.confidence;
            
            scored.push(ScoredCandidate {
                word: word.clone(),
                fluency: *fluency,
                coherence,
                diversity,
                confidence,
                total,
                ngram_order: *order,
                is_from_dict: self.dict_corpus_loaded && self.is_likely_definition(word),
            });
        }
        
        scored.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap());
        scored
    }
    
    fn score_ngram_only(
        &self,
        candidates: &[(String, f32, usize, usize)],
        recent: &[String],
    ) -> Vec<ScoredCandidate> {
        let mut scored = Vec::new();
        
        for (word, fluency, order, _count) in candidates {
            let diversity = self.compute_diversity(word, recent);
            let confidence = Self::ngram_order_to_confidence(*order);
            let total = fluency * 0.6 + diversity * 0.3 + confidence * 0.1;
            
            scored.push(ScoredCandidate {
                word: word.clone(),
                fluency: *fluency,
                coherence: 0.5,
                diversity,
                confidence,
                total,
                ngram_order: *order,
                is_from_dict: self.dict_corpus_loaded && self.is_likely_definition(word),
            });
        }
        
        scored.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap());
        scored
    }
    
    fn compute_diversity(&self, word: &str, recent: &[String]) -> f32 {
        let window = 15;
        let recent_window: Vec<_> = recent.iter().rev().take(window).collect();
        
        let mut penalty = 0.0;
        for (i, past_word) in recent_window.iter().enumerate() {
            if past_word.as_str() == word {
                let distance_factor = (window - i) as f32 / window as f32;
                penalty += distance_factor;
            }
        }
        
        (1.0 - penalty.min(1.0)).max(0.0)
    }
    
    fn is_likely_definition(&self, word: &str) -> bool {
        word == ":" || word == "(" || 
        word == "noun" || word == "verb" || 
        word == "adjective" || word == "adverb"
    }
    
    fn should_stop(&self, max_entropy: f32, min_conf: f32, min_coh: f32) -> Option<String> {
        if self.generation_steps.len() < 5 {
            return None;
        }
        
        let recent = 5;
        let start = self.generation_steps.len().saturating_sub(recent);
        let steps = &self.generation_steps[start..];
        
        let avg_conf = steps.iter().map(|s| s.confidence).sum::<f32>() / steps.len() as f32;
        if avg_conf < min_conf {
            return Some(format!("Low confidence ({:.3})", avg_conf));
        }
        
        let avg_ent = steps.iter().map(|s| s.entropy).sum::<f32>() / steps.len() as f32;
        if avg_ent > max_entropy {
            return Some(format!("High entropy ({:.3})", avg_ent));
        }
        
        if self.use_embeddings {
            let avg_coh = steps.iter().map(|s| s.coherence).sum::<f32>() / steps.len() as f32;
            if avg_coh < min_coh {
                return Some(format!("Low coherence ({:.3})", avg_coh));
            }
        }
        
        None
    }
    
    fn print_summary(&self) {
        println!("\nGENERATION SUMMARY");
        println!("==================");
        
        let steps = &self.generation_steps;
        if steps.is_empty() {
            return;
        }
        
        let avg_flu = steps.iter().map(|s| s.fluency).sum::<f32>() / steps.len() as f32;
        let avg_coh = steps.iter().map(|s| s.coherence).sum::<f32>() / steps.len() as f32;
        let avg_div = steps.iter().map(|s| s.diversity).sum::<f32>() / steps.len() as f32;
        let avg_ent = steps.iter().map(|s| s.entropy).sum::<f32>() / steps.len() as f32;
        
        println!("Steps: {}", steps.len());
        println!("Avg Fluency:   {:.3}", avg_flu);
        if self.use_embeddings {
            println!("Avg Coherence: {:.3}", avg_coh);
        }
        println!("Avg Diversity: {:.3}", avg_div);
        println!("Avg Entropy:   {:.3}", avg_ent);
        
        println!("\nN-gram order usage:");
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &n in &self.ngram_orders_used {
            *counts.entry(n).or_insert(0) += 1;
        }
        for n in 2..=12 {
            if let Some(&count) = counts.get(&n) {
                let pct = (count as f32 / self.ngram_orders_used.len() as f32) * 100.0;
                println!("  {}-gram: {} ({:.1}%)", n, count, pct);
            }
        }
        println!();
    }
    
    fn compute_entropy_with_candidates(candidates: &[ScoredCandidate]) -> f32 {
        let sum: f32 = candidates.iter().map(|c| c.total).sum();
        if sum == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for candidate in candidates {
            if candidate.total > 0.0 {
                let p_norm = candidate.total / sum;
                entropy -= p_norm * p_norm.log2();
            }
        }
        entropy
    }
    
    fn compute_structural_entropy(&self, recent_steps: &[GenerationStep]) -> f32 {
        let tokens: Vec<_> = recent_steps.iter().map(|s| s.token.as_str()).collect();
        
        let mut sentences = Vec::new();
        let mut current_sentence = Vec::new();
        
        for token in &tokens {
            current_sentence.push(*token);
            if *token == "." || *token == "!" || *token == "?" {
                if current_sentence.len() > 1 {
                    sentences.push(current_sentence.join(" "));
                }
                current_sentence.clear();
            }
        }
        
        if sentences.len() < 2 {
            return 1.0;
        }
        
        let mut total_similarity = 0.0;
        let mut comparisons = 0;
        
        for i in 0..sentences.len() {
            for j in (i+1)..sentences.len() {
                let s1: std::collections::HashSet<_> = sentences[i].split_whitespace().collect();
                let s2: std::collections::HashSet<_> = sentences[j].split_whitespace().collect();
                
                let intersection = s1.intersection(&s2).count() as f32;
                let union = s1.union(&s2).count() as f32;
                
                if union > 0.0 {
                    total_similarity += intersection / union;
                    comparisons += 1;
                }
            }
        }
        
        if comparisons == 0 {
            return 1.0;
        }
        
        let avg_similarity = total_similarity / comparisons as f32;
        1.0 - avg_similarity
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }
    
    fn compute_bridge_words(&mut self) {
        self.bridge_words.clear();
        
        if !self.use_embeddings || self.words.len() < 2 {
            return;
        }
        
        println!("\n[BRIDGE] Computing semantic bridges...");
        
        for i in 0..(self.words.len() - 1) {
            let word1 = &self.words[i];
            let word2 = &self.words[i + 1];
            
            if word1.embedding.is_none() || word2.embedding.is_none() {
                continue;
            }
            
            let emb1 = word1.embedding.as_ref().unwrap();
            let emb2 = word2.embedding.as_ref().unwrap();
            
            let mut center = vec![0.0; self.embeddings.dim];
            for j in 0..self.embeddings.dim {
                center[j] = (emb1[j] + emb2[j]) / 2.0;
            }
            
            let context = vec![word1.text.clone(), word2.text.clone()];
            let candidates = self.ngram_trie.get_candidates(&context, 20);
            let scored = self.score_dual_topology(&candidates, &center, &context);
            
            if let Some(best) = scored.first() {
                if best.coherence > 0.89 {
                    let mid_x = (word1.x + word2.x) / 2.0;
                    let mid_y = (word1.y + word2.y) / 2.0;
                    
                    self.bridge_words.push(BridgeWord {
                        text: best.word.clone(),
                        x: mid_x,
                        y: mid_y,
                        size: 2.0 + best.coherence * 2.0,
                        ngram_order: best.ngram_order,
                        coherence: best.coherence,
                        from_word: word1.text.clone(),
                        to_word: word2.text.clone(),
                    });
                }
            }
        }
        
        println!("[BRIDGE] Created {} bridges", self.bridge_words.len());
    }
    fn update_activations(&mut self) {
        self.frame_counter = (self.frame_counter + 1) % 60;
        let do_decay = self.frame_counter == 0 || self.generating || self.exploring;    
        self.activation_by_word.clear();
        let mut max_act = 0.0;
        
        if self.redraw_needed && self.semantic_mode {
            println!("[UPDATE] Recomputing semantic cells...");
            self.compute_semantic_cells();
            self.redraw_needed = false;
        }

        let camera_changed = (self.camera_x - self.last_camera.0).abs() > 100.0
            || (self.camera_y - self.last_camera.1).abs() > 100.0
            || (self.camera_zoom - self.last_camera.2).abs() > 0.2;

        if self.semantic_mode && camera_changed {
            println!("[CAMERA] Significant movement detected, recomputing cells...");
            self.compute_semantic_cells();
            self.last_camera = (self.camera_x, self.camera_y, self.camera_zoom);
        }
        
        for w in &mut self.words {
            if do_decay { w.activation *= 0.95; }
            if w.activation > max_act { max_act = w.activation; }
            self.activation_by_word.insert(w.text.clone(), w.activation);
        }
        self.any_active = max_act > 0.02;
        
        if let Some(selected) = &self.selected {
            if let Some(word) = self.words.iter_mut().find(|w| &w.text == selected) {
                word.activation = 1.0;
            }
            
            let context = vec![selected.clone()];
            let related = self.ngram_trie.get_candidates(&context, 50);
            
            if !related.is_empty() {
                let max_count = *related.iter().map(|(_, _, _, count)| count).max().unwrap_or(&1);
                
                let mut activation_updates: Vec<(String, f32)> = Vec::new();
                for (related_word, _, _, count) in related {
                    activation_updates.push((related_word.clone(), count as f32 / max_count as f32));
                }
                
                for (word_text, new_activation) in activation_updates {
                    if let Some(word) = self.words.iter_mut().find(|w| w.text == word_text) {
                        word.activation = word.activation.max(new_activation);
                    }
                }
            }
        }
    }
            
    fn update_expanded_words(&mut self) {
        self.expanded_words.clear();
        
        if let Some(selected) = &self.selected {
            if let Some(selected_word) = self.words.iter().find(|w| &w.text == selected) {
                let context = if self.context_path.is_empty() {
                    vec![selected.clone()]
                } else {
                    let mut c = self.context_path.clone();
                    c.push(selected.clone());
                    c
                };
                
                let candidates = self.ngram_trie.get_candidates(&context, 8);
                
                let semantic_center = if self.use_embeddings {
                    self.embeddings.get_semantic_center(&context)
                } else {
                    vec![]
                };
                
                let scored = if self.use_embeddings {
                    self.score_dual_topology(&candidates, &semantic_center, &context)
                } else {
                    self.score_ngram_only(&candidates, &context)
                };
                
                for (i, candidate) in scored.iter().take(8).enumerate() {
                    let angle = (i as f32 / 8.0) * std::f32::consts::TAU;
                    let base_offset = 0.15;
                    let offset_x = base_offset * angle.cos();
                    let offset_y = base_offset * angle.sin();
                    
                    self.expanded_words.push(ExpandedWord {
                        text: candidate.word.clone(),
                        x: selected_word.x + offset_x,
                        y: selected_word.y + offset_y,
                        size: 3.0 + candidate.total * 2.0,
                        ngram_order: candidate.ngram_order,
                        coherence: candidate.coherence,
                    });
                }
            }
        }
    }
    
    fn screen_to_world(&self, x: f32, y: f32) -> (f32, f32) {
        let world_x = (x - screen_width() / 2.0 - self.camera_x) / self.camera_zoom / self.poincare.radius;
        let world_y = (y - screen_height() / 2.0 - self.camera_y) / self.camera_zoom / self.poincare.radius;
        (world_x, world_y)
    }
    
    fn find_word_at(&self, x: f32, y: f32) -> Option<String> {
        let (world_x, world_y) = self.screen_to_world(x, y);
        
        for word in &self.expanded_words {
            let dx = world_x - word.x;
            let dy = world_y - word.y;
            if dx*dx + dy*dy < (word.size * 0.01) * (word.size * 0.01) {
                return Some(word.text.clone());
            }
        }
        
        for word in &self.words {
            let should_render = if self.generating {
                word.activation > 0.05 || word.in_context || Some(&word.text) == self.selected.as_ref()
            } else {
                word.in_context
                    || Some(&word.text) == self.selected.as_ref()
                    || self.expanded_words.iter().any(|e| e.text == word.text)
            };
            
            if !should_render {
                continue;
            }
            
            let dx = world_x - word.x;
            let dy = world_y - word.y;
            
            let zoom_scaled_activation = word.activation * self.camera_zoom.powf(0.5);
            let size_mult = if word.in_context { 3.0 } else { 1.0 };
            let base_pixel_size = word.base_size * self.camera_zoom * 2.0;
            let size = base_pixel_size * (1.0 + zoom_scaled_activation * 2.0) * size_mult;
            
            let click_radius = (size / self.camera_zoom / self.poincare.radius) * 0.5;
            
            if dx*dx + dy*dy < click_radius * click_radius {
                return Some(word.text.clone());
            }
        }
        None
    }
    
    fn select_word(&mut self, word: String) {
        if Some(&word) == self.selected.as_ref() {
            self.context_path.push(word.clone());
            if let Some(w) = self.words.iter_mut().find(|w| w.text == word) {
                w.in_context = true;
            }
            self.selected = Some(word);
            self.update_expanded_words();
        } else {
            let is_expanded = self.expanded_words.iter().any(|w| w.text == word);
            
            if is_expanded {
                self.context_path.push(word.clone());
                self.selected = Some(word);
                self.update_expanded_words();
            } else {
                self.selected = Some(word);
                self.update_expanded_words();
            }
        }
    }
    
    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Escape) {
            if self.text_input_active {
                self.text_input_active = false;
                self.query_text.clear();
                return;
            } else if self.config_mode {
                self.config_mode = false;
                self.config_input = self.max_ngram_order.to_string();
                return;
            } else {
                self.selected = None;
                self.expanded_words.clear();
                self.context_path.clear();
                self.focus_word = None;
                for word in &mut self.words {
                    word.in_context = false;
                }
                return;
            }
        }
        
        if is_key_pressed(KeyCode::Space) 
            && self.selected.is_some() 
            && !self.text_input_active 
            && !self.config_mode {
            let selected = self.selected.clone().unwrap();
            self.recenter_on_word(&selected);
            return;
        }
        
        if self.config_mode {
            if let Some(character) = get_char_pressed() {
                if character.is_numeric() {
                    if self.config_input.len() < 2 {
                        self.config_input.push(character);
                    }
                }
            }
            
            if is_key_pressed(KeyCode::Backspace) {
                self.config_input.pop();
            }
            
            if is_key_pressed(KeyCode::Enter) {
                self.config_mode = false;
                if !self.config_input.is_empty() {
                    if let Ok(value) = self.config_input.parse::<usize>() {
                        if value >= 2 && value <= 12 {
                            self.max_ngram_order = value;
                            self.ngram_trie.max_order = value;
                            println!("\n[CFG] N-gram order set to: {}", value);
                        }
                    }
                }
                self.config_input = self.max_ngram_order.to_string();
            }
            
            return;
        }
        
        if self.text_input_active {
            if let Some(character) = get_char_pressed() {
                if character.is_alphanumeric() || character == ' ' 
                    || "(),.!?;".contains(character) {
                    self.query_text.push(character);
                }
            }
            
            if is_key_pressed(KeyCode::Backspace) {
                self.query_text.pop();
            }
            
            if is_key_pressed(KeyCode::Enter) && !self.query_text.is_empty() {
                self.process_query();
            }
            
            return;
        }
        
        if is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl) {
            if is_key_pressed(KeyCode::L) {
                if let Some(path) = FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .set_title("Load Conversation File")
                    .pick_file()
                {
                    if let Some(path_str) = path.to_str() {
                        self.load_file(path_str);
                    }
                }
            }
            
            if is_key_pressed(KeyCode::E) {
                if let Some(path) = FileDialog::new()
                    .add_filter("TXT", &["txt"])
                    .set_title("Load GloVe Embeddings")
                    .pick_file()
                {
                    if let Some(path_str) = path.to_str() {
                        self.load_embeddings(path_str);
                    }
                }
                self.redraw_needed = true;
            }
            
            if is_key_pressed(KeyCode::T) {
                self.use_embeddings = !self.use_embeddings;
                println!("\n Topology mode: {}", 
                    if self.use_embeddings { "DUAL (n-grams + embeddings)" } 
                    else { "N-GRAMS ONLY" });
                
                if self.use_embeddings {
                    self.compute_bridge_words();
                } else {
                    self.bridge_words.clear();
                }
                self.redraw_needed = true;
            }
            
            if is_key_pressed(KeyCode::S) {
                self.semantic_mode = !self.semantic_mode;
                println!("\n Visualization mode: {}", 
                    if self.semantic_mode { "SEMANTIC CELLS" } 
                    else { "NODE MODE" });
                
                if self.semantic_mode && self.semantic_cells.is_empty() {
                    println!("[CELLS] Computing semantic cells on demand...");
                    self.compute_semantic_cells();
                }
                self.redraw_needed = true;
            }
            
            if is_key_pressed(KeyCode::O) {
                self.text_input_active = !self.text_input_active;
                if !self.text_input_active {
                    self.query_text.clear();
                }
                return;
            }
            
            if is_key_pressed(KeyCode::N) && !self.generating {
                self.config_mode = !self.config_mode;
                if self.config_mode {
                    self.config_input = self.max_ngram_order.to_string();
                } else {
                    if !self.config_input.is_empty() {
                        if let Ok(value) = self.config_input.parse::<usize>() {
                            if value >= 2 && value <= 12 {
                                self.max_ngram_order = value;
                                self.ngram_trie.max_order = value;
                                println!("\n[CFG] N-gram order set to: {}", value);
                            }
                        }
                    }
                    self.config_input = self.max_ngram_order.to_string();
                }
                self.redraw_needed = true;
                return;
            }
        }
        
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            let factor = if wheel > 0.0 { 1.1 } else { 0.9 };
            self.camera_zoom = (self.camera_zoom * factor).clamp(0.1, 5.0);
        }
        
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            if let Some(word) = self.find_word_at(mx, my) {
                self.select_word(word);
            } else {
                self.last_mouse = (mx, my);
                self.dragging = true;
            }
        }
        
        if is_mouse_button_released(MouseButton::Left) {
            self.dragging = false;
        }
        
        if self.dragging {
            let (mx, my) = mouse_position();
            self.camera_x += mx - self.last_mouse.0;
            self.camera_y += my - self.last_mouse.1;
            self.last_mouse = (mx, my);
        }
        
        if is_key_pressed(KeyCode::R) {
            self.camera_x = 0.0;
            self.camera_y = 0.0;
            self.camera_zoom = 1.0;
            self.focus_word = None;
        }
        
        if is_key_pressed(KeyCode::C) {
            self.context_path.clear();
            for word in &mut self.words {
                word.in_context = false;
            }
        }
    }
    
    fn process_query(&mut self) {
        let query_words: Vec<String> = self.query_text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        if !query_words.is_empty() {
            self.exploring = true;
            
            if !self.semantic_mode {
                self.semantic_mode = true;
                println!("\n[QUERY] Enabling semantic visualization mode");
                self.redraw_needed = true;
            }
            
            if self.semantic_cells.is_empty() {
                println!("[CELLS] Computing semantic cells for query...");
                self.compute_semantic_cells();
            }
            
            let best_path = self.generate_dual_topology(query_words.clone(), 300);
            
            let seed_len = query_words.len();
            let output_path = if best_path.len() > seed_len {
                best_path[seed_len..].to_vec()
            } else {
                best_path
            };
            
            self.best_path = output_path;
            self.playback_index = 0;
            self.generated_output = query_words.join(" ");
            self.generating = true;
            self.exploring = false;
            self.generation_timer = 0.0;
        }
        
        self.text_input_active = false;
        self.query_text.clear();
    }
    
    fn update_generation(&mut self, delta_time: f32) {
        if !self.generating {
            return;
        }
        
        self.generation_timer += delta_time;
        
        if self.generation_timer >= self.generation_delay {
            self.generation_timer = 0.0;
            
            if self.playback_index < self.best_path.len() {
                let current_word = &self.best_path[self.playback_index];
                
                if self.playback_index > 0 {
                    let is_punct = current_word == "." || current_word == "," || 
                                   current_word == "!" || current_word == "?" || 
                                   current_word == ";" || current_word == ":";
                    
                    if !is_punct {
                        self.generated_output.push(' ');
                    }
                }
                self.generated_output.push_str(current_word);
                
                if current_word == "." || current_word == "!" || current_word == "?" {
                    self.generated_output.push('\n');
                }
                
                for word in &mut self.words {
                    word.activation *= 0.7;
                }
                
                if let Some(word) = self.words.iter_mut().find(|w| &w.text == current_word) {
                    word.activation = 1.0;
                }
                
                self.playback_index += 1;
                
                if self.playback_index >= self.best_path.len() {
                    self.generating = false;
                    println!("\n=== FINAL OUTPUT ===\n{}\n====================\n", self.generated_output);
                }
            }
        }
    }
    
    fn draw(&self) {
        if self.generating || self.exploring || self.any_active || self.redraw_needed || self.config_mode {
            clear_background(GREY_BG);
        }
    
        let center_x = screen_width() / 2.0 + self.camera_x;
        let center_y = screen_height() / 2.0 + self.camera_y;
        
        let boundary_radius = self.poincare.radius * self.camera_zoom;
        draw_circle_lines(center_x, center_y, boundary_radius, 2.0, 
            Color::new(0.3, 0.3, 0.35, 0.6));
        
        for i in 1..5 {
            let r = (i as f32 / 5.0) * boundary_radius;
            draw_circle_lines(center_x, center_y, r, 1.0, 
                Color::new(0.2, 0.2, 0.25, 0.4));
        }
        
        if self.semantic_mode {
            for (cell_idx, cell) in self.semantic_cells.iter().enumerate() {
                if cell.corners.len() < 3 {
                    continue;
                }
                
                let frame_age = (self.frame_counter as i32 - (cell_idx % 60) as i32).abs() as f32;
                let fade_factor = (1.0 - (frame_age / 120.0)).max(0.0);
                
                let should_draw_this_frame = cell_idx % 60 == (self.frame_counter as usize % 60);
                
                let center_activation = self.activation_by_word
                    .get(&cell.center_word)
                    .copied()
                    .unwrap_or(0.0);
                
                let should_render = if self.generating || self.exploring {
                    center_activation > 0.01 || should_draw_this_frame
                } else {
                    center_activation > 0.05 || should_draw_this_frame
                };
                
                if !should_render || fade_factor < 0.01 {
                    continue;
                }
                
                let screen_corners: Vec<_> = cell.corners.iter()
                    .map(|(x, y)| {
                        (
                            center_x + x * self.poincare.radius * self.camera_zoom,
                            center_y + y * self.poincare.radius * self.camera_zoom,
                        )
                    })
                    .collect();
                
                if screen_corners.len() < 3 {
                    continue;
                }
                
                let brightness_boost = center_activation * 0.3;
                let boosted_color = Color::new(
                    (cell.color.r + brightness_boost).min(1.0),
                    (cell.color.g + brightness_boost).min(1.0),
                    (cell.color.b + brightness_boost).min(1.0),
                    ((cell.color.a + center_activation * 0.1).min(0.3)) * fade_factor,
                );
                
                let v0 = Vec2::new(screen_corners[0].0, screen_corners[0].1);
                for i in 1..screen_corners.len() - 1 {
                    draw_triangle(
                        v0,
                        Vec2::new(screen_corners[i].0, screen_corners[i].1),
                        Vec2::new(screen_corners[i + 1].0, screen_corners[i + 1].1),
                        boosted_color,
                    );
                }
                
                for i in 0..screen_corners.len() {
                    let next = (i + 1) % screen_corners.len();
                    let base_alpha = (cell.activation as f32 / 6.0 * 0.5).max(0.08);
                    let border_alpha = (base_alpha + center_activation * 0.3) * fade_factor;
                    draw_line(
                        screen_corners[i].0, screen_corners[i].1,
                        screen_corners[next].0, screen_corners[next].1,
                        1.5 + center_activation * 1.5,
                        Color::new(
                            (cell.color.r + brightness_boost * 0.5).min(1.0),
                            (cell.color.g + brightness_boost * 0.5).min(1.0),
                            (cell.color.b + brightness_boost * 0.5).min(1.0),
                            border_alpha.min(1.0)
                        ),
                    );
                }
            }
        }

        if let Some(selected) = &self.selected {
            if let Some(selected_word) = self.words.iter().find(|w| &w.text == selected) {
                let sel_x = center_x + selected_word.x * self.poincare.radius * self.camera_zoom;
                let sel_y = center_y + selected_word.y * self.poincare.radius * self.camera_zoom;
                
                for word in &self.expanded_words {
                    let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
                    let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
                    
                    let line_color = Self::ngram_order_to_color(word.ngram_order);
                    
                    draw_line(sel_x, sel_y, screen_x, screen_y, 2.0, 
                        Color::new(line_color.r, line_color.g, line_color.b, 0.6));
                }
            }
        }
        
        let mut renderable_words: Vec<&Word> = self.words.iter()
            .filter(|word| {
                if self.semantic_mode {
                    if self.generating {
                        word.activation > 0.05
                            || word.in_context
                            || Some(&word.text) == self.selected.as_ref()
                    } else {
                        word.in_context
                            || Some(&word.text) == self.selected.as_ref()
                            || self.expanded_words.iter().any(|e| e.text == word.text)
                    }
                } else {
                    word.render_priority > 0.5
                        || word.activation > 0.05 
                        || word.in_context 
                        || Some(&word.text) == self.selected.as_ref()
                }
            })
            .collect();
        
        if !self.semantic_mode && renderable_words.len() > self.render_threshold as usize {
            renderable_words.sort_by(|a, b| b.render_priority.partial_cmp(&a.render_priority).unwrap());
            renderable_words.truncate(self.render_threshold as usize);
        }
        
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let sigmoid_render_prob = |ngram_order: usize| -> f32 {
            let x = ngram_order as f32;
            let k = 0.50;
            let x0 = 6.5;
            1.0 / (1.0 + (k * (x - x0)).exp())
        };
        
        let should_render_word = |word_text: &str, ngram_order: usize| -> bool {
            if word_text == self.selected.as_ref().map(|s| s.as_str()).unwrap_or("") {
                return true;
            }
            
            let prob = sigmoid_render_prob(ngram_order);
            let mut hasher = DefaultHasher::new();
            word_text.hash(&mut hasher);
            let hash_val = hasher.finish();
            let random_val = (hash_val % 1000) as f32 / 200.0;
            random_val < prob
        };
        
        let mut ngram_cache: HashMap<&str, (usize, f32)> = HashMap::new();
        
        for word in &renderable_words {
            if Some(&word.text) != self.selected.as_ref() && !word.in_context {
                let context = vec![word.text.clone()];
                let neighbors = self.ngram_trie.get_candidates(&context, 50);
                
                if !neighbors.is_empty() {
                    let max_order = neighbors.iter().map(|(_, _, order, _)| order).max().unwrap_or(&2);
                    
                    if !word.in_context && word.activation < 0.05 {
                        if !should_render_word(&word.text, *max_order) {
                            continue;
                        }
                    }
                    
                    let strength = (*max_order as f32 / 12.0);
                    ngram_cache.insert(&word.text, (*max_order, strength));
                } else {
                    ngram_cache.insert(&word.text, (2, 0.0));
                }
            }
        }
        
        renderable_words.sort_by(|a, b| {
            let zoom_scaled_a = a.activation * self.camera_zoom.powf(0.5);
            let zoom_scaled_b = b.activation * self.camera_zoom.powf(0.5);
            
            let conf_a = if let Some((_, strength)) = ngram_cache.get(a.text.as_str()) {
                (zoom_scaled_a * 0.5 + strength * 0.5).min(1.0)
            } else {
                zoom_scaled_a
            };
            
            let conf_b = if let Some((_, strength)) = ngram_cache.get(b.text.as_str()) {
                (zoom_scaled_b * 0.5 + strength * 0.5).min(1.0)
            } else {
                zoom_scaled_b
            };
            
            conf_a.partial_cmp(&conf_b).unwrap()
        });
        
        for word in renderable_words {
            if !word.in_context 
                && Some(&word.text) != self.selected.as_ref() 
                && word.activation < 0.05
                && !ngram_cache.contains_key(word.text.as_str()) {
                continue;
            }
            
            let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
            
            if screen_x < -100.0 || screen_x > screen_width() + 100.0 ||
               screen_y < -100.0 || screen_y > screen_height() + 100.0 {
                continue;
            }
            
            let base_pixel_size = word.base_size * self.camera_zoom * 2.0;
            let zoom_scaled_activation = word.activation * self.camera_zoom.powf(0.5);
            
            if !self.semantic_mode && word.activation < 0.05 && Some(&word.text) != self.selected.as_ref() && !word.in_context {
                draw_circle(screen_x, screen_y, base_pixel_size * 0.4, 
                    Color::new(0.5, 0.5, 0.5, 0.3));
                continue;
            }
            
            let size_mult = if word.in_context { 2.0 } else { 1.0 };
            let size = base_pixel_size * (1.0 + zoom_scaled_activation * 2.0) * size_mult;
            
            let (color, glow_color) = if Some(&word.text) == self.selected.as_ref() {
                (CYAN, CYAN)
            } else if word.in_context {
                (YELLOW, YELLOW)
            } else {
                let base_color = Self::confidence_to_hsv_color(
                    word.ngram_confidence, 
                    (word.x, word.y)
                );
                
                let blended = Color::new(
                    base_color.r * (0.5 + zoom_scaled_activation * 0.5),
                    base_color.g * (0.5 + zoom_scaled_activation * 0.5),
                    base_color.b * (0.5 + zoom_scaled_activation * 0.5),
                    0.7 + zoom_scaled_activation * 0.3
                );
                
                let glow = Color::new(blended.r, blended.g, blended.b, 0.1);
                (blended, glow)
            };
            
            if !self.semantic_mode {
                let glow_alpha = zoom_scaled_activation * 0.06;
                draw_circle(screen_x, screen_y, size * 1.8,
                    Color::new(glow_color.r, glow_color.g, glow_color.b, glow_alpha));
                
                draw_circle(screen_x, screen_y, size, color);
            }
            
            let should_show_label = if self.semantic_mode {
                true
            } else {
                (word.activation > 0.3 || word.in_context || Some(&word.text) == self.selected.as_ref())
                    && self.camera_zoom > 0.6
            };
            
            if should_show_label {
                let font_size = if self.semantic_mode {
                    (size * 1.5).max(14.0)
                } else {
                    (size * 1.3).max(16.0)
                };
                
                let label_color = if self.semantic_mode {
                    BLACK
                } else if Some(&word.text) == self.selected.as_ref() {
                    let label_alpha = (self.camera_zoom - 0.6) * 2.5;
                    Color::new(WHITE.r, WHITE.g, WHITE.b, label_alpha.min(1.0))
                } else {
                    let label_alpha = (self.camera_zoom - 0.6) * 2.5;
                    Color::new(0.95, 0.95, 0.95, label_alpha.min(0.9))
                };
                
                draw_text(&word.text, screen_x - size, screen_y + size / 2.0, font_size, label_color);
            }
        }
        
        for bridge in &self.bridge_words {
            let screen_x = center_x + bridge.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + bridge.y * self.poincare.radius * self.camera_zoom;
            let size = 2.5;
            
            let hue = (bridge.coherence * 360.0) % 360.0;
            let sat = 0.6;
            let val = 0.5;
            let bridge_color = Self::hsv_to_rgb(hue, sat, val);
            
            let final_color = Color::new(
                bridge_color.r,
                bridge_color.g,
                bridge_color.b,
                0.4
            );
            
            draw_circle(screen_x, screen_y, size, final_color);
        }
        
        for word in &self.expanded_words {
            let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
            let size = (word.size * self.camera_zoom * 2.5).max(10.0);
            
            let ngram_color = Self::ngram_order_to_color(word.ngram_order);
            
            draw_circle(screen_x, screen_y, size * 1.8, 
                Color::new(ngram_color.r, ngram_color.g, ngram_color.b, 0.3));
            draw_circle(screen_x, screen_y, size, ngram_color);
            
            let label = if self.use_embeddings {
                format!("{}[{}:{:.1}]", word.text, word.ngram_order, word.coherence)
            } else {
                format!("{}[{}]", word.text, word.ngram_order)
            };
            
            let font_size = (size * 1.5).max(16.0);
            draw_text(&label, screen_x - size, screen_y + size / 2.0, font_size, WHITE);
        }
        
        let mode_text = if self.semantic_mode {
            if self.use_embeddings {
                if self.ngram_trie.use_disk {
                    "SEMANTIC CELLS: DUAL TOPOLOGY (N-grams[DISK] + Embeddings)"
                } else {
                    "SEMANTIC CELLS: DUAL TOPOLOGY (N-grams + Embeddings)"
                }
            } else {
                if self.ngram_trie.use_disk {
                    "SEMANTIC CELLS: N-GRAMS ONLY [DISK]"
                } else {
                    "SEMANTIC CELLS: N-GRAMS ONLY"
                }
            }
        } else {
            if self.use_embeddings {
                if self.ngram_trie.use_disk {
                    "NODE MODE: DUAL TOPOLOGY (N-grams[DISK] + Embeddings) | HSV: Position + Avg Confidence"
                } else {
                    "NODE MODE: DUAL TOPOLOGY (N-grams + Embeddings) | HSV: Position + Avg Confidence"
                }
            } else {
                if self.ngram_trie.use_disk {
                    "NODE MODE: N-GRAMS ONLY [DISK] | HSV: Position + Avg Confidence"
                } else {
                    "NODE MODE: N-GRAMS ONLY | HSV: Position + Avg Confidence"
                }
            }
        };
        
        let mode_color = if self.semantic_mode { MAGENTA } else { CYAN };
        draw_text(mode_text, 10.0, 30.0, 20.0, mode_color);
        
        draw_text("Ctrl+L: Load | Ctrl+E: Embeddings | Ctrl+T: Topology | Ctrl+S: Semantic | Ctrl+O: Query | Ctrl+N: Config", 
            10.0, 55.0, 18.0, WHITE);
        
        draw_text("ESC: Cancel | R: Reset | C: Clear | SPACE: Recenter | Mouse: Pan/Zoom", 
            10.0, 75.0, 16.0, Color::new(0.8, 0.8, 0.8, 1.0));
        
        let storage_mode = if self.ngram_trie.use_disk {
            format!("DISK ({} batches)", self.ngram_trie.disk_manager.batch_files.len())
        } else {
            "MEMORY".to_string()
        };
        
        draw_text(&format!("Files: {} | Words: {} | Cells: {} | Bridges: {} | Order: {} | Storage: {}", 
            self.files_loaded,
            self.words.len(),
            self.semantic_cells.len(), 
            self.bridge_words.len(), 
            self.max_ngram_order,
            storage_mode), 
            10.0, 95.0, 16.0, WHITE);
        
        if let Some(focus) = &self.focus_word {
            draw_text(&format!("Focus: {}", focus), 10.0, 115.0, 16.0, ORANGE);
        }
        
        let mut file_y = 135.0;
        let mut sorted_files: Vec<_> = self.loaded_files.values().collect();
        sorted_files.sort_by(|a, b| {
            b.ngram_level.cmp(&a.ngram_level)
                .then_with(|| a.filename.cmp(&b.filename))
        });
        
        for info in sorted_files {
            let display_name = if info.filename.len() > 30 {
                format!("{}...", &info.filename[..27])
            } else {
                info.filename.clone()
            };
            
            let text = format!("  {}G {}", 
                info.ngram_level, display_name);
            
            draw_text(&text, 10.0, file_y, 14.0, WHITE);
            file_y += 18.0;
        }
        
        let base_y = 135.0 + (self.loaded_files.len() as f32 * 18.0);
        
        if let Some(selected) = &self.selected {
            if let Some(word) = self.words.iter().find(|w| &w.text == selected) {
                let info = format!("Selected: {} [Avg: {:.1}g | Max: {}g | Conf: {:.2}]", 
                    selected, word.avg_ngram_order, word.max_ngram_order, word.ngram_confidence);
                draw_text(&info, 10.0, base_y + 5.0, 18.0, CYAN);
            } else {
                draw_text(&format!("Selected: {}", selected), 10.0, base_y + 5.0, 18.0, CYAN);
            }
        }
        
        if !self.context_path.is_empty() {
            let context = self.context_path.join(" → ");
            draw_text(&format!("Context: {}", context), 10.0, base_y + 30.0, 16.0, YELLOW);
        }
        
        if self.config_mode {
            let box_y = screen_height() - 80.0;
            draw_rectangle(0.0, box_y, screen_width(), 80.0, 
                Color::new(0.1, 0.1, 0.1, 0.9));
            
            let prompt = format!("Max N-gram order (2-12): {}_", self.config_input);
            draw_text(&prompt, 10.0, box_y + 30.0, 24.0, ORANGE);
            draw_text("(Affects NEXT file load)", 10.0, box_y + 10.0, 16.0, CYAN);
            draw_text("(Enter to apply | ESC to cancel)", 10.0, box_y + 50.0, 16.0, YELLOW);
        }
        else if self.text_input_active {
            let box_y = screen_height() - 60.0;
            draw_rectangle(0.0, box_y, screen_width(), 60.0, 
                Color::new(0.1, 0.1, 0.1, 0.9));
            
            let prompt = format!("Query: {}_", self.query_text);
            draw_text(&prompt, 10.0, box_y + 30.0, 24.0, CYAN);
            draw_text("(Enter to submit | ESC or Ctrl+O to cancel)", 10.0, box_y + 10.0, 16.0, YELLOW);
        } else if self.exploring {
            let box_y = screen_height() - 60.0;
            draw_rectangle(0.0, box_y, screen_width(), 60.0, 
                Color::new(0.1, 0.1, 0.1, 0.9));
            
            let text = if self.use_embeddings {
                "★ DUAL TOPOLOGY GENERATION..."
            } else {
                "★ N-GRAM GENERATION..."
            };
            draw_text(text, 10.0, box_y + 30.0, 20.0, ORANGE);
        } else if self.generating {
            let box_y = screen_height() - 150.0;
            draw_rectangle(0.0, box_y, screen_width(), 150.0, 
                Color::new(0.1, 0.1, 0.1, 0.9));
            
            draw_text("GENERATING...", 10.0, box_y + 20.0, 20.0, CYAN);
            
            let lines: Vec<&str> = self.generated_output.lines().collect();
            let start_line = lines.len().saturating_sub(3);
            let visible_lines = &lines[start_line..];
            
            let mut y_offset = box_y + 50.0;
            for line in visible_lines {
                let display_line = if line.chars().count() > 120 {
                    let skip_count = line.chars().count().saturating_sub(120);
                    format!("...{}", line.chars().skip(skip_count).collect::<String>())
                } else {
                    line.to_string()
                };
                draw_text(&display_line, 10.0, y_offset, 14.0, WHITE);
                y_offset += 18.0;
            }
            
            draw_text(&format!("Tokens: {} | Lines: {}", self.playback_index, lines.len()), 
                10.0, box_y + 130.0, 14.0, YELLOW);
        } else {
            let hint_y = screen_height() - 30.0;
            let hint_text = if self.semantic_mode {
                "Ctrl+S: Node Mode | SPACE: Recenter | Click cells | ESC: Deselect | Ctrl+O: Query"
            } else {
                "Ctrl+S: Semantic Mode | SPACE: Recenter | Node colors = HSV(position, avg confidence) | ESC: Deselect"
            };
            draw_text(hint_text, 10.0, hint_y, 18.0, Color::new(0.6, 0.6, 0.6, 1.0));
        }
    }
}

#[macroquad::main("M2 - Disk-Batched N-grams")]
async fn main() {
    let mut word_map = WordMap::new();
    
    let embedding_paths = vec![
        "glove.6B.50d.txt",
        "glove.6B.100d.txt", 
        "glove.6B.200d.txt",
        "glove.6B.300d.txt"
    ];
    
    for path in embedding_paths {
        if std::path::Path::new(path).exists() {
            println!("Found embeddings: {}", path);
            word_map.embeddings.dim = if path.contains("50d") { 50 }
                else if path.contains("100d") { 100 }
                else if path.contains("200d") { 200 }
                else { 300 };
            word_map.load_embeddings(path);
            break;
        }
    }
    
    if word_map.files_loaded == 0 {
        println!("\n========================================");
        println!("  M2 - DISK-BATCHED N-GRAMS");
        println!("========================================");
        println!("\n NEW: Automatic disk-backed storage");
        println!(" - Files >1M tokens use disk batching");
        println!(" - Memory limit: ~100MB of n-grams");
        println!(" - Batches saved to temp directory");
        println!(" - Auto cleanup on exit");
        println!("\n Hyperbolic Poincaré disc");
        println!(" Semantic cells (relationship polygons)");
        println!(" HSV color coding based on:");
        println!("   - HUE: Spatial position (angle in disc)");
        println!("   - SATURATION: Avg n-gram confidence");
        println!("   - VALUE: Avg n-gram confidence");
        println!(" Low confidence → gray/muted");
        println!(" High confidence → vibrant colors");
        println!(" Common words → center (large cells)");
        println!(" Rare words → edge (small cells)");
        println!(" SPACE to recenter on any word");
        println!("\nCtrl+L: Load corpus");
        println!("Ctrl+N: Configure n-gram order (2-12)");
        println!("Ctrl+S: Toggle semantic/node mode");
        println!("Ctrl+O: Query generation");
        println!("========================================\n");
    }
    
    loop {
        let delta = get_frame_time();
        
        word_map.handle_input();
        word_map.update_activations();
        word_map.update_generation(delta);
        word_map.draw();
        next_frame().await;
    }
}