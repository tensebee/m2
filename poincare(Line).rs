use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use serde_json::Value;
use macroquad::prelude::*;
use rfd::FileDialog;

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
// HYPERBOLIC GEOMETRY - POINCARÃ‰ DISC
// ============================================================================

struct PoincareDisc {
    radius: f32,
    inner_radius: f32, // Minimum radius to spread out center words
}

impl PoincareDisc {
    fn new(radius: f32, inner_radius: f32) -> Self {
        PoincareDisc { radius, inner_radius }
    }
    
    // Map frequency rank (0 = most common, 1 = rarest) to hyperbolic radius
    fn freq_to_radius(&self, freq_rank: f32) -> f32 {
        // Use exponential mapping to compress toward boundary
        // freq_rank 0.0 -> r = inner_radius (no longer at dead center)
        // freq_rank 1.0 -> r = 0.95 (near boundary, never quite reaching it)
        let max_r = 0.95;
        let compression = 2.5; // Higher = more compression toward edge
        
        let normalized = max_r * (1.0 - (-compression * freq_rank).exp());
        
        // Map from [0, max_r] to [inner_radius, max_r]
        self.inner_radius + normalized * (1.0 - self.inner_radius)
    }
    
    // Convert hyperbolic radius to visual size multiplier
    fn radius_to_size(&self, r: f32) -> f32 {
        // Words get smaller as they approach the boundary
        // This creates the "fish-eye" effect of hyperbolic space
        let boundary_factor = 1.0 - r;
        
        // Smooth curve: large at center, small at edge
        (boundary_factor.powf(0.5) * 5.0 + 0.5).max(0.3)
    }
    
    // Hyperbolic distance between two points in the disc
    fn hyperbolic_distance(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
        let r1_sq = x1 * x1 + y1 * y1;
        let r2_sq = x2 * x2 + y2 * y2;
        
        let numerator = (x1 - x2).powi(2) + (y1 - y2).powi(2);
        let denominator = (1.0 - r1_sq) * (1.0 - r2_sq);
        
        if denominator <= 0.0 {
            return 100.0; // Very large distance for boundary points
        }
        
        (1.0 + 2.0 * numerator / denominator).acosh().max(0.0)
    }
    
    // MÃ¶bius transformation to recenter disc around a point
    fn recenter(&self, px: f32, py: f32, points: &mut [(f32, f32)]) {
        let p_norm_sq = px * px + py * py;
        
        if p_norm_sq < 0.001 {
            return; // Already centered
        }
        
        for (x, y) in points.iter_mut() {
            let old_x = *x;
            let old_y = *y;
            
            // MÃ¶bius transformation formula
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
    x: f32,  // PoincarÃ© disc coordinates [-1, 1]
    y: f32,
    freq: usize,
    freq_rank: f32, // 0.0 = most common, 1.0 = rarest
    base_size: f32,
    activation: f32,
    layer: usize,
    in_context: bool,
    embedding: Option<Vec<f32>>,
    render_priority: f32, // Combined score for rendering decisions
    cached_color: Option<(Color, Color)>, // (base_color, glow_color) - cached to avoid recomputing
}

// Cell structure for Voronoi-style regions
#[derive(Clone)]
struct SemanticCell {
    center_word: String,
    corners: Vec<(f32, f32)>, // Voronoi cell vertices
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
// OPTIMIZED N-GRAM TRIE STRUCTURE
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
}

impl NgramTrie {
    fn new(max_order: usize) -> Self {
        NgramTrie {
            root: NgramTrieNode::new(),
            max_order,
        }
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
                
                self.root.insert(context, next);
            }
        }
    }
    
    fn get_candidates(&self, context: &[String], max_results: usize) -> Vec<(String, f32, usize, usize)> {
        let mut all_candidates = Vec::new();
        
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
    semantic_cells: Vec<SemanticCell>, // Voronoi-style regions
    
    ngram_trie: NgramTrie,
    embeddings: EmbeddingSpace,
    
    dict_corpus_loaded: bool,
    punct_model: PunctuationModel,
    weights: ScoringWeights,
    
    // Hyperbolic geometry
    poincare: PoincareDisc,
    focus_word: Option<String>, // Word at center of current view
    render_threshold: f32, // Only render top % of words (0.8 = 80%)
    
    // UI state
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
    semantic_mode: bool, // Toggle between node mode and semantic cell mode
    
    activation_by_word: HashMap<String, f32>,
    any_active: bool,
    redraw_needed: bool,
    last_camera: (f32, f32, f32),
    frame_counter: u32,

    max_ngram_order: usize,
    config_mode: bool,
    config_input: String,
    
    // Generation state
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
            ngram_trie: NgramTrie::new(7),
            embeddings: EmbeddingSpace::new(50),
            dict_corpus_loaded: false,
            punct_model: PunctuationModel::new(),
            weights: ScoringWeights::balanced(),
            poincare: PoincareDisc::new(1200.0, 0.15), // inner_radius = 0.15 to spread center
            focus_word: None,
            render_threshold: 10000.0, // Render top 10000 words max
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
            semantic_mode: false, // Start in node mode
            max_ngram_order: 7,
            config_mode: false,
            config_input: String::from("7"),
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
            target_chunk_tokens: 500,
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
        
        let is_dict = path.to_lowercase().contains("dict") || 
                      path.to_lowercase().contains("definition");
        
        // Get filename for tracking
        let filename = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .to_string();
        
        // Check if already loaded at this n-gram level
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
        
        // Calculate chunks for batching n-gram insertion
        let num_chunks = ((total_tokens as f32 / self.target_chunk_tokens as f32).ceil() as usize).max(1);
        let tokens_per_chunk = (total_tokens + num_chunks - 1) / num_chunks;
        
        println!("[BATCH] Processing n-grams: {} chunks of ~{} tokens each", 
            num_chunks, tokens_per_chunk);
        
        // Process n-grams in batches (for memory efficiency)
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * tokens_per_chunk;
            let end = ((chunk_idx + 1) * tokens_per_chunk).min(total_tokens);
            let chunk_tokens = &tokens[start..end];
            
            println!("[CHUNK {}/{}] Building n-grams (tokens {}-{})...", 
                chunk_idx + 1, num_chunks, start, end);
            
            self.ngram_trie.insert(chunk_tokens);
        }
        
        // Mark file as loaded ONCE
        self.loaded_files.insert(file_key.clone(), FileLoadInfo {
            filename: filename.clone(),
            ngram_level: self.max_ngram_order,
            load_count: 1,  // Always 1 - file loaded once
        });
        
        self.files_loaded += 1;
        
        // HYPERBOLIC LAYOUT: Frequency-based positioning
        // Build vocabulary from original tokens (deduplicated)
        let mut new_freq: HashMap<String, usize> = HashMap::new();
        for word in &tokens {
            *new_freq.entry(word.clone()).or_insert(0) += 1;
        }
        
        let unique_in_file = new_freq.len();
        println!("[VOCAB] File contains {} unique words (from {} total tokens)", 
            unique_in_file, total_tokens);
        
        // Filter out words that already exist in vocabulary
        let mut sorted_words: Vec<_> = new_freq.iter()
            .filter(|(w, _)| !existing_vocab.contains(*w))
            .collect();
        sorted_words.sort_by(|a, b| b.1.cmp(a.1));
        sorted_words.truncate(150000);
        
        let new_words_to_add = sorted_words.len();
        let already_known = unique_in_file - new_words_to_add;
        
        println!("[VOCAB] {} new words to add | {} already in vocabulary", 
            new_words_to_add, already_known);
        
        if !sorted_words.is_empty() {
            let total_new_words = sorted_words.len();
            let words_before = self.words.len();
            
            for (rank, (word, freq)) in sorted_words.iter().enumerate() {
                let freq_rank = rank as f32 / total_new_words as f32;
                
                // Map to PoincarÃ© disc radius
                let r = self.poincare.freq_to_radius(freq_rank);
                
                // Distribute angular position evenly
                let angle = (rank as f32 * 2.4) % (2.0 * std::f32::consts::PI);
                
                // PoincarÃ© disc coordinates
                let x = r * angle.cos();
                let y = r * angle.sin();
                
                // Size based on hyperbolic position (reduced by 50%)
                let size = self.poincare.radius_to_size(r) * 0.5;
                
                // Compute render priority: high frequency + activation + proximity to center
                let render_priority = (1.0 - freq_rank) * 0.7 + (1.0 - r) * 0.3;
                
                self.words.push(Word {
                    text: (*word).clone(),
                    x,
                    y,
                    freq: **freq,
                    freq_rank,
                    base_size: size,
                    activation: 0.0,
                    layer: self.files_loaded - 1,
                    in_context: false,
                    embedding: None,
                    render_priority,
                    cached_color: None, // Will be computed on first render
                });
            }
            
            println!("[VOCAB] Vocabulary: {} â†’ {} words (+{} from this file)", 
                words_before, self.words.len(), sorted_words.len());
        }
        
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
    
    // ========================================================================
    // HYPERBOLIC NAVIGATION
    // ========================================================================
    
    fn recenter_on_word(&mut self, word_text: &str) {
        if let Some(word) = self.words.iter().find(|w| w.text == word_text) {
            let px = word.x;
            let py = word.y;
            
            // Create temporary coordinate pairs for transformation
            let mut points: Vec<(f32, f32)> = self.words.iter()
                .map(|w| (w.x, w.y))
                .collect();
            
            // Apply MÃ¶bius transformation
            self.poincare.recenter(px, py, &mut points);
            
            // Update word positions
            for (word, (new_x, new_y)) in self.words.iter_mut().zip(points.iter()) {
                word.x = *new_x;
                word.y = *new_y;
                
                // Recalculate size based on new position (reduced by 50%)
                let new_r = (word.x * word.x + word.y * word.y).sqrt();
                word.base_size = self.poincare.radius_to_size(new_r) * 0.5;
                
                // Update render priority based on new position
                word.render_priority = (1.0 - word.freq_rank) * 0.7 + (1.0 - new_r) * 0.3;
            }
            
            self.focus_word = Some(word_text.to_string());
            
            // Reset camera to center
            self.camera_x = 0.0;
            self.camera_y = 0.0;
            
            // Recompute semantic cells if in semantic mode (positions changed)
            // NOTE: This is expensive! Only recompute if really needed.
            // Most of the time, existing cells are good enough.
            if self.semantic_mode && false {  // Disabled - too expensive!
                println!("[CELLS] Recomputing cells after recenter...");
                self.compute_semantic_cells();
            }
            
            println!("[HYPERBOLIC] Recentered on: {}", word_text);
        }
    }
    
    // ========================================================================
    // SEMANTIC CELL COMPUTATION - Voronoi-style regions
    // ========================================================================
    
    fn compute_semantic_cells(&mut self) {
        self.semantic_cells.clear();
        
        // Get top words to render based on priority
        let mut renderable_words: Vec<_> = self.words.iter().collect();
        
        if renderable_words.is_empty() {
            return;
        }
        
        renderable_words.sort_by(|a, b| b.render_priority.partial_cmp(&a.render_priority).unwrap());
        
        // Limit to top 10000 words
        let max_cells = self.render_threshold as usize;
        renderable_words.truncate(max_cells);
        
        println!("[CELLS] Computing cells for {} words (top {})", 
            renderable_words.len(), max_cells);
        
        // BUILD INDEX: O(n) once instead of O(n) per lookup!
        let word_index: HashMap<&str, &Word> = self.words.iter()
            .map(|w| (w.text.as_str(), w))
            .collect();
        
        // For each renderable word, find its semantic neighbors
        for word in &renderable_words {
            let context = vec![word.text.clone()];
            let neighbors = self.ngram_trie.get_candidates(&context, 6);
            
            if neighbors.is_empty() {
                continue;
            }
            
            // Build corners from neighbor positions
            let mut corners = Vec::new();
            
            for (neighbor_text, _, _, _) in neighbors.iter().take(6) {
                // O(1) LOOKUP instead of O(n) search!
                if let Some(&neighbor_word) = word_index.get(neighbor_text.as_str()) {
                    // Corner is midpoint between word and neighbor
                    let mid_x = (word.x + neighbor_word.x) / 2.0;
                    let mid_y = (word.y + neighbor_word.y) / 2.0;
                    corners.push((mid_x, mid_y));
                }
            }
            
            // Need at least 3 corners for a cell
            if corners.len() < 3 {
                continue;
            }
            
            // Sort corners by angle around center to form proper polygon
            let cx = word.x;
            let cy = word.y;
            corners.sort_by(|a, b| {
                let angle_a = (a.1 - cy).atan2(a.0 - cx);
                let angle_b = (b.1 - cy).atan2(b.0 - cx);
                angle_a.partial_cmp(&angle_b).unwrap()
            });
            
            // Cell color based on position and frequency (HSV gradient)
            let hue = (word.freq_rank * 360.0) % 360.0;
            let sat = 0.6;
            let val = 0.3 + word.activation * 0.4;
            
            let color = Self::hsv_to_rgb(hue, sat, val);
            
            // Opacity proportional to n-gram connections
            // More connections = more transparent (lighter)
            // Fewer connections = more opaque (darker/more solid)
            let connection_count = neighbors.len();
            let base_alpha = if connection_count > 0 {
                // Map 1-6 connections to alpha 0.5 down to 0.08
                // Divide base opacity by connection count
                (0.5 / connection_count as f32).max(0.08)
            } else {
                0.5 // No connections = most opaque
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
    
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        Color::new(r + m, g + m, b + m, 0.5)
    }
    
    // ========================================================================
    // LAYER 3: GENERATION ENGINE
    // ========================================================================
    
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
        
        let min_tokens = 500000;
        let max_entropy = 3.5;
        let min_confidence = 0.10;
        let min_coherence = 0.16;
        
        for step in 0..max_tokens {
            println!(" Step {}", step);
            let context_display: Vec<_> = output.iter().rev().take(5).rev().cloned().collect();
            println!(" Context: {:?}", context_display);
            
            let ngram_candidates = self.ngram_trie.get_candidates(&output, 15000000);
            
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
            let confidence = match order {
                7 => 1.0,
                6 => 0.95,
                5 => 0.9,
                4 => 0.75,
                3 => 0.6,
                2 => 0.4,
                _ => 0.2,
            };
            
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
            let confidence = match order {
                7 => 1.0,
                6 => 0.95,
                5 => 0.9,
                4 => 0.75,
                3 => 0.6,
                2 => 0.4,
                _ => 0.2,
            };
            
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
        for n in 2..=7 {
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
    
    // ========================================================================
    // UI & VISUALIZATION - HYPERBOLIC RENDERING
    // ========================================================================
    
    fn update_activations(&mut self) {
        self.frame_counter = (self.frame_counter + 1) % 4;
        let do_decay = self.frame_counter == 0 || self.generating || self.exploring;    
        self.activation_by_word.clear();
        let mut max_act = 0.0;

        for w in &mut self.words {
            if do_decay { w.activation *= 0.95; }
            if w.activation > max_act { max_act = w.activation; }
            self.activation_by_word.insert(w.text.clone(), w.activation);
        }
        self.any_active = max_act > 0.02;
        
        // Update activations based on selected word
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
                    
                    // Position in hyperbolic space relative to selected word
                    let base_offset = 0.15; // Hyperbolic distance
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
        
        // Check expanded words first (highest priority)
        for word in &self.expanded_words {
            let dx = world_x - word.x;
            let dy = world_y - word.y;
            if dx*dx + dy*dy < (word.size * 0.01) * (word.size * 0.01) {
                return Some(word.text.clone());
            }
        }
        
        // ONLY check words that are ACTUALLY RENDERED
        for word in &self.words {
            // Must match the exact same rendering logic in draw()
            let should_render = if self.generating {
                word.activation > 0.05 || word.in_context || Some(&word.text) == self.selected.as_ref()
            } else {
                word.in_context
                    || Some(&word.text) == self.selected.as_ref()
                    || self.expanded_words.iter().any(|e| e.text == word.text)
            };
            
            // Skip if not rendered
            if !should_render {
                continue;
            }
            
            let dx = world_x - word.x;
            let dy = world_y - word.y;
            
            let zoom_scaled_activation = word.activation * self.camera_zoom.powf(0.5);
            let size_mult = if word.in_context { 3.0 } else { 1.0 };
            let base_pixel_size = word.base_size * self.camera_zoom * 2.0;
            let size = base_pixel_size * (1.0 + zoom_scaled_activation * 2.0) * size_mult;
            
            // Tight hitbox based on actual visual size
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
                self.focus_word = None; // Clear hyperbolic focus
                for word in &mut self.words {
                    word.in_context = false;
                }
                return;
            }
            self.redraw_needed = true;

        }
        
        // SPACE: Recenter on selected word (but not during query/config input)
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
        
        // Ctrl+L: Load file
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
            
            // Ctrl+E: Load embeddings
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
            }
            
            // Ctrl+T: Toggle topology mode
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
            }
            
            // Ctrl+S: Toggle semantic mode
            if is_key_pressed(KeyCode::S) {
                self.semantic_mode = !self.semantic_mode;
                println!("\n Visualization mode: {}", 
                    if self.semantic_mode { "SEMANTIC CELLS" } 
                    else { "NODE MODE" });
                
                // Compute cells when entering semantic mode (if not already computed)
                if self.semantic_mode && self.semantic_cells.is_empty() {
                    println!("[CELLS] Computing semantic cells on demand...");
                    self.compute_semantic_cells();
                }
            }
            
            // Ctrl+O: Query (mnemonic: Open query)
            if is_key_pressed(KeyCode::O) {
                self.text_input_active = !self.text_input_active;
                if !self.text_input_active {
                    self.query_text.clear();
                }
                return;
            }
            
            // Ctrl+N: Configure n-gram order
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
            
            // Enable semantic mode for query visualization
            if !self.semantic_mode {
                self.semantic_mode = true;
                println!("\n[QUERY] Enabling semantic visualization mode");
            }
            
            // Compute semantic cells if needed (only if truly empty)
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
                
                // Add newline after sentence-ending punctuation
                if current_word == "." || current_word == "!" || current_word == "?" {
                    self.generated_output.push('\n');
                }
                
                // Decay all activations
                for word in &mut self.words {
                    word.activation *= 0.7;
                }
                
                // Force activate the current generated word (regardless of selection)
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
        // Grey background instead of black
        clear_background(GREY_BG);
        
        let center_x = screen_width() / 2.0 + self.camera_x;
        let center_y = screen_height() / 2.0 + self.camera_y;
        
        // Draw PoincarÃ© disc boundary
        let boundary_radius = self.poincare.radius * self.camera_zoom;
        draw_circle_lines(center_x, center_y, boundary_radius, 2.0, 
            Color::new(0.3, 0.3, 0.35, 0.6));
        
        // Draw radial guides (frequency zones)
        for i in 1..5 {
            let r = (i as f32 / 5.0) * boundary_radius;
            draw_circle_lines(center_x, center_y, r, 1.0, 
                Color::new(0.2, 0.2, 0.25, 0.4));
        }
        
        // ========================================================================
        // DRAW SEMANTIC CELLS (only when activated) - HIGHLY OPTIMIZED
        // ========================================================================
        if self.semantic_mode {
            // Precompute which cells to render (avoid repeated lookups)
            let mut cells_to_render: Vec<(&SemanticCell, f32)> = Vec::new();
            
            for (cell_idx, cell) in self.semantic_cells.iter().enumerate() {
                if cell.corners.len() < 3 {
                    continue;
                }
                
                // O(1) LOOKUP using HashMap instead of O(n) linear search!
                let center_activation = self.activation_by_word
                    .get(&cell.center_word)
                    .copied()
                    .unwrap_or(0.0);
                
                // Check if we should render this cell
                let should_render = if self.generating || self.exploring {
                    // During generation, show any activated cell (very low threshold)
                    center_activation > 0.01
                } else {
                    // In semantic mode but not generating:
                    // Show activated cells OR every 10th cell for default visualization
                    center_activation > 0.05 || cell_idx % 10 == 0
                };
                
                if should_render {
                    cells_to_render.push((cell, center_activation));
                }
            }
            
            // Debug: Print cell count periodically
            static mut FRAME_COUNT: u32 = 0;
            unsafe {
                FRAME_COUNT += 1;
                if FRAME_COUNT % 300 == 0 {  // Less frequent - every 5 seconds
                    println!("[PERF] Rendering {} / {} cells | {} words total", 
                        cells_to_render.len(), self.semantic_cells.len(), self.words.len());
                }
            }
            
            // Batch render all cells
            for (cell, center_activation) in cells_to_render {
                // Precompute screen corners once
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
                
                // Brighten cell based on activation (but keep transparent)
                let brightness_boost = center_activation * 0.3;
                let boosted_color = Color::new(
                    (cell.color.r + brightness_boost).min(1.0),
                    (cell.color.g + brightness_boost).min(1.0),
                    (cell.color.b + brightness_boost).min(1.0),
                    (cell.color.a + center_activation * 0.1).min(0.3),
                );
                
                // Draw filled polygon using triangle fan from first vertex
                // More efficient than computing center
                let v0 = Vec2::new(screen_corners[0].0, screen_corners[0].1);
                for i in 1..screen_corners.len() - 1 {
                    draw_triangle(
                        v0,
                        Vec2::new(screen_corners[i].0, screen_corners[i].1),
                        Vec2::new(screen_corners[i + 1].0, screen_corners[i + 1].1),
                        boosted_color,
                    );
                }
                
                // Draw cell boundary
                for i in 0..screen_corners.len() {
                    let next = (i + 1) % screen_corners.len();
                    let base_alpha = 0.3 + cell.activation * 0.4;
                    let border_alpha = base_alpha + center_activation * 0.3;
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
        
        // Connection lines to expanded words
        if let Some(selected) = &self.selected {
            if let Some(selected_word) = self.words.iter().find(|w| &w.text == selected) {
                let sel_x = center_x + selected_word.x * self.poincare.radius * self.camera_zoom;
                let sel_y = center_y + selected_word.y * self.poincare.radius * self.camera_zoom;
                
                for word in &self.expanded_words {
                    let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
                    let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
                    
                    // N-gram order color gradient (green spectrum)
                    let line_color = match word.ngram_order {
                        2 => Color::new(0.0, 1.0, 0.0, 0.6),
                        3 => Color::new(0.0, 1.0, 1.0, 0.6),
                        4 => Color::new(1.0, 1.0, 0.0, 0.6),
                        5 => Color::new(1.0, 0.6, 0.0, 0.6),
                        6 => Color::new(1.0, 0.0, 1.0, 0.6),
                        7 => Color::new(1.0, 0.2, 0.2, 0.6),
                        _ => Color::new(1.0, 1.0, 1.0, 0.6),
                    };
                    
                    draw_line(sel_x, sel_y, screen_x, screen_y, 2.0, line_color);
                }
            }
        }
        
        // ========================================================================
        // DRAW WORDS - OPTIMIZED: Cache n-gram lookups, filter early
        // ========================================================================
        
        // OPTIMIZATION: Build a cache of n-gram data for words we'll actually render
        let mut ngram_cache: HashMap<&str, (usize, f32)> = HashMap::new(); // (max_order, connection_strength)
        
        // Collect renderable words with their confidence scores
        let mut renderable_words: Vec<&Word> = self.words.iter()
            .filter(|word| {
                // In node mode: render top words based on priority
                // In semantic mode: only render if activated or selected
                if self.semantic_mode {
                    // Semantic mode: only show activated/selected words
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
                    // Node mode: EARLY FILTER - only consider top priority words
                    // This prevents processing thousands of low-priority words
                    word.render_priority > 0.5  // Filter to top ~50% before expensive operations
                        || word.activation > 0.05 
                        || word.in_context 
                        || Some(&word.text) == self.selected.as_ref()
                }
            })
            .collect();
        
        // Limit rendering in node mode to top N words BEFORE expensive calculations
        if !self.semantic_mode && renderable_words.len() > self.render_threshold as usize {
            renderable_words.sort_by(|a, b| b.render_priority.partial_cmp(&a.render_priority).unwrap());
            renderable_words.truncate(self.render_threshold as usize);
        }
        
        // OPTIMIZATION: Now do ONE n-gram lookup per word and cache it
        // SIGMOID FILTERING: Probabilistically render based on n-gram order
        // - Low order (2-3g) = render most (common connections)
        // - High order (6-7g) = render few (rare specific patterns)
        
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Sigmoid function: 1 / (1 + e^(-k*(x-x0)))
        // Returns probability of rendering based on n-gram order
        let sigmoid_render_prob = |ngram_order: usize| -> f32 {
            let x = ngram_order as f32;
            let k = -1.0;  // Steepness (negative = inverted sigmoid)
            let x0 = 4.5;  // Midpoint (50% render probability at 4.5-grams)
            
            1.0 / (1.0 + (k * (x - x0)).exp())
        };
        
        // Deterministic hash-based selection (same word always gets same result)
        let should_render_word = |word_text: &str, ngram_order: usize| -> bool {
            // Always render selected, context, and activated words
            if word_text == self.selected.as_ref().map(|s| s.as_str()).unwrap_or("") {
                return true;
            }
            
            let prob = sigmoid_render_prob(ngram_order);
            
            // Use hash for deterministic pseudo-random selection
            let mut hasher = DefaultHasher::new();
            word_text.hash(&mut hasher);
            let hash_val = hasher.finish();
            let random_val = (hash_val % 1000) as f32 / 1000.0;
            
            random_val < prob
        };
        
        for word in &renderable_words {
            if Some(&word.text) != self.selected.as_ref() && !word.in_context {
                let context = vec![word.text.clone()];
                let neighbors = self.ngram_trie.get_candidates(&context, 50);
                
                let connection_strength = if !neighbors.is_empty() {
                    let max_order = neighbors.iter().map(|(_, _, order, _)| order).max().unwrap_or(&2);
                    
                    // SIGMOID FILTER: Decide if we should render this word
                    if !word.in_context && word.activation < 0.05 {
                        if !should_render_word(&word.text, *max_order) {
                            // Skip this word - filtered out by sigmoid
                            continue;
                        }
                    }
                    
                    let strength = (*max_order as f32 / 7.0);
                    ngram_cache.insert(&word.text, (*max_order, strength));
                    strength
                } else {
                    ngram_cache.insert(&word.text, (2, 0.0));
                    0.0
                };
            }
        }
        
        // Sort by confidence: lowest first, highest last (drawn on top)
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
        
        // Debug output every 5 seconds
        static mut DEBUG_FRAME: u32 = 0;
        unsafe {
            DEBUG_FRAME += 1;
            if DEBUG_FRAME % 300 == 0 {
                // Calculate filtering stats
                let mut order_counts: HashMap<usize, usize> = HashMap::new();
                for (_, (order, _)) in &ngram_cache {
                    *order_counts.entry(*order).or_insert(0) += 1;
                }
                
                println!("[RENDER] Drawing {} words (cached: {}) in {} mode | Total vocab: {}", 
                    renderable_words.len(),
                    ngram_cache.len(),
                    if self.semantic_mode { "SEMANTIC" } else { "NODE" },
                    self.words.len());
                
                println!("  Sigmoid filter: 2g:{} 3g:{} 4g:{} 5g:{} 6g:{} 7g:{}", 
                    order_counts.get(&2).unwrap_or(&0),
                    order_counts.get(&3).unwrap_or(&0),
                    order_counts.get(&4).unwrap_or(&0),
                    order_counts.get(&5).unwrap_or(&0),
                    order_counts.get(&6).unwrap_or(&0),
                    order_counts.get(&7).unwrap_or(&0));
            }
        }
        
        for word in renderable_words {
            // Skip words that were filtered out by sigmoid (not in cache and not special)
            if !word.in_context 
                && Some(&word.text) != self.selected.as_ref() 
                && word.activation < 0.05
                && !ngram_cache.contains_key(word.text.as_str()) {
                continue;
            }
            
            let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
            
            // Cull offscreen
            if screen_x < -100.0 || screen_x > screen_width() + 100.0 ||
               screen_y < -100.0 || screen_y > screen_height() + 100.0 {
                continue;
            }
            
            let base_pixel_size = word.base_size * self.camera_zoom * 2.0;
            
            // Activation scales with zoom level
            let zoom_scaled_activation = word.activation * self.camera_zoom.powf(0.5);
            
            // Dormant words rendered as small dots (only in node mode)
            if !self.semantic_mode && word.activation < 0.05 && Some(&word.text) != self.selected.as_ref() && !word.in_context {
                draw_circle(screen_x, screen_y, base_pixel_size * 0.4, 
                    Color::new(0.5, 0.5, 0.5, 0.3)); // Lighter gray dots
                continue;
            }
            
            // Calculate size and color for all active words
            let size_mult = if word.in_context { 2.0 } else { 1.0 };
            let size = base_pixel_size * (1.0 + zoom_scaled_activation * 2.0) * size_mult;
            
            // Node color - use CACHED n-gram data for confidence calculation
            let (color, glow_color) = if Some(&word.text) == self.selected.as_ref() {
                (CYAN, CYAN) // Selected stays cyan
            } else if word.in_context {
                (YELLOW, YELLOW) // Context stays yellow
            } else {
                // OPTIMIZED: Use cached connection strength instead of recalculating
                let connection_strength = ngram_cache.get(word.text.as_str())
                    .map(|(_, strength)| *strength)
                    .unwrap_or(0.0);
                
                let confidence = (zoom_scaled_activation * 0.5 + connection_strength * 0.5).min(1.0);
                
                // More confident = more blue, less confident = more gray
                let base_color = Color::new(
                    0.3 + confidence * 0.0,  // Red: stays low
                    0.3 + confidence * 0.4,  // Green: slight increase
                    0.3 + confidence * 0.7,  // Blue: high increase
                    0.7 + confidence * 0.3
                );
                
                let glow = Color::new(
                    base_color.r,
                    base_color.g,
                    base_color.b,
                    0.1
                );
                
                (base_color, glow)
            };
            
            // In semantic mode: don't draw nodes, only text labels
            if !self.semantic_mode {
                // Glow
                let glow_alpha = zoom_scaled_activation * 0.06;
                draw_circle(screen_x, screen_y, size * 1.8,
                    Color::new(glow_color.r, glow_color.g, glow_color.b, glow_alpha));
                
                draw_circle(screen_x, screen_y, size, color);
            }
            
            // Labels - in semantic mode, show as black text on cells
            let should_show_label = if self.semantic_mode {
                // Semantic mode: show labels for any visible word
                true
            } else {
                // Node mode: only show labels if zoomed in enough
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
                    // Black text on colored cells
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
        
        // Draw bridge words - small 5px circles with cell-matching colors
        for bridge in &self.bridge_words {
            let screen_x = center_x + bridge.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + bridge.y * self.poincare.radius * self.camera_zoom;
            let size = 2.5; // Fixed 5px diameter (2.5px radius)
            
            // Match cell color scheme - use coherence to determine hue
            let hue = (bridge.coherence * 360.0) % 360.0;
            let sat = 0.6;
            let val = 0.5;
            let bridge_color = Self::hsv_to_rgb(hue, sat, val);
            
            // Semi-transparent to blend additively with cells
            let final_color = Color::new(
                bridge_color.r,
                bridge_color.g,
                bridge_color.b,
                0.4
            );
            
            draw_circle(screen_x, screen_y, size, final_color);
        }
        
        // Draw expanded words - green spectrum
        for word in &self.expanded_words {
            let screen_x = center_x + word.x * self.poincare.radius * self.camera_zoom;
            let screen_y = center_y + word.y * self.poincare.radius * self.camera_zoom;
            let size = (word.size * self.camera_zoom * 2.5).max(10.0);
            
            let ngram_color = match word.ngram_order {
                2 => Color::new(0.0, 1.0, 0.0, 1.0),
                3 => Color::new(0.0, 1.0, 1.0, 1.0),
                4 => Color::new(1.0, 1.0, 0.0, 1.0),
                5 => Color::new(1.0, 0.6, 0.0, 1.0),
                6 => Color::new(1.0, 0.0, 1.0, 1.0),
                7 => Color::new(1.0, 0.2, 0.2, 1.0),
                _ => WHITE,
            };
            
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
        
        // UI
        let mode_text = if self.semantic_mode {
            if self.use_embeddings {
                "SEMANTIC CELLS: DUAL TOPOLOGY (N-grams + Embeddings)"
            } else {
                "SEMANTIC CELLS: N-GRAMS ONLY"
            }
        } else {
            if self.use_embeddings {
                "NODE MODE: DUAL TOPOLOGY (N-grams + Embeddings)"
            } else {
                "NODE MODE: N-GRAMS ONLY"
            }
        };
        
        let mode_color = if self.semantic_mode { MAGENTA } else { CYAN };
        draw_text(mode_text, 10.0, 30.0, 20.0, mode_color);
        
        draw_text("Ctrl+L: Load | Ctrl+E: Embeddings | Ctrl+T: Topology | Ctrl+S: Semantic | Ctrl+O: Query | Ctrl+N: Config", 
            10.0, 55.0, 18.0, WHITE);
        
        draw_text("ESC: Cancel | R: Reset | C: Clear | SPACE: Recenter | Mouse: Pan/Zoom", 
            10.0, 75.0, 16.0, Color::new(0.8, 0.8, 0.8, 1.0));
        
        let rendered_count = if self.generating {
            self.words.iter()
                .filter(|w| w.activation > 0.05 || w.in_context || Some(&w.text) == self.selected.as_ref())
                .count()
        } else {
            self.words.iter()
                .filter(|w| w.in_context || Some(&w.text) == self.selected.as_ref())
                .count()
                + self.expanded_words.len()
        };
        
        draw_text(&format!("Files: {} | Words: {} | Cells: {} | Bridges: {} | Order: {}", 
            self.files_loaded,
            self.words.len(),
            self.semantic_cells.len(), self.bridge_words.len(), self.max_ngram_order), 
            10.0, 95.0, 16.0, WHITE);
        
        if let Some(focus) = &self.focus_word {
            draw_text(&format!("Focus: {}", focus), 10.0, 115.0, 16.0, ORANGE);
        }
        
        // File list
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
            draw_text(&format!("Selected: {}", selected), 10.0, base_y + 5.0, 18.0, CYAN);
        }
        
        if !self.context_path.is_empty() {
            let context = self.context_path.join(" â†’ ");
            draw_text(&format!("Context: {}", context), 10.0, base_y + 30.0, 16.0, YELLOW);
        }
        
        // Modal overlays
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
                "â˜… DUAL TOPOLOGY GENERATION..."
            } else {
                "â˜… N-GRAM GENERATION..."
            };
            draw_text(text, 10.0, box_y + 30.0, 20.0, ORANGE);
        } else if self.generating {
            let box_y = screen_height() - 150.0;
            draw_rectangle(0.0, box_y, screen_width(), 150.0, 
                Color::new(0.1, 0.1, 0.1, 0.9));
            
            draw_text("GENERATING...", 10.0, box_y + 20.0, 20.0, CYAN);
            
            // Show last 3 lines of generated output
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
                "Ctrl+S: Semantic Mode | SPACE: Recenter | Click nodes | ESC: Deselect | Ctrl+O: Query"
            };
            draw_text(hint_text, 10.0, hint_y, 18.0, Color::new(0.6, 0.6, 0.6, 1.0));
        }
    }
}



#[macroquad::main("M2 - SEMANTIC CELLS (Relationship-Defined Regions)")]
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
        println!("  M2 - SEMANTIC CELLS");
        println!("========================================");
        println!("\nâ€¢ Hyperbolic PoincarÃ© disc");
        println!("â€¢ Semantic cells (relationship polygons)");
        println!("â€¢ Only top 80% of words rendered");
        println!("â€¢ Common words â†’ center (large cells)");
        println!("â€¢ Rare words â†’ edge (small cells)");
        println!("â€¢ Cell boundaries = n-gram relationships");
        println!("â€¢ SPACE to recenter on any word");
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