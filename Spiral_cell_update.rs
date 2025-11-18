// M2 v2 - Radial Semantic Memory System
// Part 1: INITIALIZATION + WORD POINT VISUALIZATION MATH

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use macroquad::prelude::*;
use rfd::FileDialog;
use chrono::{DateTime, Utc};

// ============================================================================
// FILE SYSTEM SETUP
// ============================================================================

fn ensure_data_directories() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let base_dir = PathBuf::from("./m2_data");
    let states_dir = base_dir.join("states");
    let ngrams_dir = base_dir.join("ngrams");
    let corpora_dir = base_dir.join("corpora");
    
    fs::create_dir_all(&states_dir)?;
    fs::create_dir_all(&ngrams_dir)?;
    fs::create_dir_all(&corpora_dir)?;
    
    Ok(base_dir)
}

// ============================================================================
// INITIALIZATION - Constants
// ============================================================================

const MAX_VISIBLE_WORDS: usize = 500;  // Viewport capacity
const WORDS_PER_RING: usize = 100;     // Words per radial ring
const RING_SPACING: f32 = 50.0;        // Pixels between rings
const MAX_NGRAM_ORDER: usize = 6;      // Track 2-6 grams by default
const BRIDGE_THRESHOLD: f64 = 2.0;     // Minimum bridge score to highlight

// Color constants
const BLACK: Color = Color::new(0.0, 0.0, 0.0, 1.0);
const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);
const CYAN: Color = Color::new(0.0, 1.0, 1.0, 1.0);

// ============================================================================
// SERIALIZABLE COLOR WRAPPER
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableColor {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl SerializableColor {
    fn from_color(c: Color) -> Self {
        Self { r: c.r, g: c.g, b: c.b, a: c.a }
    }
    
    fn to_color(&self) -> Color {
        Color::new(self.r, self.g, self.b, self.a)
    }
}

// ============================================================================
// INITIALIZATION - Data Structures
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Word {
    text: String,
    
    // Position in radial layout
    ring: usize,           // 0 = center, higher = outer
    sector: usize,         // Behavioral cluster ID
    angle: f32,            // Position within sector (radians)
    
    // Cached screen position (not serialized - recalculated on load)
    #[serde(skip)]
    x: f32,
    #[serde(skip)]
    y: f32,
    
    // Statistics
    total_count: u64,      // Total occurrences
    
    // Context analysis
    dominant_order: usize,  // Which n-gram order dominates (2-6+)
    confidence: f64,        // 0.0-1.0, how confident in dominant order
    
    // Rendering
    color: SerializableColor,          // HSV-based color
    size: f32,             // Display size
    
    #[serde(skip)]
    activation: f32,       // Query activation (0.0-1.0)
}

impl Word {
    fn get_color(&self) -> Color {
        self.color.to_color()
    }
    
    fn set_color(&mut self, color: Color) {
        self.color = SerializableColor::from_color(color);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Ngram {
    words: Vec<String>,
    count: u64,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
}

impl Ngram {
    fn order(&self) -> usize {
        self.words.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WordBehavior {
    word: String,
    left_context: HashMap<String, u64>,   // Words that appear before
    right_context: HashMap<String, u64>,  // Words that appear after
    ngram_orders: HashMap<usize, u64>,    // Which orders does this word participate in
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WordCluster {
    id: usize,
    members: HashSet<String>,
    centroid: HashMap<String, f64>,  // Average behavioral profile
    angular_range: (f32, f32),       // (start_angle, end_angle) in radians
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SemanticCell {
    words: Vec<String>,
    edge_colors: Vec<SerializableColor>,
    fill_color: SerializableColor,
    ngram: Ngram,
    
    #[serde(skip)]
    activation: f32,
}

impl SemanticCell {
    fn get_edge_colors(&self) -> Vec<Color> {
        self.edge_colors.iter().map(|c| c.to_color()).collect()
    }
    
    fn get_fill_color(&self) -> Color {
        self.fill_color.to_color()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CorpusMetadata {
    id: String,
    name: String,
    corpus_type: CorpusType,
    created_at: DateTime<Utc>,
    dependencies: Vec<String>,  // IDs of required corpora
    tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum CorpusType {
    Original,      // Base material
    Conversation,  // Dialogue sessions
    Reaction,      // Response to other corpora
    Synthesis,     // Combines multiple sources
}

#[derive(Serialize, Deserialize)]
struct M2State {
    ngrams: HashMap<Vec<String>, Ngram>,
    word_behaviors: HashMap<String, WordBehavior>,
    clusters: Vec<WordCluster>,
    loaded_corpora: Vec<CorpusMetadata>,
    created_at: DateTime<Utc>,
    last_modified: DateTime<Utc>,
}

struct RadialViewport {
    ring_contents: Vec<Vec<String>>,  // ring_contents[0] = Ring 0 words
    query_context: Option<String>,
    center_time: DateTime<Utc>,
}

// Helper functions
fn get_last_key_pressed() -> Option<KeyCode> {
    for key_code in [
        KeyCode::Key0, KeyCode::Key1, KeyCode::Key2, KeyCode::Key3, KeyCode::Key4,
        KeyCode::Key5, KeyCode::Key6, KeyCode::Key7, KeyCode::Key8, KeyCode::Key9,
        KeyCode::Backspace, KeyCode::Enter,
    ] {
        if is_key_pressed(key_code) {
            return Some(key_code);
        }
    }
    None
}

// ============================================================================
// WORD POINT VISUALIZATION MATH
// ============================================================================

/// Calculate bridge score: how well does this word connect different clusters?
fn calculate_bridge_score(
    word: &str, 
    word_behaviors: &HashMap<String, WordBehavior>,
    clusters: &[WordCluster]
) -> f64 {
    let behavior = match word_behaviors.get(word) {
        Some(b) => b,
        None => return 0.0,
    };
    
    // Get all neighbors (words that co-occur with this word)
    let mut neighbors = HashSet::new();
    for neighbor in behavior.left_context.keys() {
        neighbors.insert(neighbor.clone());
    }
    for neighbor in behavior.right_context.keys() {
        neighbors.insert(neighbor.clone());
    }
    
    // Find which clusters these neighbors belong to
    let mut cluster_connections: HashMap<usize, usize> = HashMap::new();
    for neighbor in neighbors {
        for cluster in clusters {
            if cluster.members.contains(&neighbor) {
                *cluster_connections.entry(cluster.id).or_insert(0) += 1;
            }
        }
    }
    
    if cluster_connections.is_empty() {
        return 0.0;
    }
    
    // Bridge score = number of distinct clusters × balance
    let num_clusters = cluster_connections.len() as f64;
    let counts: Vec<usize> = cluster_connections.values().copied().collect();
    let balance = calculate_entropy(&counts);
    
    num_clusters * balance
}

/// Calculate Shannon entropy for cluster connection balance
fn calculate_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    
    let probabilities: Vec<f64> = counts.iter()
        .map(|&count| count as f64 / total as f64)
        .collect();
    
    -probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f64>()
}

/// Assign word to radial position based on query relevance and bridge score
fn assign_word_to_ring(
    _word: &str,
    relevance_score: f64,
    bridge_score: f64,
) -> usize {
    // Combined score determines ring (lower ring = more central/important)
    let combined_score = relevance_score + (bridge_score * 0.5);
    
    if combined_score > 10.0 {
        0  // Center ring
    } else if combined_score > 5.0 {
        1
    } else if combined_score > 2.0 {
        2
    } else if combined_score > 1.0 {
        3
    } else {
        4  // Outer ring
    }
}

/// Calculate angle within sector based on neighbor similarity
fn calculate_angle_in_sector(
    word: &str,
    sector_id: usize,
    word_behaviors: &HashMap<String, WordBehavior>,
    clusters: &[WordCluster],
) -> f32 {
    let cluster = &clusters[sector_id];
    let sector_start = cluster.angular_range.0;
    let sector_width = cluster.angular_range.1 - cluster.angular_range.0;
    
    // Use simple hash-based positioning within sector for deterministic placement
    let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    let normalized_hash = (hash as f32) / (u32::MAX as f32);
    
    sector_start + (normalized_hash * sector_width)
}

/// Calculate HSV color based on dominant n-gram order and confidence
fn calculate_word_color(dominant_order: usize, confidence: f64) -> Color {
    // Hue: 2-gram=0° (red), 3-gram=72° (orange), 4-gram=144° (yellow-green), 
    //      5-gram=216° (cyan), 6-gram=288° (purple)
    let hue = ((dominant_order - 2) as f32 * 72.0) / 360.0;
    
    // Saturation and Value based on confidence
    let saturation = 0.3 + (confidence as f32 * 0.7);  // 30% to 100%
    let value = 0.5 + (confidence as f32 * 0.5);       // 50% to 100%
    
    hsv_to_rgb(hue, saturation, value)
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let h = h * 360.0;
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
    
    Color::new(r + m, g + m, b + m, 1.0)
}

/// Calculate similarity between two word behaviors using neighbor overlap
fn calculate_word_similarity(
    a: &WordBehavior,
    b: &WordBehavior,
) -> f64 {
    // Jaccard similarity on combined context (left + right neighbors)
    let mut a_neighbors = HashSet::new();
    for n in a.left_context.keys() {
        a_neighbors.insert(n.clone());
    }
    for n in a.right_context.keys() {
        a_neighbors.insert(n.clone());
    }
    
    let mut b_neighbors = HashSet::new();
    for n in b.left_context.keys() {
        b_neighbors.insert(n.clone());
    }
    for n in b.right_context.keys() {
        b_neighbors.insert(n.clone());
    }
    
    let intersection = a_neighbors.intersection(&b_neighbors).count();
    let union = a_neighbors.union(&b_neighbors).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// Fixed: type signature to match usage
fn jaccard_similarity_maps(a: &HashMap<usize, u64>, b: &HashMap<usize, u64>) -> f64 {
    let all_keys: HashSet<usize> = a.keys().chain(b.keys()).copied().collect();
    
    let mut intersection: u64 = 0;
    let mut union: u64 = 0;
    
    for key in all_keys {
        let a_val = a.get(&key).unwrap_or(&0);
        let b_val = b.get(&key).unwrap_or(&0);
        intersection += a_val.min(b_val);
        union += a_val.max(b_val);
    }
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ============================================================================
// RADIAL VIEWPORT - Query-driven content batching
// ============================================================================

impl RadialViewport {
    fn new() -> Self {
        Self {
            ring_contents: vec![Vec::new(); 5],
            query_context: None,
            center_time: Utc::now(),
        }
    }
    
    /// Update viewport based on query, re-batching words into rings
    fn update_from_query(
        &mut self,
        query: &str,
        words: &HashMap<String, Word>,
        word_behaviors: &HashMap<String, WordBehavior>,
        clusters: &[WordCluster],
    ) {
        // Clear existing rings
        for ring in &mut self.ring_contents {
            ring.clear();
        }
        
        self.query_context = Some(query.to_string());
        
        // Score all words by query relevance
        let query_lower = query.to_lowercase();
        let mut scored_words: Vec<(String, f64, f64)> = Vec::new();
        
        for (word_text, word) in words {
            // Relevance: direct match gives high score
            let relevance = if word_text.to_lowercase().contains(&query_lower) {
                10.0 + (word.total_count as f64).log10()
            } else {
                // Check if word appears in same n-grams as query words
                0.0  // Simplified for now
            };
            
            let bridge = calculate_bridge_score(word_text, word_behaviors, clusters);
            
            if relevance > 0.0 || bridge > BRIDGE_THRESHOLD {
                scored_words.push((word_text.clone(), relevance, bridge));
            }
        }
        
        // Sort by combined score
        scored_words.sort_by(|a, b| {
            let score_a = a.1 + (a.2 * 0.5);
            let score_b = b.1 + (b.2 * 0.5);
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        // Limit to MAX_VISIBLE_WORDS
        scored_words.truncate(MAX_VISIBLE_WORDS);
        
        // Assign to rings
        for (word_text, relevance, bridge) in scored_words {
            let ring = assign_word_to_ring(&word_text, relevance, bridge);
            if ring < self.ring_contents.len() {
                self.ring_contents[ring].push(word_text);
            }
        }
    }
}

// ============================================================================
// SEMANTIC CELL GENERATION
// ============================================================================

fn generate_semantic_cells(
    viewport: &RadialViewport,
    ngrams: &HashMap<Vec<String>, Ngram>,
    words: &HashMap<String, Word>,
) -> Vec<SemanticCell> {
    let mut cells = Vec::new();
    
    // Get all words visible in viewport
    let mut visible_words = HashSet::new();
    for ring in &viewport.ring_contents {
        for word in ring {
            visible_words.insert(word.clone());
        }
    }
    
    // Find n-grams where ALL words are in viewport
    for (ngram_words, ngram) in ngrams {
        if ngram_words.iter().all(|w| visible_words.contains(w)) {
            // Collect edge colors (one per word in n-gram)
            let edge_colors: Vec<SerializableColor> = ngram_words.iter()
                .filter_map(|w| words.get(w))
                .map(|word| word.color.clone())
                .collect();
            
            // Fill color = average hue of edges
            let avg_color = if edge_colors.is_empty() {
                SerializableColor::from_color(Color::new(0.5, 0.5, 0.5, 0.3))
            } else {
                // Simple average
                let avg_r = edge_colors.iter().map(|c| c.r).sum::<f32>() / edge_colors.len() as f32;
                let avg_g = edge_colors.iter().map(|c| c.g).sum::<f32>() / edge_colors.len() as f32;
                let avg_b = edge_colors.iter().map(|c| c.b).sum::<f32>() / edge_colors.len() as f32;
                SerializableColor { r: avg_r, g: avg_g, b: avg_b, a: 0.3 }
            };
            
            cells.push(SemanticCell {
                words: ngram_words.clone(),
                edge_colors,
                fill_color: avg_color,
                ngram: ngram.clone(),
                activation: 0.0,
            });
        }
    }
    
    cells
}

// ============================================================================
// SAVE/LOAD - Bincode with proper API
// ============================================================================

impl M2State {
    fn new() -> Self {
        Self {
            ngrams: HashMap::new(),
            word_behaviors: HashMap::new(),
            clusters: Vec::new(),
            loaded_corpora: Vec::new(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
        }
    }
    
    fn save_binary(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        fs::write(path, encoded)?;
        println!("========== Saved state to {:?} ==========", path);
        Ok(())
    }
    
    fn load_binary(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        let state: M2State = bincode::deserialize(&bytes)?;
        println!("========== Loaded state from {:?} ==========", path);
        Ok(state)
    }
    
    fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        println!("========== Saved JSON to {:?} ==========", path);
        Ok(())
    }
    
    fn load_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let state: M2State = serde_json::from_reader(file)?;
        println!("========== Loaded JSON from {:?} ==========", path);
        Ok(state)
    }
}

/// Export n-grams only (JSONL format)
fn export_ngrams(
    ngrams: &HashMap<Vec<String>, Ngram>,
    path: &Path
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for (_words, ngram) in ngrams {
        let line = serde_json::to_string(&ngram)?;
        writeln!(writer, "{}", line)?;
    }
    
    writer.flush()?;
    println!("========== Exported {} n-grams to {:?} ==========", ngrams.len(), path);
    Ok(())
}

/// Auto-export ngrams when they get too large (memory management)
fn auto_export_if_needed(
    ngrams: &HashMap<Vec<String>, Ngram>,
    data_dir: &Path,
    threshold: usize,  // Export if ngrams exceed this count
) -> Result<(), Box<dyn std::error::Error>> {
    if ngrams.len() > threshold {
        let timestamp = Utc::now().format("%Y-%m-%d_%H-%M-%S");
        let export_path = data_dir.join("ngrams").join(format!("ngrams_auto_{}.jsonl", timestamp));
        export_ngrams(ngrams, &export_path)?;
        println!("⚠️  Auto-exported {} n-grams to disk (memory threshold reached)", ngrams.len());
    }
    Ok(())
}

// ============================================================================
// N-GRAM BUILDING FROM TEXT
// ============================================================================

fn build_ngrams_from_text(
    text: &str,
    existing_ngrams: &mut HashMap<Vec<String>, Ngram>,
    max_order: usize,
) {
    let now = Utc::now();
    let words: Vec<String> = text
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .collect();
    
    for order in 2..=max_order {
        for window in words.windows(order) {
            let key = window.to_vec();
            
            existing_ngrams.entry(key.clone())
                .and_modify(|ngram| {
                    ngram.count += 1;
                    ngram.last_seen = now;
                })
                .or_insert(Ngram {
                    words: key,
                    count: 1,
                    first_seen: now,
                    last_seen: now,
                });
        }
    }
}

fn build_word_behaviors(ngrams: &HashMap<Vec<String>, Ngram>) -> HashMap<String, WordBehavior> {
    let mut behaviors: HashMap<String, WordBehavior> = HashMap::new();
    
    for (ngram_words, ngram) in ngrams {
        let order = ngram_words.len();
        
        for (i, word) in ngram_words.iter().enumerate() {
            behaviors.entry(word.clone())
                .and_modify(|behavior| {
                    // Track n-gram order participation
                    *behavior.ngram_orders.entry(order).or_insert(0) += ngram.count;
                    
                    // Track left context
                    if i > 0 {
                        *behavior.left_context.entry(ngram_words[i-1].clone()).or_insert(0) += ngram.count;
                    }
                    
                    // Track right context
                    if i < ngram_words.len() - 1 {
                        *behavior.right_context.entry(ngram_words[i+1].clone()).or_insert(0) += ngram.count;
                    }
                })
                .or_insert_with(|| {
                    let mut behavior = WordBehavior {
                        word: word.clone(),
                        left_context: HashMap::new(),
                        right_context: HashMap::new(),
                        ngram_orders: HashMap::new(),
                    };
                    
                    behavior.ngram_orders.insert(order, ngram.count);
                    
                    if i > 0 {
                        behavior.left_context.insert(ngram_words[i-1].clone(), ngram.count);
                    }
                    if i < ngram_words.len() - 1 {
                        behavior.right_context.insert(ngram_words[i+1].clone(), ngram.count);
                    }
                    
                    behavior
                });
        }
    }
    
    behaviors
}

// ============================================================================
// CLUSTERING
// ============================================================================

fn cluster_words_by_behavior(
    behaviors: &HashMap<String, WordBehavior>,
    num_clusters: usize,
) -> Vec<WordCluster> {
    // Simplified k-means style clustering
    let mut clusters: Vec<WordCluster> = Vec::new();
    
    // Initialize clusters with even angular spacing
    let angle_per_cluster = std::f32::consts::TAU / num_clusters as f32;
    for i in 0..num_clusters {
        clusters.push(WordCluster {
            id: i,
            members: HashSet::new(),
            centroid: HashMap::new(),
            angular_range: (i as f32 * angle_per_cluster, (i + 1) as f32 * angle_per_cluster),
        });
    }
    
    // Assign words to nearest cluster (using simple hash for determinism)
    for (word, _behavior) in behaviors {
        let hash = word.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
        let cluster_id = hash % num_clusters;
        clusters[cluster_id].members.insert(word.clone());
    }
    
    clusters
}

// ============================================================================
// WORD CONSTRUCTION
// ============================================================================

fn build_words_from_behaviors(
    behaviors: &HashMap<String, WordBehavior>,
    clusters: &[WordCluster],
) -> HashMap<String, Word> {
    let mut words = HashMap::new();
    
    for (word_text, behavior) in behaviors {
        // Find dominant n-gram order
        let dominant_entry = behavior.ngram_orders.iter()
            .max_by_key(|(_, &count)| count);
        
        let (dominant_order, total_for_order) = if let Some((&order, &count)) = dominant_entry {
            (order, count)
        } else {
            (2, 0)
        };
        
        let total: u64 = behavior.ngram_orders.values().sum();
        let confidence = if total > 0 {
            total_for_order as f64 / total as f64
        } else {
            0.0
        };
        
        // Determine sector (cluster membership)
        let sector = clusters.iter()
            .position(|c| c.members.contains(word_text))
            .unwrap_or(0);
        
        // Calculate angle within sector
        let angle = calculate_angle_in_sector(word_text, sector, behaviors, clusters);
        
        // Calculate color
        let color_obj = calculate_word_color(dominant_order, confidence);
        
        words.insert(word_text.clone(), Word {
            text: word_text.clone(),
            ring: 0,  // Will be updated by viewport
            sector,
            angle,
            x: 0.0,
            y: 0.0,
            total_count: total,
            dominant_order,
            confidence,
            color: SerializableColor::from_color(color_obj),
            size: 10.0,
            activation: 0.0,
        });
    }
    
    words
}

// ============================================================================
// RENDERING - Calculate actual screen positions
// ============================================================================

fn calculate_word_positions(
    words: &mut HashMap<String, Word>,
    viewport: &RadialViewport,
    center_x: f32,
    center_y: f32,
) {
    for (ring_idx, ring_words) in viewport.ring_contents.iter().enumerate() {
        let radius = (ring_idx + 1) as f32 * RING_SPACING;
        
        for word_text in ring_words {
            if let Some(word) = words.get_mut(word_text) {
                word.ring = ring_idx;
                word.x = center_x + radius * word.angle.cos();
                word.y = center_y + radius * word.angle.sin();
            }
        }
    }
}

fn draw_radial_grid(center_x: f32, center_y: f32, num_rings: usize) {
    // Draw rings
    for i in 1..=num_rings {
        let radius = i as f32 * RING_SPACING;
        draw_circle_lines(center_x, center_y, radius, 1.0, Color::new(0.2, 0.2, 0.2, 0.5));
    }
}

fn draw_semantic_cells(
    cells: &[SemanticCell],
    words: &HashMap<String, Word>,
    _center_x: f32,
    _center_y: f32,
) {
    for cell in cells {
        // Get positions of words in this n-gram
        let positions: Vec<(f32, f32)> = cell.words.iter()
            .filter_map(|w| words.get(w))
            .map(|word| (word.x, word.y))
            .collect();
        
        if positions.len() >= 2 {
            // Draw polygon
            let fill_color = cell.get_fill_color();
            
            // Draw filled polygon (simplified - just draw triangles from first point)
            for i in 1..positions.len()-1 {
                draw_triangle(
                    Vec2::new(positions[0].0, positions[0].1),
                    Vec2::new(positions[i].0, positions[i].1),
                    Vec2::new(positions[i+1].0, positions[i+1].1),
                    fill_color,
                );
            }
            
            // Draw edges with word colors
            let edge_colors = cell.get_edge_colors();
            for i in 0..positions.len() {
                let next_i = (i + 1) % positions.len();
                let color = edge_colors.get(i).copied().unwrap_or(WHITE);
                draw_line(
                    positions[i].0, positions[i].1,
                    positions[next_i].0, positions[next_i].1,
                    2.0,
                    color,
                );
            }
        }
    }
}

fn draw_words(
    words: &HashMap<String, Word>,
    viewport: &RadialViewport,
) {
    for ring_words in &viewport.ring_contents {
        for word_text in ring_words {
            if let Some(word) = words.get(word_text) {
                let color = word.get_color();
                
                // Draw word point
                draw_circle(word.x, word.y, word.size / 2.0, color);
                
                // Draw label
                let params = TextParams {
                    font_size: 16,
                    color,
                    ..Default::default()
                };
                draw_text_ex(&word.text, word.x + word.size, word.y, params);
            }
        }
    }
}

fn draw_ui(_state: &M2State, query: &str, status_text: &str) {
    // Query box
    let box_height = 40.0;
    let box_y = screen_height() - box_height - 10.0;
    draw_rectangle(10.0, box_y, 400.0, box_height, Color::new(0.1, 0.1, 0.1, 0.9));
    draw_text(&format!("Query: {}", query), 15.0, box_y + 25.0, 20.0, WHITE);
    
    // Status
    draw_text(status_text, 10.0, screen_height() - 5.0, 16.0, CYAN);
    
    // Instructions
    draw_text("L=Load | S=Save | E=Export N-grams", 10.0, 20.0, 16.0, WHITE);
}

// ============================================================================
// INLINE HIGHLIGHTING
// ============================================================================

fn highlight_text_with_ngram_colors(
    text: &str,
    words: &HashMap<String, Word>,
    x: f32,
    y: f32,
) {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut current_x = x;
    
    for token in tokens {
        let token_lower = token.to_lowercase();
        let color = if let Some(word) = words.get(&token_lower) {
            word.get_color()
        } else {
            WHITE
        };
        
        draw_text(token, current_x, y, 20.0, color);
        current_x += measure_text(token, None, 20, 1.0).width + 8.0;
    }
}

// ============================================================================
// MAIN LOOP
// ============================================================================

#[macroquad::main("M2 v2 - Radial Semantic Memory")]
async fn main() {
    // Create data directories
    let data_dir = match ensure_data_directories() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Failed to create data directories: {}", e);
            return;
        }
    };
    
    let mut state = M2State::new();
    let mut words: HashMap<String, Word> = HashMap::new();
    let mut viewport = RadialViewport::new();
    let mut semantic_cells: Vec<SemanticCell> = Vec::new();
    
    let mut query = String::new();
    let mut status = String::from("Ready. L=Load | S=Save | E=Export N-grams");
    let mut input_text = String::new();
    
    const NGRAM_MEMORY_THRESHOLD: usize = 50000;  // Auto-export if exceeding 50k n-grams
    
    loop {
        clear_background(BLACK);
        
        let center_x = screen_width() / 2.0;
        let center_y = screen_height() / 2.0;
        
        // ====== INPUT HANDLING ======
        
        // File operations
        if is_key_pressed(KeyCode::L) {
            if let Some(file_path) = FileDialog::new()
                .add_filter("JSON", &["json"])
                .add_filter("Binary", &["bin"])
                .pick_file()
            {
                let result = if file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                    M2State::load_json(&file_path)
                } else {
                    M2State::load_binary(&file_path)
                };
                
                match result {
                    Ok(loaded_state) => {
                        state = loaded_state;
                        words = build_words_from_behaviors(&state.word_behaviors, &state.clusters);
                        status = format!("Loaded: {} ({} n-grams)", file_path.display(), state.ngrams.len());
                    }
                    Err(e) => {
                        status = format!("Load error: {}", e);
                    }
                }
            }
        }
        
        if is_key_pressed(KeyCode::S) {
            let timestamp = Utc::now().format("%Y-%m-%d_%H-%M-%S");
            let default_name = format!("m2_state_{}.json", timestamp);
            
            if let Some(file_path) = FileDialog::new()
                .add_filter("JSON", &["json"])
                .add_filter("Binary", &["bin"])
                .set_file_name(&default_name)
                .save_file()
            {
                let result = if file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                    state.save_json(&file_path)
                } else {
                    state.save_binary(&file_path)
                };
                
                match result {
                    Ok(_) => status = format!("Saved: {}", file_path.display()),
                    Err(e) => status = format!("Save error: {}", e),
                }
            }
        }
        
        // Export n-grams (E key)
        if is_key_pressed(KeyCode::E) {
            let timestamp = Utc::now().format("%Y-%m-%d_%H-%M-%S");
            let export_path = data_dir.join("ngrams").join(format!("ngrams_{}.jsonl", timestamp));
            
            match export_ngrams(&state.ngrams, &export_path) {
                Ok(_) => status = format!("Exported {} n-grams to {}", state.ngrams.len(), export_path.display()),
                Err(e) => status = format!("Export error: {}", e),
            }
        }
        
        // Auto-export if memory threshold reached
        if let Err(e) = auto_export_if_needed(&state.ngrams, &data_dir, NGRAM_MEMORY_THRESHOLD) {
            status = format!("Auto-export error: {}", e);
        }
        
        // Query input
        if let Some(character) = get_char_pressed() {
            if character.is_alphanumeric() || character == ' ' {
                query.push(character);
            }
        }
        
        if is_key_pressed(KeyCode::Backspace) {
            query.pop();
        }
        
        if is_key_pressed(KeyCode::Enter) && !query.is_empty() {
            // Update viewport based on query
            viewport.update_from_query(&query, &words, &state.word_behaviors, &state.clusters);
            calculate_word_positions(&mut words, &viewport, center_x, center_y);
            semantic_cells = generate_semantic_cells(&viewport, &state.ngrams, &words);
            status = format!("Query: {} ({} cells, {} n-grams total)", query, semantic_cells.len(), state.ngrams.len());
        }
        
        // Text input for building ngrams (Ctrl+T)
        if is_key_down(KeyCode::LeftControl) && is_key_pressed(KeyCode::T) {
            input_text = String::from("Type text and press Enter to add to corpus...");
        }
        
        // ====== RENDERING ======
        
        draw_radial_grid(center_x, center_y, 5);
        draw_semantic_cells(&semantic_cells, &words, center_x, center_y);
        draw_words(&words, &viewport);
        draw_ui(&state, &query, &status);
        
        // Show sample highlighted text if words exist
        if !words.is_empty() {
            let sample = "The quick brown fox jumps over the lazy dog";
            highlight_text_with_ngram_colors(sample, &words, 10.0, screen_height() / 2.0);
        }
        
        next_frame().await;
    }
}