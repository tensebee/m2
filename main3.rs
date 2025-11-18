// =====================================================================
//  SECTION 1 — IMPORTS, CONSTANTS, CORE TYPES
//
//  This file is intended to be a full `main.rs` once all sections
//  are assembled in order. Each section is self-contained and
//  documented.
//
//  This section defines:
//    • Imports
//    • Color constants
//    • Global configuration constants
//    • Core data types (WordNode, WordBatchIndex)
// =====================================================================

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};

use macroquad::prelude::*;
use rfd::FileDialog;
use serde_json::Value;

// ---------------------------------------------------------
// Color palette for the UI
// ---------------------------------------------------------
const BLACK:   Color = Color::new(0.0, 0.0, 0.0, 1.0);
const GREEN:   Color = Color::new(0.0, 1.0, 0.0, 1.0);
const CYAN:    Color = Color::new(0.0, 1.0, 1.0, 1.0);
const YELLOW:  Color = Color::new(1.0, 1.0, 0.0, 1.0);
const ORANGE:  Color = Color::new(1.0, 0.5, 0.0, 1.0);
const WHITE:   Color = Color::new(1.0, 1.0, 1.0, 1.0);
const MAGENTA: Color = Color::new(1.0, 0.0, 1.0, 1.0);
const RED:     Color = Color::new(1.0, 0.0, 0.0, 1.0);
const BLUE:    Color = Color::new(0.3, 0.3, 1.0, 1.0);

// ---------------------------------------------------------
// Global configuration knobs
// ---------------------------------------------------------

// Hard cap on how many words we render each frame.
// This is the TOP-K limit and is applied after sorting
// by activation/frequency.
pub const MAX_WORDS_RENDERED: usize = 600;

// How many words go into a single on-disk batch.
// Adjust this up/down depending on RAM limits.
// Typical range: 2_000 – 10_000.
pub const WORD_BATCH_SIZE: usize = 5_000;

// Folder where word batches are stored as JSON files.
pub const WORD_BATCH_FOLDER: &str = "./word_batches";

// ---------------------------------------------------------
// Core data types
// ---------------------------------------------------------

// A single word instance in world space.
// All visualization logic builds on this.
#[derive(Clone, Debug)]
pub struct WordNode {
    pub text: String,    // The word itself
    pub x: f32,          // World-space X
    pub y: f32,          // World-space Y
    pub activation: f32, // 0.0–1.0, used for intensity
    pub frequency: f32,  // Optional: frequency / importance
}

// Index of all known batches on disk.
// Maps batch_number → filename.
#[derive(Clone, Debug)]
pub struct WordBatchIndex {
    pub batches: HashMap<usize, String>,
}
// =====================================================================
//  SECTION 2 — CONTROLS MAP (WITH LEFT-CTRL MODIFIER)
//
//  All global actions require LEFT CTRL to prevent interfering with
//  text input during query mode or accidental keypresses.
//
//  Query mode specifically *ignores* the CTRL requirement so the user
//  can type normally.
//
//  Camera controls (mouse) remain unaffected.
// =====================================================================

use macroquad::prelude::{KeyCode, MouseButton};

// ---------------------------
// Modifier key (required for actions)
// ---------------------------
pub const MOD_GLOBAL_ACTION: KeyCode = KeyCode::LeftControl;

// ---------------------------
// Mouse bindings
// ---------------------------

// Drag to pan camera
pub const MOUSE_BUTTON_PAN: MouseButton    = MouseButton::Right;

// Click to select words
pub const MOUSE_BUTTON_SELECT: MouseButton = MouseButton::Left;

// ---------------------------
// Keyboard bindings (must be used WITH MOD_GLOBAL_ACTION)
// ---------------------------

// ==== Corpus & batch management (Ctrl + L, Ctrl + ←, Ctrl + →) ====
pub const KEY_LOAD_CORPUS: KeyCode   = KeyCode::L;
pub const KEY_PREV_BATCH: KeyCode    = KeyCode::Left;
pub const KEY_NEXT_BATCH: KeyCode    = KeyCode::Right;

// ==== View & system toggles (Ctrl + Tab, Ctrl + H) ====
pub const KEY_TOGGLE_VIEW: KeyCode   = KeyCode::Tab;
pub const KEY_TOGGLE_HELP: KeyCode   = KeyCode::H;

// ---------------------------
// Query mode keys (NO Ctrl modifier)
// ---------------------------

// Enter/exit query mode
pub const KEY_TOGGLE_QUERY: KeyCode  = KeyCode::Slash;  // '/'

// Confirm query
pub const KEY_SUBMIT_QUERY: KeyCode  = KeyCode::Enter;

// Cancel/clear
pub const KEY_CANCEL_QUERY: KeyCode  = KeyCode::Escape;

// =====================================================================
//  SECTION 3 — MOUSE CAMERA (PAN + ZOOM)
//
//  Responsibilities:
//    • Hold camera position and zoom
//    • Convert between world and screen coordinates
//    • Update based on mouse input only:
//         - MOUSE_BUTTON_PAN drag  → pan
//         - mouse wheel            → zoom centered on cursor
//
//  No keyboard handling here — all keyboard logic lives in AppState
//  and uses the controls defined in SECTION 2.
// =====================================================================

#[derive(Debug)]
pub struct Camera {
    /// Camera center in world space.
    pub x: f32,
    pub y: f32,

    /// Zoom factor (1.0 = default scale).
    pub zoom: f32,

    // Internal mouse state for panning.
    dragging: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
}

impl Camera {
    /// Create a new camera centered at (0,0) with default zoom.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            zoom: 1.0,
            dragging: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
        }
    }

    /// Convert world-space coordinates to screen-space coordinates.
    /// Use this for drawing words/nodes.
    pub fn world_to_screen(&self, wx: f32, wy: f32) -> (f32, f32) {
        let cx = screen_width() * 0.5;
        let cy = screen_height() * 0.5;

        let sx = cx + (wx - self.x) * self.zoom;
        let sy = cy + (wy - self.y) * self.zoom;

        (sx, sy)
    }

    /// Convert screen-space coordinates to world-space.
    /// Use this for hit-testing / clicking on words.
    pub fn screen_to_world(&self, sx: f32, sy: f32) -> (f32, f32) {
        let cx = screen_width() * 0.5;
        let cy = screen_height() * 0.5;

        let wx = (sx - cx) / self.zoom + self.x;
        let wy = (sy - cy) / self.zoom + self.y;

        (wx, wy)
    }

    /// Update camera based on mouse input.
    ///
    /// - Hold and drag MOUSE_BUTTON_PAN to move the camera.
    /// - Use the mouse wheel to zoom in/out, centered on the cursor.
    pub fn update_from_mouse(&mut self) {
        let (mx, my) = mouse_position();

        // ----- Start drag -----
        if is_mouse_button_pressed(MOUSE_BUTTON_PAN) {
            self.dragging = true;
            self.last_mouse_x = mx;
            self.last_mouse_y = my;
        }

        // ----- End drag -----
        if is_mouse_button_released(MOUSE_BUTTON_PAN) {
            self.dragging = false;
        }

        // ----- Drag movement -----
        if self.dragging {
            let dx = mx - self.last_mouse_x;
            let dy = my - self.last_mouse_y;

            // Move camera opposite to mouse movement.
            // Divide by zoom so panning feels consistent at all scales.
            self.x -= dx / self.zoom;
            self.y -= dy / self.zoom;

            self.last_mouse_x = mx;
            self.last_mouse_y = my;
        }

        // ----- Scroll zoom -----
        let scroll = mouse_wheel().1; // (x, y) → use y axis
        if scroll != 0.0 {
            // Exponential zoom factor for smooth feel.
            let zoom_factor = 1.05_f32.powf(scroll);

            // 1) world position under cursor BEFORE zoom
            let (before_wx, before_wy) = self.screen_to_world(mx, my);

            // 2) apply zoom with limits
            self.zoom = (self.zoom * zoom_factor).clamp(0.1, 8.0);

            // 3) world position under cursor AFTER zoom
            let (after_wx, after_wy) = self.screen_to_world(mx, my);

            // 4) adjust camera so that same world point stays under the cursor
            self.x += before_wx - after_wx;
            self.y += before_wy - after_wy;
        }
    }
}

// =====================================================================
//  SECTION 4 — SPIRAL LAYOUT (PORTED FROM main1.rs)
//
//  This restores the original spiral placement logic:
//      theta = index * 2.0          (radians)
//      r     = 20.0 + index * 0.25
//      x     = cos(theta) * r
//      y     = sin(theta) * r
//
//  We treat "index" as a global word index, so that if we ever append
//  new words later, the spiral can continue smoothly from where it left
//  off.
//
//  This section DOES NOT know about batches or files yet.
//  It only sets (x, y) on WordNode according to the old behavior.
// =====================================================================

/// Compute the spiral position for a given global index.
///
/// This is a direct port of the old main1.rs logic:
///   theta = index * 2.0
///   r     = 20.0 + index * 0.25
///   x     = cos(theta) * r
///   y     = sin(theta) * r
pub fn spiral_position_for_index(global_index: usize) -> (f32, f32) {
    let i = global_index as f32;

    let theta = i * 2.0;           // angle in radians
    let r = 20.0 + i * 0.25;       // radius grows slowly outward

    let x = theta.cos() * r;
    let y = theta.sin() * r;

    (x, y)
}

/// Lay out a contiguous slice of WordNode on the spiral,
/// starting at a given global starting index.
///
/// Example:
///   - If you already have 10_000 words.
///   - You load 500 new ones.
///   - Call layout_words_spiral_segment(&mut new_words, 10_000).
///
/// For now, in our simpler version, we'll usually call it with
/// start_index = 0 for "all words".
pub fn layout_words_spiral_segment(words: &mut [WordNode], start_index: usize) {
    for (offset, node) in words.iter_mut().enumerate() {
        let global_index = start_index + offset;
        let (x, y) = spiral_position_for_index(global_index);
        node.x = x;
        node.y = y;
    }
}
// =====================================================================
//  SECTION 5 — WORD RENDERING (GEOMETRY, TOP-K, HIT TESTING)
//
//  This section is responsible for:
//    • Sorting words by importance (activation / frequency)
//    • Applying a TOP-K cap for performance
//    • Transforming spiral world positions → screen positions
//    • Drawing text at those positions
//    • Detecting mouse clicks on words
//
//  NOTE: Color is delegated to `color_for_word(&WordNode)`, which will
//  be defined later in the COLOR section. Here we only care about
//  *where* and *how* to draw, not how to color.
// =====================================================================

/// Whether font size should scale with zoom level.
/// If true, zooming in also makes text larger; if false, text remains
/// the same size regardless of zoom.
pub const SCALE_TEXT_WITH_ZOOM: bool = true;

/// Extract a TOP-K view of words, sorted by "importance".
///
/// Primary key: activation (descending)
/// Secondary:   frequency (descending)
pub fn get_top_k_words<'a>(words: &'a [WordNode]) -> Vec<&'a WordNode> {
    let mut v: Vec<&WordNode> = words.iter().collect();

    v.sort_by(|a, b| {
        b.activation
            .partial_cmp(&a.activation)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.frequency
                    .partial_cmp(&a.frequency)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    v.truncate(MAX_WORDS_RENDERED);
    v
}

/// Draw a single word node at the correct screen-space position.
///
/// Geometry + text layout only:
///   - transforms world → screen using Camera
///   - chooses font size
///   - chooses color via `color_for_word(word)` (defined later)
pub fn draw_word_node(word: &WordNode, cam: &Camera) {
    let (sx, sy) = cam.world_to_screen(word.x, word.y);

    // Decide font size (optionally zoom-aware)
    let font_size = if SCALE_TEXT_WITH_ZOOM {
        (20.0 * cam.zoom.clamp(0.5, 3.0)) as u16
    } else {
        20
    };

    // Color comes from context-dependent logic, defined later.
    let color = color_for_word(word);

    draw_text_ex(
        &word.text,
        sx,
        sy,
        TextParams {
            font_size,
            color,
            ..Default::default()
        },
    );
}

/// Detect whether the user just clicked on any of the given words.
///
/// Uses:
///   - MOUSE_BUTTON_SELECT from SECTION 2
///   - Camera for world→screen mapping
///
/// Returns a reference to the clicked word, if any.
pub fn detect_word_click<'a>(
    words: &'a [WordNode],
    cam: &Camera,
) -> Option<&'a WordNode> {
    // Only react on the *moment* the button is pressed, not held.
    if !is_mouse_button_pressed(MOUSE_BUTTON_SELECT) {
        return None;
    }

    let (mx, my) = mouse_position();

    for word in words.iter() {
        let (sx, sy) = cam.world_to_screen(word.x, word.y);

        // Match font size logic used in draw_word_node.
        let font_size = if SCALE_TEXT_WITH_ZOOM {
            (20.0 * cam.zoom.clamp(0.5, 3.0)) as f32
        } else {
            20.0
        };

        // Very simple text bounds approximation:
        // width ~ half font_size * text length
        let w = font_size * (word.text.len() as f32 * 0.5);
        let h = font_size;

        let left = sx;
        let top = sy - h;
        let right = sx + w;
        let bottom = sy;

        if mx >= left && mx <= right && my >= top && my <= bottom {
            return Some(word);
        }
    }

    None
}

/// Primary entry point for rendering ALL words each frame.
///
/// Responsibilities:
///   • Update camera from mouse
///   • Choose TOP-K words
///   • Draw them
///   • Detect a click and return the selected word (cloned)
pub fn render_words(words: &[WordNode], cam: &mut Camera) -> Option<WordNode> {
    // Update camera based on mouse first.
    cam.update_from_mouse();

    // Get the most "important" words.
    let top_words = get_top_k_words(words);

    // Draw them.
    for w in &top_words {
        draw_word_node(w, cam);
    }

    // Handle click selection on the full list (not just top-k, but you
    // could change to &top_words if you prefer).
    if let Some(selected) = detect_word_click(words, cam) {
        return Some(selected.clone());
    }

    None
}
// =====================================================================
//  SECTION 6 — COLOR / HSV FROM CONTEXT
//
//  This section defines how we color words based on their context:
//
//    Inputs:
//      • word.x, word.y       (position on spiral)
//      • word.activation      (0.0 – 1.0)
//      • word.frequency       (0.0+)
//
//    Outputs:
//      • macroquad::Color
//
//  Strategy:
//    • Hue (H)    ← angular position via atan2(y, x)
//    • Saturation ← function of activation
//    • Value (V)  ← mix of activation and frequency
//
//  All drawing code calls `color_for_word(&WordNode)` from SECTION 5.
// =====================================================================

/// Convert HSV to RGB (macroquad Color).
///
/// h: 0..360 (degrees)
/// s: 0..1
/// v: 0..1
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let c = v * s;
    let hh = h / 60.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());

    let (r1, g1, b1) = if hh < 1.0 {
        (c, x, 0.0)
    } else if hh < 2.0 {
        (x, c, 0.0)
    } else if hh < 3.0 {
        (0.0, c, x)
    } else if hh < 4.0 {
        (0.0, x, c)
    } else if hh < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;
    Color::new(r1 + m, g1 + m, b1 + m, 1.0)
}

/// Compute an HSV triple from a word's spatial + activation context.
///
///   - Hue from angle: atan2(y, x) in world space → 0..360 deg
///   - Saturation from activation: low activation → washed-out, high → vivid
///   - Value from activation & log-frequency: bright for active + common words
pub fn hsv_from_word_context(word: &WordNode) -> (f32, f32, f32) {
    // --- Hue from position ---
    // Use the angle on the spiral so regions have stable colors.
    let ang_rad = word.y.atan2(word.x);         // -π..π
    let mut h = ang_rad.to_degrees();           // -180..180
    if h < 0.0 {
        h += 360.0;                              // 0..360
    }

    // --- Normalize activation ---
    // Clamp to sane range to avoid weird values from upstream.
    let act = word.activation.clamp(0.0, 1.0);

    // --- Approximate frequency scaling ---
    // Simple log-ish scaling: low frequency → small boost, high → more.
    // We assume frequency >= 0 (or at least not insane negative).
    let freq_base = word.frequency.max(0.0);
    let freq_factor = if freq_base <= 1.0 {
        0.0
    } else {
        (freq_base.ln() / 5.0).clamp(0.0, 1.0)  // squashed to 0..1-ish
    };

    // --- Saturation ---
    // Keep a baseline saturation so even low-activation words have some color,
    // but let activation dominate.
    let s = (0.3 + act * 0.7).clamp(0.0, 1.0);

    // --- Value (brightness) ---
    // Mix activation and frequency factor:
    //   - activation: immediate salience
    //   - freq_factor: global importance
    let v_raw = 0.4 + act * 0.4 + freq_factor * 0.3;
    let v = v_raw.clamp(0.0, 1.0);

    (h, s, v)
}

/// Main entry point for coloring a word.
///
/// Used by the rendering layer:
///   • SECTION 5 calls this inside draw_word_node().
pub fn color_for_word(word: &WordNode) -> Color {
    let (h, s, v) = hsv_from_word_context(word);
    hsv_to_rgb(h, s, v)
}
// =====================================================================
//  SECTION 7 — SEMANTIC LAYER (N-GRAM CONNECTIONS)
//
//  Concept:
//    • Semantic "cells" are realized here as edges between WordNode
//      vertices, derived from n-gram statistics.
//    • Each edge represents an n-gram connection:
//          from_word → to_word
//      with brightness reflecting the confidence of that n-gram.
//
//  Rendering:
//    • Uses the *existing* word positions (spiral) and camera.
//    • Draws lines between the words.
//    • Brightness (and alpha) of the line is based on confidence.
//
//  Query behavior:
//    • By default, ALL edges are visible.
//    • Later, the AppState/query layer can pass filters to only
//      highlight edges relevant to a query.
// =====================================================================

/// Single n-gram connection:
///   `from` → `to`
/// with associated confidence (0.0–1.0) and optional n-gram length.
#[derive(Clone, Debug)]
pub struct NGramEdge {
    pub from: String,      // source word text
    pub to: String,        // target word text
    pub confidence: f32,   // 0.0–1.0, brightness driver
    pub n: usize,          // n-gram length (e.g. 2 for bigram, 3 for trigram)
}

/// Collection of n-gram edges comprising the semantic layer.
#[derive(Clone, Debug)]
pub struct SemanticLayer {
    pub edges: Vec<NGramEdge>,
}

impl SemanticLayer {
    /// Create an empty semantic layer.
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add a single n-gram edge to the layer.
    pub fn add_edge(&mut self, from: String, to: String, confidence: f32, n: usize) {
        self.edges.push(NGramEdge {
            from,
            to,
            confidence,
            n,
        });
    }

    /// Bulk load edges from any iterator of (from, to, confidence, n).
    pub fn extend_from_iter<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, String, f32, usize)>,
    {
        for (from, to, conf, n) in iter {
            self.add_edge(from, to, conf, n);
        }
    }
}

// ---------------------------------------------------------------------
// Edge -> Color mapping
//
// Brightness (and alpha) reflect confidence:
//   • Low confidence  → dim, faint lines
//   • High confidence → bright, stronger lines
//
// Hue here is neutral-ish (cyan) by default, but you could later
// encode different n-gram lengths or types using hue if you want.
// ---------------------------------------------------------------------

/// Map an NGramEdge to a Color, based on its confidence.
/// For now we fix hue-ish region and vary value + alpha.
pub fn color_for_edge(edge: &NGramEdge) -> Color {
    // Clamp confidence to [0,1]
    let c = edge.confidence.clamp(0.0, 1.0);

    // Base value and alpha scaled by confidence
    let value = 0.2 + c * 0.8;   // 0.2..1.0
    let alpha = 0.1 + c * 0.9;   // 0.1..1.0

    // Slight tint: cyan/blue-ish, but we could later vary hue by n.
    let r = 0.2 * value;
    let g = 0.9 * value;
    let b = 1.0 * value;

    Color::new(r, g, b, alpha)
}

// ---------------------------------------------------------------------
// Rendering semantic layer
//
// We need to map from word text -> position, using the WordNode world
// coordinates (spiral positions).
// ---------------------------------------------------------------------

/// Render *all* n-gram edges in the semantic layer.
///
/// - `layer` : the semantic connections
/// - `words` : the current word nodes (same batch you're rendering text for)
/// - `cam`   : the camera (for world→screen mapping)
///
/// This default version draws *all edges*; later we can add query-based
/// filtering at a higher layer, by passing in filtered edges instead.
pub fn render_semantic_layer(
    layer: &SemanticLayer,
    words: &[WordNode],
    cam: &Camera,
) {
    // Build a quick lookup from word text → (x, y).
    // NOTE: For large batches you might want to reuse this map instead
    // of rebuilding each frame, but this is clean for now.
    let mut pos_map: HashMap<&str, (f32, f32)> = HashMap::with_capacity(words.len());
    for w in words {
        pos_map.insert(w.text.as_str(), (w.x, w.y));
    }

    // Draw each edge as a line between the corresponding words.
    for edge in &layer.edges {
        let (from_pos, to_pos) = match (
            pos_map.get(edge.from.as_str()),
            pos_map.get(edge.to.as_str()),
        ) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue, // skip edges whose words aren't in the current batch
        };

        // World → screen
        let (sx1, sy1) = cam.world_to_screen(from_pos.0, from_pos.1);
        let (sx2, sy2) = cam.world_to_screen(to_pos.0, to_pos.1);

        let color = color_for_edge(edge);

        // Thin lines; you can parameterize thickness later.
        let thickness = 1.0_f32;
        draw_line(sx1, sy1, sx2, sy2, thickness, color);
    }
}
// =====================================================================
//  SECTION 8 — DISK PERSISTENCE (WORDS + SEMANTIC LAYER)
//
//  Responsibilities:
//    • Ensure the batch folder exists
//    • Save/load word batches as JSON
//    • Build batches from a large Vec<WordNode>
//    • Index available word batches on disk
//    • Save/load the SemanticLayer (all NGramEdge entries)
//
//  File layout:
//
//    WORD_BATCH_FOLDER/
//        batch_00000.json
//        batch_00001.json
//        ...
//        semantic_edges.json
//
//  All functions return Result<...> so the AppState layer can decide
//  how to handle errors without panicking.
// =====================================================================

/// Make sure the batch folder exists.
///
/// This is called by all save/load functions.
pub fn ensure_batch_folder() -> Result<(), String> {
    if let Err(e) = fs::create_dir_all(WORD_BATCH_FOLDER) {
        return Err(format!("Failed to create batch folder '{}': {}", WORD_BATCH_FOLDER, e));
    }
    Ok(())
}

/// Save a single batch of WordNode entries to disk.
///
/// The filename is:
///   WORD_BATCH_FOLDER/batch_{:05}.json
pub fn save_word_batch(batch_num: usize, words: &[WordNode]) -> Result<String, String> {
    ensure_batch_folder()?;

    let filename = format!("{}/batch_{:05}.json", WORD_BATCH_FOLDER, batch_num);

    // Represent each WordNode as a JSON object.
    let json_vec: Vec<Value> = words
        .iter()
        .map(|w| {
            serde_json::json!({
                "text": w.text,
                "x": w.x,
                "y": w.y,
                "activation": w.activation,
                "frequency": w.frequency
            })
        })
        .collect();

    let json_string = serde_json::to_string_pretty(&json_vec)
        .map_err(|e| format!("JSON serialization error for {}: {}", filename, e))?;

    fs::write(&filename, json_string)
        .map_err(|e| format!("Failed to write '{}': {}", filename, e))?;

    Ok(filename)
}

/// Load a single batch of WordNode entries from disk.
///
/// Expects the file:
///   WORD_BATCH_FOLDER/batch_{:05}.json
pub fn load_word_batch(batch_num: usize) -> Result<Vec<WordNode>, String> {
    let filename = format!("{}/batch_{:05}.json", WORD_BATCH_FOLDER, batch_num);

    let data = fs::read_to_string(&filename)
        .map_err(|e| format!("Failed to read '{}': {}", filename, e))?;

    let parsed: Vec<Value> = serde_json::from_str(&data)
        .map_err(|e| format!("JSON parse error in '{}': {}", filename, e))?;

    let mut nodes = Vec::with_capacity(parsed.len());

    for v in parsed {
        if let (Some(text), Some(x), Some(y), Some(act), Some(freq)) = (
            v.get("text").and_then(|x| x.as_str()),
            v.get("x").and_then(|x| x.as_f64()),
            v.get("y").and_then(|x| x.as_f64()),
            v.get("activation").and_then(|x| x.as_f64()),
            v.get("frequency").and_then(|x| x.as_f64()),
        ) {
            nodes.push(WordNode {
                text: text.to_string(),
                x: x as f32,
                y: y as f32,
                activation: act as f32,
                frequency: freq as f32,
            });
        }
    }

    Ok(nodes)
}

/// Build all word batches from a flat list of WordNode.
///
/// Splits into chunks of WORD_BATCH_SIZE and calls save_word_batch().
///
/// Returns the number of batches created.
pub fn build_batches_from_words(words: &[WordNode]) -> Result<usize, String> {
    ensure_batch_folder()?;

    let mut batch_num = 0;
    let mut index = 0;

    while index < words.len() {
        let end = usize::min(index + WORD_BATCH_SIZE, words.len());
        let slice = &words[index..end];

        save_word_batch(batch_num, slice)?;
        batch_num += 1;
        index = end;
    }

    println!(
        "Disk persistence: Saved {} total batches (size ~{} each).",
        batch_num,
        WORD_BATCH_SIZE
    );

    Ok(batch_num)
}

// ---------------------------------------------------------------------
// WordBatchIndex implementation
// ---------------------------------------------------------------------

impl WordBatchIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            batches: HashMap::new(),
        }
    }

    /// Scan the batch folder and construct an index of available word batches.
    ///
    /// It looks for files named:
    ///   batch_XXXXX.json
    /// and extracts the numeric index XXXXX.
    pub fn load_from_folder() -> Result<Self, String> {
        ensure_batch_folder()?;

        let mut idx = WordBatchIndex::new();

        let entries = fs::read_dir(WORD_BATCH_FOLDER)
            .map_err(|e| format!("Failed to read batch directory '{}': {}", WORD_BATCH_FOLDER, e))?;

        for entry in entries {
            if let Ok(entry) = entry {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("batch_") && name.ends_with(".json") {
                    if let Some(num) = extract_batch_number(&name) {
                        idx.batches.insert(num, name);
                    }
                }
            }
        }

        Ok(idx)
    }
}

/// Extract the batch number from a filename like "batch_00012.json".
pub fn extract_batch_number(filename: &str) -> Option<usize> {
    let stripped = filename
        .trim_start_matches("batch_")
        .trim_end_matches(".json");
    stripped.parse().ok()
}

// ---------------------------------------------------------------------
// Semantic layer persistence
// ---------------------------------------------------------------------

/// File where we store ALL semantic edges (for now).
///
/// You can later switch to per-batch edge files without touching the
/// rest of the code, as long as you keep the same load/save signatures.
pub const SEMANTIC_EDGES_FILE: &str = "semantic_edges.json";

/// Save the entire SemanticLayer to disk as JSON.
///
/// The file will live at:
///   WORD_BATCH_FOLDER/semantic_edges.json
pub fn save_semantic_layer(layer: &SemanticLayer) -> Result<String, String> {
    ensure_batch_folder()?;

    let filename = format!("{}/{}", WORD_BATCH_FOLDER, SEMANTIC_EDGES_FILE);

    let json_vec: Vec<Value> = layer
        .edges
        .iter()
        .map(|e| {
            serde_json::json!({
                "from":       e.from,
                "to":         e.to,
                "confidence": e.confidence,
                "n":          e.n
            })
        })
        .collect();

    let json_string = serde_json::to_string_pretty(&json_vec)
        .map_err(|e| format!("JSON serialization error for semantic_edges: {}", e))?;

    fs::write(&filename, json_string)
        .map_err(|e| format!("Failed to write '{}': {}", filename, e))?;

    Ok(filename)
}

/// Load the SemanticLayer (all edges) from disk.
///
/// If the file does not exist, returns an empty layer instead of failing.
pub fn load_semantic_layer() -> Result<SemanticLayer, String> {
    ensure_batch_folder()?;

    let filename = format!("{}/{}", WORD_BATCH_FOLDER, SEMANTIC_EDGES_FILE);

    let data = match fs::read_to_string(&filename) {
        Ok(d) => d,
        Err(_) => {
            // No semantic_edges file yet → treat as empty semantic layer.
            println!(
                "Semantic persistence: no '{}' found; starting with empty layer.",
                filename
            );
            return Ok(SemanticLayer::new());
        }
    };

    let parsed: Vec<Value> = serde_json::from_str(&data)
        .map_err(|e| format!("JSON parse error in '{}': {}", filename, e))?;

    let mut layer = SemanticLayer::new();

    for v in parsed {
        if let (Some(from), Some(to), Some(conf), Some(n)) = (
            v.get("from").and_then(|x| x.as_str()),
            v.get("to").and_then(|x| x.as_str()),
            v.get("confidence").and_then(|x| x.as_f64()),
            v.get("n").and_then(|x| x.as_u64()),
        ) {
            layer.edges.push(NGramEdge {
                from: from.to_string(),
                to: to.to_string(),
                confidence: conf as f32,
                n: n as usize,
            });
        }
    }

    Ok(layer)
}
// =====================================================================
//  SECTION 9 — APP STATE & RUNTIME WIRING
//
//  This section ties together:
//    • Camera
//    • Active word batch
//    • Word batch index
//    • SemanticLayer (n-gram edges)
//    • Global controls (Ctrl+L, Ctrl+←/→, Ctrl+Tab, Ctrl+H)
//
//  It exposes a single per-frame entry point:
//
//      app.frame()
//
//  which:
//      - handles keyboard shortcuts (with LeftCtrl modifier)
//      - updates the camera
//      - renders words
//      - optionally renders the semantic layer
//      - tracks the last selected word for HUD display
// =====================================================================

/// High-level application state.
pub struct AppState {
    pub camera: Camera,

    // Word batches
    pub batch_index: WordBatchIndex,
    pub active_batch_num: usize,
    pub words: Vec<WordNode>,

    // Semantic connections
    pub semantic_layer: SemanticLayer,
    pub semantic_visible: bool, // can be toggled via Ctrl+Tab

    // UI / interaction state
    pub show_help: bool,
    pub last_selected: Option<WordNode>,
}

// =====================================================================
//  SECTION 9 HELPER — CORPUS TEXT EXTRACTION
//
//  Many of your corpora are arrays of JSON objects that might look like:
//    { "Q": "...", "A": "..." }
//    { "question": "...", "answer": "..." }
//    { "prompt": "...", "completion": "..." }
//    { "input": "...", "output": "..." }
//  or plain strings.
//
//  This helper tries multiple common keys and patterns to recover a
//  usable text string, instead of defaulting to "<?>".
// =====================================================================

fn extract_text_from_corpus_item(item: &Value) -> Option<String> {
    // Simple string case: ["hello", "world", ...]
    if let Some(s) = item.as_str() {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // Object case
    if let Some(obj) = item.as_object() {
        // 1) Direct keys we’d like to treat as the main "text"
        let primary_keys = [
            "text",
            "word",
            "token",
            "phrase",
            "content",
            "line",
        ];

        for key in &primary_keys {
            if let Some(v) = obj.get(*key) {
                if let Some(s) = v.as_str() {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        // 2) Q/A style objects: combine question + answer
        let qa_keys = [("Q", "A"), ("question", "answer")];
        for (qk, ak) in &qa_keys {
            if let (Some(qv), Some(av)) = (obj.get(*qk), obj.get(*ak)) {
                if let (Some(qs), Some(as_)) = (qv.as_str(), av.as_str()) {
                    let qt = qs.trim();
                    let at = as_.trim();
                    if !qt.is_empty() && !at.is_empty() {
                        return Some(format!("Q: {} || A: {}", qt, at));
                    }
                }
            }
        }

        // 3) Prompt/completion or input/output style
        let pair_keys = [("prompt", "completion"), ("input", "output")];
        for (ik, ok) in &pair_keys {
            if let (Some(iv), Some(ov)) = (obj.get(*ik), obj.get(*ok)) {
                if let (Some(is_), Some(os_)) = (iv.as_str(), ov.as_str()) {
                    let it = is_.trim();
                    let ot = os_.trim();
                    if !it.is_empty() && !ot.is_empty() {
                        return Some(format!("{} → {}", it, ot));
                    }
                }
            }
        }
    }

    // If we get here, we genuinely don't know how to read this item.
    None
}


impl AppState {
    /// Create a new AppState with default values.
    ///
    /// Call `init_from_disk()` afterward to load existing batches/edges,
    /// or let the user load a corpus at runtime with Ctrl+L.
    pub fn new() -> Self {
        Self {
            camera: Camera::new(),
            batch_index: WordBatchIndex::new(),
            active_batch_num: 0,
            words: Vec::new(),
            semantic_layer: SemanticLayer::new(),
            semantic_visible: true,  // show edges by default
            show_help: false,
            last_selected: None,
        }
    }

    /// Initialize from disk:
    ///   • load batch index
    ///   • load batch 0 (if any)
    ///   • load semantic layer (edges)
    ///
    /// This is safe even if no data exists yet — it won't panic.
    pub fn init_from_disk(&mut self) -> Result<(), String> {
        // Word batches
        match WordBatchIndex::load_from_folder() {
            Ok(idx) => {
                self.batch_index = idx;
            }
            Err(e) => {
                println!(
                    "Init: could not read batch directory '{}': {}. Starting with empty words.",
                    WORD_BATCH_FOLDER, e
                );
                self.batch_index = WordBatchIndex::new();
                self.words.clear();
            }
        }

        // Load batch 0 if any batches exist.
        if !self.batch_index.batches.is_empty() {
            if let Err(e) = self.switch_batch(0) {
                println!("Init: failed to load initial batch 0: {}", e);
            }
        } else {
            println!(
                "Init: no word batches found in '{}'. Use Ctrl+L to load a corpus.",
                WORD_BATCH_FOLDER
            );
        }

        // Semantic layer
        match load_semantic_layer() {
            Ok(layer) => {
                println!(
                    "Init: loaded semantic layer with {} edges.",
                    layer.edges.len()
                );
                self.semantic_layer = layer;
            }
            Err(e) => {
                println!("Init: failed to load semantic layer: {}", e);
                self.semantic_layer = SemanticLayer::new();
            }
        }

        Ok(())
    }

    /// Helper: check if a "global action" key was pressed with the
    /// LeftCtrl modifier held down.
    fn global_action_pressed(key: KeyCode) -> bool {
        is_key_down(MOD_GLOBAL_ACTION) && is_key_pressed(key)
    }

    /// Switch to a different batch (if it exists).
    pub fn switch_batch(&mut self, batch_num: usize) -> Result<(), String> {
        if !self.batch_index.batches.contains_key(&batch_num) {
            return Err(format!("Batch {} does not exist.", batch_num));
        }

        println!("AppState: switching to batch {}…", batch_num);
        self.words = load_word_batch(batch_num)?;
        self.active_batch_num = batch_num;

        println!(
            "AppState: batch {} now active ({} words).",
            batch_num,
            self.words.len()
        );

        Ok(())
    }

    /// Handle global keybindings that require the LeftCtrl modifier:
    ///   - Ctrl+L     : load corpus JSON, build batches
    ///   - Ctrl+←/→   : switch batches
    ///   - Ctrl+Tab   : toggle semantic layer visibility
    ///   - Ctrl+H     : toggle help overlay
    fn handle_global_shortcuts(&mut self) {
        // --- Load corpus (Ctrl+L) ---
        if Self::global_action_pressed(KEY_LOAD_CORPUS) {
            self.load_corpus_via_dialog();
        }

        // --- Previous batch (Ctrl+Left) ---
        if Self::global_action_pressed(KEY_PREV_BATCH) {
            if self.active_batch_num > 0 {
                if let Err(e) = self.switch_batch(self.active_batch_num - 1) {
                    println!("Error: {}", e);
                }
            }
        }

        // --- Next batch (Ctrl+Right) ---
        if Self::global_action_pressed(KEY_NEXT_BATCH) {
            let next_idx = self.active_batch_num + 1;
            if self.batch_index.batches.contains_key(&next_idx) {
                if let Err(e) = self.switch_batch(next_idx) {
                    println!("Error: {}", e);
                }
            }
        }

        // --- Toggle semantic layer visibility (Ctrl+Tab) ---
        if Self::global_action_pressed(KEY_TOGGLE_VIEW) {
            self.semantic_visible = !self.semantic_visible;
            println!("Semantic layer visibility: {}", self.semantic_visible);
        }

        // --- Toggle help overlay (Ctrl+H) ---
        if Self::global_action_pressed(KEY_TOGGLE_HELP) {
            self.show_help = !self.show_help;
        }
    }

        /// Load a corpus JSON file at runtime via file dialog, then:
    ///   • build WordNode list
    ///   • layout on the spiral (using SECTION 4)
    ///   • build batches and re-scan batch index
    ///   • switch to batch 0
    ///
    /// Expected JSON formats:
    ///   1) [ "word", "another", "more", ... ]
    ///   2) [ { "text": "...", ... }, { "Q": "...", "A": "..." }, ... ]
    pub fn load_corpus_via_dialog(&mut self) {
        println!("AppState: opening file dialog to load corpus (Ctrl+L)…");

        let dialog = FileDialog::new()
            .add_filter("JSON files", &["json"])
            .set_directory(".");

        let Some(path) = dialog.pick_file() else {
            println!("AppState: corpus load cancelled by user.");
            return;
        };

        println!("AppState: selected corpus file: {:?}", path);

        let data = match fs::read_to_string(&path) {
            Ok(d) => d,
            Err(e) => {
                println!("Error: failed to read corpus file: {}", e);
                return;
            }
        };

        let json_val: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: failed to parse JSON: {}", e);
                return;
            }
        };

        let mut nodes: Vec<WordNode> = Vec::new();

        if let Some(arr) = json_val.as_array() {
            for item in arr {
                if let Some(text) = extract_text_from_corpus_item(item) {
                    nodes.push(WordNode {
                        text,
                        x: 0.0,
                        y: 0.0,
                        activation: 0.5,
                        frequency: 1.0,
                    });
                } else {
                    // Silent skip for now; you can log if you want to debug.
                    // println!("Warning: could not extract text from item: {:?}", item);
                }
            }
        } else {
            // If the top level isn't an array, you *could* add additional logic
            // here to handle { "data": [ ... ] } style wrappers.
            println!("Error: corpus JSON must be an array at the top level.");
            return;
        }

        if nodes.is_empty() {
            println!("Warning: loaded corpus but found 0 usable text items.");
            return;
        }

        println!("AppState: corpus parsed — {} words. Laying out spiral…", nodes.len());

        // Layout all words on the spiral starting from global index 0.
        layout_words_spiral_segment(&mut nodes, 0);

        // Build batches + save to disk.
        match build_batches_from_words(&nodes) {
            Ok(num_batches) => println!("AppState: built {} batches from corpus.", num_batches),
            Err(e) => {
                println!("Error while building batches: {}", e);
                return;
            }
        }

        // Reload batch index and switch to batch 0.
        match WordBatchIndex::load_from_folder() {
            Ok(idx) => {
                self.batch_index = idx;
                if let Err(e) = self.switch_batch(0) {
                    println!("Error switching to batch 0 after corpus load: {}", e);
                }
            }
            Err(e) => {
                println!("Error reloading batch index after corpus load: {}", e);
            }
        }
    }


    /// One per-frame call that:
    ///   • handles global keybindings
    ///   • updates camera
    ///   • renders words
    ///   • renders semantic layer (if visible)
    ///   • tracks last selected word
    pub fn frame(&mut self) {
        // Handle global shortcuts that require LeftCtrl.
        self.handle_global_shortcuts();

        // Draw semantic layer first (so words are on top), if enabled.
        if self.semantic_visible && !self.words.is_empty() && !self.semantic_layer.edges.is_empty()
        {
            render_semantic_layer(&self.semantic_layer, &self.words, &self.camera);
        }

        // Render words (and update camera inside).
        let selected = render_words(&self.words, &mut self.camera);

        if let Some(w) = selected {
            self.last_selected = Some(w);
        }
    }
}
// =====================================================================
//  SECTION 10 — MAIN RUNTIME LOOP + HUD
//
//  This is the entry point for the whole program.
//
//  Responsibilities:
//    • Create AppState
//    • Initialize from disk (word batches + semantic edges)
//    • Enter Macroquad loop
//    • Clear background
//    • Let AppState render one frame (words + semantic)
//    • Draw HUD (FPS, batch info, camera, selection, help)
// =====================================================================

#[macroquad::main("M2 / M4 Word + Semantic Viewer")]
async fn main() {
    // -----------------------------
    // Create and initialize state
    // -----------------------------
    let mut app = AppState::new();

    if let Err(e) = app.init_from_disk() {
        println!("Initialization warning: {}", e);
    }

    println!("Controls:");
    println!("  Mouse: Right-drag to pan, wheel to zoom, Left-click to select word");
    println!("  Ctrl+L: Load corpus JSON and build batches");
    println!("  Ctrl+Left / Ctrl+Right: Switch word batch");
    println!("  Ctrl+Tab: Toggle semantic n-gram edges on/off");
    println!("  Ctrl+H: Toggle help overlay");

    // -----------------------------
    // Main loop
    // -----------------------------
    loop {
        // Clear background
        clear_background(BLACK);

        // Let AppState handle one frame:
        //  - global shortcuts (Ctrl+L, Ctrl+Tab, etc.)
        //  - semantic layer rendering (if enabled)
        //  - word rendering + camera + click selection
        app.frame();

        // -------------------------
        // HUD overlay (top-left)
        // -------------------------
        let fps = get_fps();
        draw_text(
            &format!("FPS: {}", fps),
            10.0,
            20.0,
            20.0,
            GREEN,
        );

        draw_text(
            &format!(
                "Batch: {}   ({} words)",
                app.active_batch_num,
                app.words.len()
            ),
            10.0,
            40.0,
            20.0,
            CYAN,
        );

        draw_text(
            &format!(
                "Camera: x={:.1} y={:.1} zoom={:.2}",
                app.camera.x,
                app.camera.y,
                app.camera.zoom
            ),
            10.0,
            60.0,
            20.0,
            YELLOW,
        );

        draw_text(
            &format!(
                "Semantic layer: {} ({} edges)",
                if app.semantic_visible { "ON" } else { "OFF" },
                app.semantic_layer.edges.len()
            ),
            10.0,
            80.0,
            20.0,
            ORANGE,
        );

        if let Some(ref w) = app.last_selected {
            draw_text(
                &format!("Selected: {}", w.text),
                10.0,
                100.0,
                20.0,
                MAGENTA,
            );
        }

        // -------------------------
        // Optional help overlay
        // -------------------------
        if app.show_help {
            let help_lines = [
                "HELP (Ctrl+H to hide):",
                "",
                "  Right mouse drag   : pan camera",
                "  Mouse wheel        : zoom in/out (centered on cursor)",
                "  Left click         : select word",
                "",
                "  Ctrl+L             : load corpus JSON and rebuild batches",
                "  Ctrl+Left/Right    : switch word batch",
                "  Ctrl+Tab           : toggle semantic n-gram edges on/off",
                "  Ctrl+H             : toggle this help overlay",
            ];

            let mut y = 140.0;
            for line in help_lines.iter() {
                draw_text(line, 10.0, y, 20.0, WHITE);
                y += 20.0;
            }
        }

        // Present frame
        next_frame().await;
    }
}
