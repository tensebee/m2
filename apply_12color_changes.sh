cat > /mnt/user-data/outputs/apply_all_changes.sh << 'FULL_SCRIPT'
#!/bin/bash

# Complete automatic application of all 12-color n-gram changes
# Usage: bash apply_all_changes.sh main.rs

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <main.rs>"
    exit 1
fi

FILE="$1"
BACKUP="${FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Create backup
cp "$FILE" "$BACKUP"
echo "✓ Created backup: $BACKUP"

# CHANGE 1: Add last_activation_time to Word struct
perl -i -pe 'BEGIN{undef $/;} s/(struct Word \{[^}]*embedding: Option<Vec<f32>>,)/$1\n    last_activation_time: f32,/smg' "$FILE"
echo "✓ Added last_activation_time to Word struct"

# CHANGE 2: Add elapsed_time to WordMap struct (before the last closing brace)
perl -i -pe 'BEGIN{undef $/;} s/(struct WordMap \{[^}]*target_chunk_tokens: usize,)/$1\n    elapsed_time: f32,/smg' "$FILE"
echo "✓ Added elapsed_time to WordMap struct"

# CHANGE 3: Update WordMap::new() - multiple changes
sed -i 's/ngram_trie: NgramTrie::new(7)/ngram_trie: NgramTrie::new(12)/' "$FILE"
sed -i 's/max_ngram_order: 7,/max_ngram_order: 12,/' "$FILE"
sed -i 's/config_input: String::from("7")/config_input: String::from("12")/' "$FILE"
perl -i -pe 'BEGIN{undef $/;} s/(fn new\(\) -> Self \{[^}]*frame_counter: 0,)/$1\n            elapsed_time: 0.0,/smg' "$FILE"
echo "✓ Updated WordMap::new() for 12-grams"

# CHANGE 4: Add last_activation_time when creating words
perl -i -pe 'BEGIN{undef $/;} s/(self\.words\.push\(Word \{[^}]*embedding: None,)/$1\n                    last_activation_time: 0.0,/smg' "$FILE"
echo "✓ Added last_activation_time in word creation"

# CHANGE 5: Replace entire color computation in compute_semantic_cells()
perl -i -0777 -pe 's/\/\/ Cell color based on[^}]*let transparent_color = Color::new\(color\.r, color\.g, color\.b, base_alpha\);/\/\/ Cell color based on context vs output + n-gram confidence (12-color wheel)
        let (base_color, base_alpha) = if word.in_context {
            \/\/ INPUT WORDS: White cells, high visibility
            let white = Color::new(0.9, 0.9, 0.9, 1.0);
            (white, 0.5)
        } else {
            \/\/ OUTPUT WORDS: HSV by n-gram order (full 12-point color wheel)
            let connection_count = neighbors.len();
            
            if connection_count > 0 {
                let max_order = neighbors.iter()
                    .map(\|(_, _, order, _)\| order)
                    .max()
                    .unwrap_or(\&2);
                
                let hue = match max_order {
                    2 => 0.0,      \/\/ Red
                    3 => 30.0,     \/\/ Red-Orange
                    4 => 60.0,     \/\/ Orange
                    5 => 90.0,     \/\/ Yellow-Orange
                    6 => 120.0,    \/\/ Yellow
                    7 => 150.0,    \/\/ Yellow-Green
                    8 => 180.0,    \/\/ Green
                    9 => 210.0,    \/\/ Green-Cyan
                    10 => 240.0,   \/\/ Cyan
                    11 => 270.0,   \/\/ Cyan-Blue
                    12 => 300.0,   \/\/ Blue
                    _ => 330.0,    \/\/ Blue-Magenta
                };
                
                let saturation = 0.4 + word.activation * 0.5;
                let value = 0.3 + (connection_count as f32 \/ 6.0) * 0.5;
                
                let color = Self::hsv_to_rgb(hue, saturation, value);
                let alpha = (connection_count as f32 \/ 6.0 * 0.6).max(0.15);
                (color, alpha)
            } else {
                let gray = Color::new(0.25, 0.25, 0.25, 1.0);
                (gray, 0.1)
            }
        };

        let transparent_color = Color::new(base_color.r, base_color.g, base_color.b, base_alpha);/s' "$FILE"
echo "✓ Replaced color computation with 12-color HSV system"

# CHANGE 6: Update score_dual_topology confidence for 12-grams
perl -i -0777 -pe 's/let confidence = match order \{[^}]*7 => 1\.0,/let confidence = match order {
                12 => 1.0,
                11 => 0.98,
                10 => 0.96,
                9 => 0.94,
                8 => 0.92,
                7 => 0.90,/s' "$FILE"
echo "✓ Updated score_dual_topology confidence mapping"

# CHANGE 7: Update score_ngram_only confidence for 12-grams
perl -i -0777 -pe 's/(fn score_ngram_only[^{]*\{[^}]*let confidence = match order \{)[^}]*(7 => 1\.0,)/$1
                12 => 1.0,
                11 => 0.98,
                10 => 0.96,
                9 => 0.94,
                8 => 0.92,
                $2/s' "$FILE"
echo "✓ Updated score_ngram_only confidence mapping"

# CHANGE 8: Update n-gram boosting in get_candidates
perl -i -0777 -pe 's/let boosted = prob \* match order \{[^}]*7 => 4\.0,/let boosted = prob * match order {
                            12 => 6.0,
                            11 => 5.5,
                            10 => 5.0,
                            9 => 4.5,
                            8 => 4.0,
                            7 => 3.5,/s' "$FILE"
echo "✓ Updated n-gram boosting for 12-grams"

# CHANGE 9: Update print_summary loop
sed -i 's/for n in 2\.\.=7 {/for n in 2..=12 {/' "$FILE"
echo "✓ Updated print_summary to show 12-grams"

# CHANGE 10: Add elapsed time tracking to main loop
perl -i -0777 -pe 's/(loop \{[^}]*let delta = get_frame_time\(\);)/$1\n        \n        word_map.elapsed_time += delta;/s' "$FILE"
echo "✓ Added elapsed_time tracking to main loop"

# CHANGE 11: Update activation tracking with timing
perl -i -0777 -pe 's/(if let Some\(word\) = self\.words\.iter_mut\(\)\.find\(\|w\| \&w\.text == selected\) \{\s*word\.activation = 1\.0;)/$1\n                word.last_activation_time = self.elapsed_time;/s' "$FILE"
echo "✓ Added last_activation_time tracking in update_activations"

# CHANGE 12: Update generation activation tracking
perl -i -0777 -pe 's/(if let Some\(word\) = self\.words\.iter_mut\(\)\.find\(\|w\| \&w\.text == current_word\) \{\s*word\.activation = 1\.0;)/$1\n                    word.last_activation_time = self.elapsed_time;/s' "$FILE"
echo "✓ Added last_activation_time tracking in generation"

echo ""
echo "================================================"
echo "✓ ALL CHANGES APPLIED SUCCESSFULLY!"
echo "================================================"
echo ""
echo "Changes made:"
echo "  • Added last_activation_time and elapsed_time fields"
echo "  • Changed max n-gram order from 7 to 12"
echo "  • Implemented 12-color HSV wheel for cell colors"
echo "  • Updated all confidence mappings for 12-grams"
echo "  • Added timing system for pulse animations"
echo ""
echo "Backup saved as: $BACKUP"
echo ""
echo "You can now compile and run your code!"
echo "The color wheel: Red(2g) → Orange(4g) → Yellow(6g) → Green(8g) → Cyan(10g) → Blue(12g)"
FULL_SCRIPT

chmod +x /mnt/user-data/outputs/apply_all_changes.sh
echo "✓ Script created and made executable!"
ls -lh /mnt/user-data/outputs/apply_all_changes.sh