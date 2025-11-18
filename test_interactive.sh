#!/bin/bash
# Test script for Phase 2 interactive mode

echo "Testing interactive mode..."
echo ""

# Send test queries
echo "hello" | cargo run --bin m2_base --quiet 2>/dev/null | tail -20

echo ""
echo "========================================"
echo "To run interactively, use:"
echo "  cargo run --bin m2_base"
echo ""
echo "Then try queries like:"
echo "  m2> hello"
echo "  m2> how are you"
echo "  m2> machine learning"
echo "  m2> :help"
echo "  m2> :quit"
echo "========================================"
