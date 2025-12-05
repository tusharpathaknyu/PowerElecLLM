#!/bin/bash
# Monitor SPICE evaluation and commit results when complete

cd /Users/tushardhananjaypathak/Desktop/PowerElecLLM

echo "$(date): Starting monitor for SPICE evaluation (PID 3012)..."

# Wait for the process to finish
while ps -p 3012 > /dev/null 2>&1; do
    sleep 60
    echo "$(date): Evaluation still running..."
    tail -3 spice_eval_650.log 2>/dev/null | head -1
done

echo "$(date): Evaluation completed!"

# Wait a moment for files to be written
sleep 5

# Generate results summary
echo "=== GENERATING RESULTS SUMMARY ===" 

python3 << 'EOF'
import json
import os
from datetime import datetime

results_dir = "benchmarks/results"
summary_lines = []
summary_lines.append("# SPICE Evaluation Results Summary")
summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Find all SPICE result files
spice_files = [f for f in os.listdir(results_dir) if f.startswith('spice_full_') and f.endswith('.json')]

for fname in sorted(spice_files):
    filepath = os.path.join(results_dir, fname)
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        results = data if isinstance(data, list) else data.get('results', [])
        model_name = fname.replace('spice_full_', '').replace('.json', '').split('_2025')[0]
        
        summary_lines.append(f"## {model_name}")
        summary_lines.append(f"- Total problems: {len(results)}")
        
        # Count by criteria
        correct = sum(1 for r in results if r.get('spice_correct', False))
        summary_lines.append(f"- Correct (SPICE): {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
        
        # By level
        for lvl in [1, 2, 3, 4]:
            lvl_results = [r for r in results if r.get('level') == lvl or f'L{lvl}' in r.get('problem_id', '')]
            if lvl_results:
                lvl_correct = sum(1 for r in lvl_results if r.get('spice_correct', False))
                summary_lines.append(f"  - Level {lvl}: {lvl_correct}/{len(lvl_results)} ({lvl_correct/len(lvl_results)*100:.1f}%)")
        
        # Test set results
        test_results = [r for r in results if r.get('split') == 'test']
        if test_results:
            test_correct = sum(1 for r in test_results if r.get('spice_correct', False))
            summary_lines.append(f"  - **Test Set**: {test_correct}/{len(test_results)} ({test_correct/len(test_results)*100:.1f}%)")
        
        summary_lines.append("")
    except Exception as e:
        summary_lines.append(f"## {fname}: Error reading - {e}\n")

# Write summary
with open("benchmarks/results/SPICE_RESULTS_SUMMARY.md", "w") as f:
    f.write("\n".join(summary_lines))

print("Summary written to benchmarks/results/SPICE_RESULTS_SUMMARY.md")
EOF

# Show summary
echo ""
cat benchmarks/results/SPICE_RESULTS_SUMMARY.md

# Git operations
echo ""
echo "=== COMMITTING TO GIT ==="

git add -A
git status

git commit -m "SPICE Evaluation Results - $(date '+%Y-%m-%d %H:%M')

- Full 650-problem SPICE-based evaluation completed
- Models evaluated: GPT-4o base, Fine-tuned GPT-4o, Fine-tuned GPT-4o-mini
- Test set v2 with all 10 topologies (150 problems)
- Results saved in benchmarks/results/

See SPICE_RESULTS_SUMMARY.md for detailed breakdown."

echo ""
echo "$(date): Done! Results committed."
