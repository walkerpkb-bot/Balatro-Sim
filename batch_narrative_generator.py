#!/usr/bin/env python3
"""
Batch Narrative Generator
Process all run logs and generate stories in multiple formats.
"""

import json
from pathlib import Path
from narrative_engine import process_log

def process_all_logs(log_dir="run_logs", output_dir="narratives", styles=None):
    """
    Process all log files in a directory.
    
    Args:
        log_dir: Directory containing run_*.json files
        output_dir: Where to save narrative outputs
        styles: List of styles to generate (default: all)
    """
    if styles is None:
        styles = ['literary', 'haiku', 'dev_commentary', 'comic']
    
    log_path = Path(log_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all log files
    log_files = sorted(log_path.glob("run_*.json"))
    
    if not log_files:
        print(f"No log files found in {log_dir}/")
        return
    
    print(f"Found {len(log_files)} run logs")
    print(f"Generating {len(styles)} narrative style(s) for each")
    print("=" * 60)
    
    for log_file in log_files:
        run_name = log_file.stem  # e.g., "run_20260111_150640_standard"
        print(f"\n📖 Processing: {run_name}")
        
        # Create output directory for this run
        run_output = output_path / run_name
        run_output.mkdir(exist_ok=True)
        
        try:
            # Generate each style
            for style in styles:
                story, analysis = process_log(str(log_file), style)
                
                # Save story
                story_file = run_output / f"{style}.txt"
                with open(story_file, 'w') as f:
                    f.write(story)
                print(f"  ✓ {style}: {story_file}")
            
            # Save analysis JSON
            _, analysis = process_log(str(log_file), 'literary')  # Get analysis
            analysis_file = run_output / "analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"  ✓ analysis: {analysis_file}")
            
            # Create quick summary
            summary_file = run_output / "SUMMARY.txt"
            with open(summary_file, 'w') as f:
                quality = analysis['story_quality']
                f.write(f"{run_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Archetype: {quality['archetype'].replace('_', ' ').title()}\n")
                f.write(f"Theme: {analysis['theme'].replace('_', ' ').title()}\n")
                f.write(f"Story Quality: {quality['overall']:.2f}/1.0\n")
                f.write(f"Tension: {quality['tension']:.2f}/1.0\n")
                f.write(f"Development: {quality['development']:.2f}/1.0\n\n")
                
                # Load original log for summary
                with open(log_file) as lf:
                    log_data = json.load(lf)
                    summary = log_data.get('summary', {})
                    f.write(f"Result: {'Victory' if summary.get('victory') else 'Defeat'}\n")
                    f.write(f"Blinds: {summary.get('blinds_won', 0)}/{summary.get('total_blinds_attempted', 0)}\n")
                    f.write(f"Close Calls: {summary.get('close_calls', 0)}\n")
                    f.write(f"Jokers: {summary.get('jokers_acquired', 0)}\n")
            
            print(f"  ✓ summary: {summary_file}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✨ Done! Narratives saved to {output_dir}/")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    run_TIMESTAMP_PRESET/")
    print(f"      literary.txt")
    print(f"      haiku.txt")
    print(f"      dev_commentary.txt")
    print(f"      comic.txt")
    print(f"      analysis.json")
    print(f"      SUMMARY.txt")


def generate_index(output_dir="narratives"):
    """Generate an index.html to browse all narratives"""
    output_path = Path(output_dir)
    
    # Find all run directories
    runs = sorted([d for d in output_path.iterdir() if d.is_dir()], reverse=True)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Balatro Run Narratives</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            max-width: 1200px; 
            margin: 40px auto; 
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        h1 { color: #ff6b6b; border-bottom: 2px solid #ff6b6b; padding-bottom: 10px; }
        .run { 
            background: #2a2a2a; 
            padding: 20px; 
            margin: 20px 0; 
            border-left: 4px solid #4ecdc4;
            border-radius: 4px;
        }
        .run h2 { color: #4ecdc4; margin-top: 0; }
        .metrics { display: flex; gap: 20px; margin: 10px 0; }
        .metric { 
            background: #333; 
            padding: 10px 15px; 
            border-radius: 4px;
            flex: 1;
        }
        .metric-label { font-size: 0.8em; color: #999; }
        .metric-value { font-size: 1.2em; font-weight: bold; }
        .links { margin-top: 15px; }
        .links a { 
            display: inline-block;
            margin-right: 10px; 
            padding: 5px 15px;
            background: #444;
            color: #4ecdc4;
            text-decoration: none;
            border-radius: 3px;
        }
        .links a:hover { background: #555; }
        .archetype { 
            display: inline-block;
            padding: 5px 10px;
            background: #ff6b6b;
            color: white;
            border-radius: 3px;
            font-size: 0.9em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <h1>🃏 Balatro Run Narratives</h1>
    <p><em>Stories extracted from simulation data</em></p>
"""
    
    for run_dir in runs:
        # Load summary
        summary_file = run_dir / "SUMMARY.txt"
        if not summary_file.exists():
            continue
        
        with open(summary_file) as f:
            lines = f.readlines()
        
        # Parse summary
        run_name = lines[0].strip()
        archetype = ""
        quality = ""
        result = ""
        blinds = ""
        
        for line in lines:
            if line.startswith("Archetype:"):
                archetype = line.split(":", 1)[1].strip()
            elif line.startswith("Story Quality:"):
                quality = line.split(":", 1)[1].strip()
            elif line.startswith("Result:"):
                result = line.split(":", 1)[1].strip()
            elif line.startswith("Blinds:"):
                blinds = line.split(":", 1)[1].strip()
        
        html += f"""
    <div class="run">
        <h2>{run_name} <span class="archetype">{archetype}</span></h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Result</div>
                <div class="metric-value">{result}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Blinds</div>
                <div class="metric-value">{blinds}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Story Quality</div>
                <div class="metric-value">{quality}</div>
            </div>
        </div>
        <div class="links">
            <a href="{run_dir.name}/literary.txt">📖 Literary</a>
            <a href="{run_dir.name}/haiku.txt">🎋 Haiku</a>
            <a href="{run_dir.name}/dev_commentary.txt">💻 Dev Commentary</a>
            <a href="{run_dir.name}/comic.txt">📚 Comic</a>
            <a href="{run_dir.name}/analysis.json">📊 Raw Analysis</a>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    index_file = output_path / "index.html"
    with open(index_file, 'w') as f:
        f.write(html)
    
    print(f"\n📄 Index created: {index_file}")
    print(f"   Open in browser to browse all narratives")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "run_logs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "narratives"
    
    # Process all logs
    process_all_logs(log_dir, output_dir)
    
    # Generate index
    generate_index(output_dir)
    
    print(f"\n🎭 To view: open {output_dir}/index.html in a browser")
