import time
import ocr
import os
import json
import argparse
import yaml
from core.scorer import OCRScorer

# ==========================================
# 1. SETUP & UTILS
# ==========================================
CACHE_FILE = "benchmark_cache.json"

# ANSI Colors for dynamic assignment
COLORS = [
    '\033[96m', # Cyan
    '\033[92m', # Green
    '\033[93m', # Yellow
    '\033[94m', # Blue
    '\033[95m', # Magenta
    '\033[91m', # Red
    '\033[37m', # White
]

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)

def print_header(title):
    print(f"\n\033[1m{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}\033[0m")

# ==========================================
# 2. MAIN LOGIC
# ==========================================
def run_benchmark(image_path, truth_path=None):
    
    # 1. Load Config & Discover Engines
    config = load_config()
    engines_to_test = []
    
    # Filter for enabled engines in config
    for name, settings in config.get('engines', {}).items():
        if settings.get('enabled', False):
            engines_to_test.append(name)
    
    if not engines_to_test:
        print("Error: No enabled engines found in config.yaml")
        return

    # 2. Load Ground Truth (if provided)
    ground_truth = ""
    if truth_path:
        if os.path.exists(truth_path):
            with open(truth_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read()
            print(f"Loaded Ground Truth from: {truth_path}")
        else:
            print(f"Warning: Ground truth file '{truth_path}' not found. Scoring disabled.")
            truth_path = None

    # 3. Initialize Utils
    cache = load_cache()
    scorer = OCRScorer()
    
    # Prepare Result Data
    results = {}
    
    # 4. Run Loop
    print_header(f"BENCHMARK: {os.path.basename(image_path)}")
    
    print("   [System] Initializing Core...")
    reader = ocr.OCR(config_path="config.yaml")
    
    for i, engine_name in enumerate(engines_to_test):
        color = COLORS[i % len(COLORS)]
        print(f"\n{color}>>> ENGINE: {engine_name} \033[0m")
        
        # Init results dict
        results[engine_name] = {
            'color': color,
            'load_time': 0, 
            'inf_time': 0, 
            'text': ""
        }

        # Check Cache
        cache_key = f"{engine_name}::{image_path}"
        
        if cache_key in cache:
            print("   [Cache] Results loaded from disk.")
            cached = cache[cache_key]
            results[engine_name]['load_time'] = cached.get('load_time', 0)
            results[engine_name]['inf_time'] = cached.get('inf_time', 0)
            results[engine_name]['text'] = cached.get('raw_text', "")
        else:
            try:
                # Measure Load
                t0 = time.perf_counter()
                reader.load_engine(engine_name)
                t_load = time.perf_counter() - t0
                print(f"   [Time] Load:      {t_load:.4f}s")

                # Measure Inference
                t0 = time.perf_counter()
                # Default to 'ocr' task for comparison consistency
                res = reader.read(image_path, task="ocr")
                t_inf = time.perf_counter() - t0
                print(f"   [Time] Inference: {t_inf:.4f}s")

                if res:
                    results[engine_name]['load_time'] = t_load
                    results[engine_name]['inf_time'] = t_inf
                    results[engine_name]['text'] = res['text']
                    
                    # Update Cache
                    cache[cache_key] = {
                        'load_time': t_load,
                        'inf_time': t_inf,
                        'raw_text': res['text']
                    }
                    save_cache(cache)
                else:
                    results[engine_name]['error'] = "No Output"

            except Exception as e:
                print(f"   [Error] Failed: {e}")
                results[engine_name]['error'] = str(e)

        # Scoring (Run every time, even if cached)
        if truth_path:
            scores = scorer.evaluate(results[engine_name]['text'], ground_truth)
            results[engine_name]['scores'] = scores
            print(f"   [Score] Struct: {scores['levenshtein']:.2f}% | Content: {scores['cosine']:.2f}%")

    # 5. Print Report
    print_report(results, has_truth=(truth_path is not None))

def print_report(results, has_truth):
    print_header("FINAL RESULTS")
    
    # Print Text Preview
    for name, data in results.items():
        if 'error' in data: continue
        print(f"{data['color']} ------------ {name.upper()} ------------ \n{data['text']}\n")

    print("\n")
    
    # Table Header
    if has_truth:
        header = "{:<15} | {:<8} | {:<8} | {:<10} | {:<10}".format("Model", "Load(s)", "Infer(s)", "Struct %", "Content %")
    else:
        header = "{:<15} | {:<8} | {:<8}".format("Model", "Load(s)", "Infer(s)")
        
    print(header)
    print("-" * len(header))
    
    # Track Winner
    best_score = -1
    best_model = None
    max_t = -1
    slowest_model = None
    for name, data in results.items():
        if 'error' in data:
            print(f"{name:<15} | ERROR    | ERROR")
            continue
            
        load = f"{data['load_time']:.2f}"
        inf = f"{data['inf_time']:.2f}"
        tot_t = data['load_time'] + data['inf_time']
        if tot_t >= max_t:
            max_t = tot_t
            slowest_model = name
        
        if has_truth:
            lev = data['scores']['levenshtein']
            cos = data['scores']['cosine']
            print(f"{name:<15} | {load:<8} | {inf:<8} | {lev:<10.2f} | {cos:<10.2f}")
            
            # Simple "Best" logic: Average of both scores
            avg_score = (lev + cos) / 2
            if avg_score > best_score:
                best_score = avg_score
                best_model = name
        else:
            print(f"{name:<15} | {load:<8} | {inf:<8}")

    print("-" * len(header))
    if best_model:
        print(f"\n🏆 \033[1mOVERALL WINNER: {best_model}\033[0m")
        print(f"\n🏆 \033[1m SLOWEST MODEL: {slowest_model}\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local OCR Benchmark Tool")
    parser.add_argument("image", help="Path to the image file to test")
    parser.add_argument("truth", nargs="?", help="Path to a text file containing ground truth (Optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
    else:
        run_benchmark(args.image, args.truth)