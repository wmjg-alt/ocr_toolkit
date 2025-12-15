import argparse
import sys
import os
import contextlib

# 1. GLOBAL SILENCE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TQDM_DISABLE"] = "1" 

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import ocr

# Disable TQDM programmatically
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

def main():
    parser = argparse.ArgumentParser(description="Local OCR CLI Tool")
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("--engine", default=None, help="Force a specific engine")
    parser.add_argument("--task", default="ocr", help="Task type: ocr, markdown, table, formula, det")
    parser.add_argument("--raw", action="store_true", help="Return raw JSON instead of just text")
    
    args = parser.parse_args()

    abs_image_path = os.path.abspath(args.image_path)
    if not os.path.exists(abs_image_path):
        print(f"Error: File not found at {abs_image_path}", file=sys.stderr)
        sys.exit(1)

    # config.yaml is assumed to be next to this script
    config_path = os.path.join(current_dir, "config.yaml")

    try:
        result = None
        
        # 4. THE TOTAL SILENCER
        # Redirect both stdout (prints) AND stderr (progress bars/warnings) to devnull.
        # We assume that if something goes wrong, it raises an Exception, 
        # which we catch outside this block and print to stderr.
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                
                # Initialize
                reader = ocr.OCR(engine=args.engine, config_path=config_path, verbose=False)
                
                # Read
                result = reader.read(abs_image_path, task=args.task)
        
        # 5. Output Result
        if result:
            if args.raw:
                import json
                print(json.dumps(result, default=str))
            else:
                try:
                    print(result['text'])
                except UnicodeEncodeError:
                    print(result['text'].encode('utf-8').decode(sys.stdout.encoding, errors='ignore'))
        else:
            # If result is None but no exception raised, it's an internal error
            print("Error: No result returned from engine.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        # Critical errors still get printed to console because we are outside the context manager
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()