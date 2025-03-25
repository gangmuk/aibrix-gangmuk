import json
import argparse

def process_jsonl_prefix(input_file, output_file, common_prompt):
    print(f"Adding common prompt: {common_prompt}")
    common_prompt_length = len(common_prompt)
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            try:
                data = json.loads(line.strip())
                
                # Update each request in the requests array
                for request in data['requests']:
                    # Add prefix to prompt
                    if request.get('prompt') is not None:
                        request['prompt'] = common_prompt + " " + request['prompt']
                        
                        # Update Prompt Length if it exists
                        if 'Prompt Length' in request:
                            request['Prompt Length'] = len(request['prompt'])
                
                fout.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON: {e}")
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Add common prompt to JSONL file')
    parser.add_argument('--input_file', required=True, help='Input JSONL file path')
    parser.add_argument('--output_file', required=True, help='Output JSONL file path')
    parser.add_argument('--common_prompt', required=True, help='Common prompt to prepend')
    
    args = parser.parse_args()
    
    try:
        print(f"args.input_file: {args.input_file}")
        print(f"args.output_file: {args.output_file}")
        print(f"args.common_prompt: {args.common_prompt}")
        process_jsonl_prefix(args.input_file, args.output_file, args.common_prompt)
        print(f"Processing complete. Output written to: {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
