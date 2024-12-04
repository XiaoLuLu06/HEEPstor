#!/usr/bin/env python3

from pathlib import Path
from string import Template
import argparse

def generate_heepstor_defs(template: str, enable_debug: bool, use_software_dnn: bool) -> str:
    """Generate heepstor_defs.h content from template and parameters."""
    template = Template(template)
    return template.substitute(
        ENABLE_DEBUG_HEEPSTOR_ASSERTIONS='1' if enable_debug else '0',
        USE_SOFTWARE_DNN_LAYER_OPERATORS='1' if use_software_dnn else '0'
    )

def main():
    parser = argparse.ArgumentParser(prog="heepstorgen")
    parser.add_argument('template_path', type=Path, help='Path to heepstor_defs.h.tpl template file')
    parser.add_argument('--enable-debug', type=int, required=True, choices=[0,1],
                      help='Enable debug assertions in HeepStor (0 or 1)')
    parser.add_argument('--use-software-dnn', type=int, required=True, choices=[0,1],
                      help='Use software DNN layer operators (0 or 1)')
    
    args = parser.parse_args()
    
    with open(args.template_path, 'r') as f:
        template = f.read()
        
    content = generate_heepstor_defs(
        template, 
        bool(args.enable_debug), 
        bool(args.use_software_dnn)
    )
    
    output_path = args.template_path.parent / 'heepstor_defs.h'
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"Generated {output_path}")

if __name__ == "__main__":
    main()