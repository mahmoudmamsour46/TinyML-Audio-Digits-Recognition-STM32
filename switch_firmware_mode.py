"""
Seamless firmware mode switcher for STM32 deployment:
- mode = 'linear' -> Enable Linear Quantization
- mode = 'pot'    -> Enable Paper Power-of-Two PTQ
"""

import os
import shutil
import sys

STM_INC = r"STM CODE\Audio_Digits_Recognitio\Core\Inc"
STM_SRC = r"STM CODE\Audio_Digits_Recognitio\Core\Src"
BACKUP_DIR = r"export\backup_pot"

def switch_to(mode):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print(f"Switching STM32 project mode to: {mode.upper()}...")
    
    # 1. Backup original Power-of-Two files if not already present
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(BACKUP_DIR, 'weights_data.h')):
        shutil.copy2(os.path.join(STM_INC, 'weights_data.h'), os.path.join(BACKUP_DIR, 'weights_data.h'))
        shutil.copy2(os.path.join(STM_INC, 'model_parameters.h'), os.path.join(BACKUP_DIR, 'model_parameters.h'))
        shutil.copy2(os.path.join(STM_SRC, 'nn_inference.c'), os.path.join(BACKUP_DIR, 'nn_inference.c'))
        print("  - Backed up original Power-of-Two files in export/backup_pot/.")
        
    if mode.lower() == 'linear':
        # Copy Linear Quantization files to STM32 project
        shutil.copy2(r"export\weights_data_linear.h", os.path.join(STM_INC, 'weights_data.h'))
        shutil.copy2(r"export\model_parameters_linear.h", os.path.join(STM_INC, 'model_parameters.h'))
        shutil.copy2(r"export\nn_inference_linear.c", os.path.join(STM_SRC, 'nn_inference.c'))
        
        # Set SCALE_IN in audio_processor.c to 36.3160f
        ap_path = os.path.join(STM_SRC, 'audio_processor.c')
        with open(ap_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        content_mod = content.replace("#define SCALE_IN   (32.0f)", "#define SCALE_IN   (36.3160f)")
        with open(ap_path, 'w', encoding='utf-8') as f:
            f.write(content_mod)
            
        print("\n[SUCCESS] Enabled Linear Quantization (Fixed-Point Integer) in STM32 firmware.")
        print("  - Quantized weights: export/weights_data_linear.h")
        print("  - Parameters & Multipliers: export/model_parameters_linear.h")
        print("  - Inference routine: export/nn_inference_linear.c")
        print("  - Audio scale (SCALE_IN): 36.3160f")
        
    elif mode.lower() == 'pot':
        # Restore Power-of-Two files
        shutil.copy2(os.path.join(BACKUP_DIR, 'weights_data.h'), os.path.join(STM_INC, 'weights_data.h'))
        shutil.copy2(os.path.join(BACKUP_DIR, 'model_parameters.h'), os.path.join(STM_INC, 'model_parameters.h'))
        shutil.copy2(os.path.join(BACKUP_DIR, 'nn_inference.c'), os.path.join(STM_SRC, 'nn_inference.c'))
        
        # Reset SCALE_IN in audio_processor.c to 32.0f
        ap_path = os.path.join(STM_SRC, 'audio_processor.c')
        with open(ap_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        content_mod = content.replace("#define SCALE_IN   (36.3160f)", "#define SCALE_IN   (32.0f)")
        with open(ap_path, 'w', encoding='utf-8') as f:
            f.write(content_mod)
            
        print("\n[SUCCESS] Restored Power-of-Two PTQ (Paper Original) in STM32 firmware.")
        
    else:
        print("Unknown mode! Use 'linear' or 'pot'.")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'linear'
    switch_to(mode)
