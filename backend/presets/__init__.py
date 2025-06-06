import os
from pathlib import Path
from importlib import import_module

# Dictionary to store loaded personas
PERSONAS = {}

# Get current directory of this __init__.py file
preset_dir = Path(__file__).parent  # NOT Path(__file__).parent / "presets"

for f in os.listdir(preset_dir):
    if f.endswith(".py") and not f.startswith("__"):
        try:
            module_name = f"{f[:-3]}"  # Remove .py extension
            module = import_module(f"presets.{module_name}")
            if hasattr(module, "name") and hasattr(module, "system_prompt"):
                PERSONAS[module.name] = module.system_prompt
        except Exception as e:
            print(f"Error loading persona {f}: {e}")

print(f"🧠 Loaded {len(PERSONAS)} personas: {list(PERSONAS.keys())}")