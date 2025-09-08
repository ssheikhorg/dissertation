# utils/organize_results.py
import os
import shutil
import json
from pathlib import Path

model_name = "mistral-7b"

def organize_downloaded_results(source_dir=".", target_dir="evaluation_results"):
    """Organize downloaded evaluation results into proper directory structure"""

    target_dir = target_dir + f"/{model_name}"

    # File patterns to look for
    result_patterns = [
        "*_comprehensive_results.json",
        "*_ui_export_data.json",
        "*_results.json"
    ]

    viz_patterns = [
        "*_chart.png",
        "comparison_table.html",
        "radar_chart.png"
    ]

    # Move result files
    for pattern in result_patterns:
        for file_path in Path(source_dir).glob(pattern):
            if file_path.is_file():
                shutil.move(str(file_path), os.path.join(target_dir, file_path.name))
                print(f"Moved: {file_path.name} -> {target_dir}/")

    # Move visualization files
    for pattern in viz_patterns:
        for file_path in Path(source_dir).glob(pattern):
            if file_path.is_file():
                shutil.move(str(file_path), os.path.join("visualizations", file_path.name))
                print(f"Moved: {file_path.name} -> visualizations/")

    print("Organization complete!")


if __name__ == "__main__":
    organize_downloaded_results()