from jinja2 import Environment, FileSystemLoader
import pandas as pd
import os
from typing import Dict


class ReportGenerator:
    def __init__(self, template_dir="templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_html_report(self, results: Dict, output_path: str) -> None:
        template = self.env.get_template("report_template.html")

        # Convert results to pandas for easy display
        df = pd.DataFrame.from_dict(results, orient="index")

        html_content = template.render(
            models_results=df.to_html(classes="data"), metrics=list(df.columns)
        )

        with open(output_path, "w") as f:
            f.write(html_content)
