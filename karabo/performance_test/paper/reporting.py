import os
import tempfile
from pathlib import PurePath

import PIL
from jinja2 import Environment, FileSystemLoader
from markdown_pdf import MarkdownPdf, Section

"""
Collects the results of several perfomance runs and
summarizes them in a pdf.
Author: andreas.wassmer@fhnw.ch
"""


def resize_image(fname: str, width: int) -> str:
    tmp_file = tempfile.mkstemp(suffix=".png", dir=".")[1]
    # use file name only without path.
    p = PurePath(tmp_file)
    im = PIL.Image.open(fname)
    aspect = float(im.height) / float(im.width)
    im = im.resize((width, int(width * aspect)))
    im.save(p.name)
    return p.name


env = Environment(loader=FileSystemLoader("reporting"))
template = env.get_template("benchmark_report_template.md")

script = open("euler_scripts/run_paper_benchmark.sh")
script_lines = []
for line in script:
    if line == "":
        script_lines.append(" ")
    script_lines.append(line)


compute_image_graph = resize_image("computing_times.png", width=400)
filesize_image_graph = resize_image("fits_filesize.png", width=350)

output = template.render(
    cluster="Euler",
    script=script_lines,
    sky_model="GLEAM",
    num_sources=96860,
    num_channels="50, 100, 200, 400, 800, 1000",
    compute_image=compute_image_graph,
    size_image=filesize_image_graph,
)

CSS = """
code {font-size:11pt; color:#000099;}
ul {font-family: sans-serif}
ol {font-family: sans-serif}
p {font-family: sans-serif}
h1 {font-size:12pt; font-family: sans-serif}
h2 {font-family: sans-serif}
h3 {font-family: sans-serif}
h4 {font-family: sans-serif}
"""

pdf = MarkdownPdf(toc_level=1)
pdf.add_section(Section(output), user_css=CSS)
pdf.save("report.pdf")

os.remove(compute_image_graph)
os.remove(filesize_image_graph)
