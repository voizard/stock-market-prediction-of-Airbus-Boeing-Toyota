"""Render a Markdown memo to PDF in the user's preferred WeasyPrint look."""
import sys
import markdown
from weasyprint import HTML

CSS = """
@page {
  size: A4; margin: 20mm 18mm 18mm 18mm;
  @top-right { content: "AIF-Update 2026"; font-family:'DejaVu Sans'; font-size:8pt; color:#666; }
  @bottom-center { content: counter(page) " / " counter(pages);
                   font-family:'DejaVu Sans'; font-size:8pt; color:#666; }
}
body { font-family:'DejaVu Serif'; font-size:10.5pt; line-height:1.5; color:#111; }
h1 { font-family:'DejaVu Sans'; color:#0b4a8a; font-size:19pt;
     border-bottom:2px solid #0b4a8a; padding-bottom:4px; }
h2 { font-family:'DejaVu Sans'; color:#0b4a8a; font-size:14pt; margin-top:18px; }
h3 { font-family:'DejaVu Sans'; color:#144d8a; font-size:11.5pt; margin-top:14px; }
blockquote { border-left:4px solid #0b4a8a; background:#f3f6fb; margin:10px 0;
             padding:6px 10px; font-style:italic; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:9.5pt; }
th { background:#0b4a8a; color:#fff; text-align:left; padding:5px 7px;
     font-family:'DejaVu Sans'; }
td { border:1px solid #ccd6e4; padding:4px 7px; }
tr:nth-child(even) td { background:#f2f5fa; }
pre { font-family:'DejaVu Sans Mono'; font-size:8.5pt; background:#f4f4f4;
      border:1px solid #ddd; border-radius:4px; padding:7px; white-space:pre-wrap; }
code { font-family:'DejaVu Sans Mono'; font-size:9pt; }
img { max-width:100%; }
hr { border:none; border-top:1px solid #ccd6e4; margin:16px 0; }
"""


def render(md_path, pdf_path):
    md = open(md_path, encoding="utf-8").read()
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    doc = ("<html><head><meta charset='utf-8'><style>%s</style></head>"
           "<body>%s</body></html>" % (CSS, html))
    HTML(string=doc, base_url=md_path).write_pdf(pdf_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
