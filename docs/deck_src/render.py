import os, sys, win32com.client
src = os.path.abspath(sys.argv[1]); out = os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True)
app = win32com.client.Dispatch("PowerPoint.Application")
p = app.Presentations.Open(src, True, False, False)
for i, s in enumerate(p.Slides, 1):
    s.Export(os.path.join(out, f"s{i:02d}.png"), "PNG", 1600, 900)
p.Close(); print(p.Slides.Count if False else "rendered")
