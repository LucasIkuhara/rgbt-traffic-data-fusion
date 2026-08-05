from os import listdir
from datetime import datetime


PUB_DIR = "publication"
PUB_SRC_DIR = f"{PUB_DIR}/src"

files = listdir(PUB_SRC_DIR)
files = [f for f in files if ".md" in f]
files.sort()

print("Source files used:", ", ".join(files))

# Load header and set the current date
final_md = open(f"{PUB_SRC_DIR}/header.yml").read()
final_md = final_md.replace("{{CURRENT_TS}}", datetime.now().date().strftime("%Y-%m-%d"))
print("Header used:", final_md)

for f in files:
    final_md += open(f"{PUB_SRC_DIR}/{f}").read() + "\n"

with open(f"{PUB_DIR}/tmp.md", "w") as f:
    f.write(final_md)
