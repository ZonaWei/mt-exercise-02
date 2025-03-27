
import sys
import html

for line in sys.stdin:
    if line.strip() == "":
        continue
    else:
        line = line.replace(u'\ufeff', '')
        line = html.unescape(line)
        line = " ".join(line.split())
        sys.stdout.write(line + "\n")
        