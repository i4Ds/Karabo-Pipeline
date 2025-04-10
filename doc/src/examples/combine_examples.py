import os

# Change working directory to the directory of this file
file_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

example_structure = open("example_structure.md", "r").read()

if os.path.exists("examples.md"):
    os.remove("examples.md")
example_md = open("examples.md", "x")

for line in example_structure.splitlines():
    # print(line)
    if line.startswith("<") and line.endswith(">"):
        example_path = os.path.join("_example_scripts", line[1:-1])
        example = open(example_path, "r").read()
        example_md.write(example)
    else:
        example_md.write(line)
        example_md.write("\n")
