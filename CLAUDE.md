# Guidelines to follow
- Be extremely concise. Sacrifice grammar for the sake of concision.
- All docs-names are written in lower case with dashes connecting words, e.g. this-is-a-doc.md
- You will never use emojis or other such characters.

# Python usage
- We run python using `uv`
- uv add *package* - never add packages manually in the toml.
- uv add --dev *package*

# Shell Tool Usage Guidelines

<shell_tools> 

When you need to call tools from the shell, use this rubric:
- Find Files: `fd`
- Find Text: `rg` (ripgrep)
- Find Code Structure (Python): `ast-grep`
    - For Python files:
        - `.py` → `ast-grep --lang python -p '<pattern>'`
- For other languages, set `--lang` appropriately (e.g., `--lang rust`, `--lang tsx`, `--lang ts`, etc.)
- Select among matches: pipe to `fzf`
- JSON: `jq`
- YAML/XML: `yq`

**Note:** Use the `-h` or `--help` flag with any tool to see usage examples and available options (e.g., `fd -h`, `rg --help`) 

</shell_tools>


<todays_date>

When you mention or need current date for e.g. web online search, run date command. 

</todays_date>

# Code Architecture Principles

<file_length_and_structure> 

- Never allow a file to exceed 600 lines.
- If a file approaches 500 lines, break it up immediately.
- Treat 1000 lines as unacceptable, even temporarily.
- Use folders and naming conventions to keep small files logically grouped. 

</file_length_and_structures>


<oop_first> 

- Every functionality should be in a dedicated class, struct, or protocol, even if it's small.
- Favor composition over inheritance, but always use object-oriented thinking.
- Code must be built for reuse, not just to "make it work." 

</oop_first>


<single_responsibility_principle> 

- Every file, class, and function should do one thing only.
- If it has multiple responsibilities, split it immediately.
- Each manager, or utility etc. should be laser-focused on one concern. 

</single_responsibility_principle>


<modular_design> 

- Code should connect like Lego - interchangeable, testable, and isolated.
- Ask: "Can I reuse this class in a different screen or project?" If not, refactor it.
- Reduce tight coupling between components. Favor dependency injection or protocols. 

</modular_design>