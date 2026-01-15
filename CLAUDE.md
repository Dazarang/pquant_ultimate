# Guidelines

- Be extremely concise. Sacrifice grammar for the sake of concision.
- Only use emojis or any other special characters if explicitly asked to do so.
- Only add try/except when you have a specific recovery strategy; catch precise exception types and handle meaningfully, never silently suppress.
- Only write .md documents if explicitly asked to do so.
- Your context window will be automatically compacted as it approaches its limit, allowing you to continue working indefinitely from where you left off. Therefore, do not stop tasks early due to token budget concerns. As you approach your token budget limit, save your current progress and state to memory before the context window refreshes. Always be as persistent and autonomous as possible and complete tasks fully, even if the end of your budget is approaching. Never artificially stop any task early regardless of the context remaining.
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.
- Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.
- Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.

## Shell Tool Usage Guidelines

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


## Code Architecture Principles

<investigate_before_answering>
- Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</investigate_before_answering>

<file_length_and_structure>
- Never allow a file to exceed 600 lines (LoC-lines of code).
- If a file approaches 500 lines, break it up immediately.
- Treat 1000 lines as unacceptable, even temporarily.
- Use folders and naming conventions to keep small files logically grouped. 

</file_length_and_structure>

<oop_first>
- Every functionality should be in a dedicated class, struct, or protocol, even if it's small.
- Favor composition over inheritance, but always use object-oriented thinking.
- Code must be built for reuse, not just to "make it work." 

</oop_first>

<single_responsibility_principle>
- Every file, class, and function should do one thing only.
- If it has multiple responsibilities, split it immediately.
- Each manager, service, or utility should be laser-focused on one concern. 

</single_responsibility_principle>

<modular_design>
- Code should connect like Lego - interchangeable, testable, and isolated.
- Ask: "Can I reuse this class in a different module or project?" If not, refactor it.
- Reduce tight coupling between components. Favor dependency injection or protocols. 

</modular_design>

<edge_and_corner_cases>
- Double and triple check what you write uses the correct syntax based on the version of the library you are using.
- Double and triple check that you cover edge and corner cases without complexifying the logic or code.

</edge_and_corner_cases>