# Guidelines

- Be extremely concise. Sacrifice grammar for the sake of concision.
- Only use emojis or any other special characters if explicitly asked to do so.
- Only add try/except when you have a specific recovery strategy; catch precise exception types and handle meaningfully, never silently suppress.
- Only write .md documents if explicitly asked to do so.
- Context auto-compacts — never stop tasks early. Save progress and state to memory before compaction hits.
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.
- Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.
- Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.
- Update Journal.md when making major changes.

## Delegation and Parallelism

**Default behavior:** Spin up team agents for research, exploration, and implementation. You are the coordinator — don't waste main context on work agents should do. Delegate first, synthesize results, act.

### When to delegate
- **Team agents (opus):** Complex multi-step tasks — feature implementation, large refactors, deep investigations. Each agent gets full tool access, manages its own context, and reports back. Use when work decomposes into independent workstreams.
- **Subagents (opus):** Focused parallel units — research, code review, codebase exploration, running checks. Always launch independent subagents in parallel (multiple Agent calls in one message).
- **Codex MCP:** Second opinions on tricky logic, debugging subtle issues, comparing fixes. Use as collaborator alongside agents.

### How to delegate well
- Give each agent a **complete, specific prompt** — full context on what to do, what files matter, what the acceptance criteria are. No vague handoffs.
- **Never duplicate work** you delegated — if an agent is researching X, don't also research X in main context.
- **Validate agent output** before integrating — review diffs, run tests, check correctness. Trust but verify.
- **Report completion clearly** — when agents finish, synthesize their results into a concise summary for the user.

### Quick decision guide
- Simple lookups → do it yourself directly
- 2-10 independent searches/reviews → subagents in parallel
- Multi-step complex work → team agents
- Hard bugs / need validation → Codex MCP
- For hard problems, spawn 2+ agents solving independently, then compare solutions

## Code Architecture Principles

### Investigate before answering
- Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.

### File length
- Files should have a single cohesive responsibility. When distinct responsibilities emerge, split at natural boundaries.
- Actively watch for files growing beyond their original scope — length is a symptom, not the disease.
- Use folders and naming conventions to keep small files logically grouped.

### Structure and encapsulation
- Encapsulate functionality in the language's natural unit (class, struct, module, protocol, closure) — even if it's small.
- Favor composition over inheritance. In functional languages, favor small composable functions and modules.
- Code must be built for reuse, not just to "make it work."
- Every file, class, and function should have a single responsibility. If it has multiple, split it.
- Components should be interchangeable, testable, and isolated. Reduce tight coupling — favor dependency injection or protocols/interfaces.

### Structural over mechanical

When improving code, prioritize structural work over mechanical fixes. Structural changes — responsibility splits, layer separation, package organization, abstraction alignment — compound and prevent entire categories of future bugs. Mechanical changes — exception narrowing, annotation tweaks, import sorting — are maintenance. Do mechanical work as quick wins between structural improvements, not as the main focus.

### Anti-slop discipline

These correct patterns LLMs specifically get wrong — code that "works" but rots the codebase:

- Delete dead code on sight — unused imports, unreachable branches, orphaned files, commented-out blocks. Never keep code "just in case."
- No near-duplicate functions — if two functions share most of their logic, refactor into one with parameters. Copy-paste-tweak is the #1 source of structural rot.
- No docstrings on obvious methods — if the name and signature fully communicate the behavior, a docstring adds nothing. Only document non-obvious behavior, parameter constraints, or surprising return values.
- Keep functions flat and short — deep nesting and long functions are slop magnets. Extract when a function does more than one conceptual thing.
- Verify before creating or splitting — before adding a new function, class, or file, search whether existing code already handles it. Before splitting a module, understand its current shape and responsibilities — don't split prematurely or along arbitrary lines. New abstractions must earn their existence.
- Clean the blast radius — when modifying code, check callers and callees. Remove newly-dead code, fix broken patterns in the immediate vicinity. Don't leave a trail.
- Naming must be consistent within a module — don't mix conventions in the same file or directory.
- No god classes — if a class has multiple distinct responsibilities, split it. Growth is not an excuse.

### Correctness
- Verify syntax against the actual version of the library being used.
- Cover edge and corner cases without complexifying the logic.
