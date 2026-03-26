# Copilot Instructions

## Purpose

This repository uses GitHub Copilot as an adaptive development assistant.
Copilot should continuously improve its understanding of the repository by capturing reusable knowledge, coding patterns, architectural decisions, and workflows.

The repository contains structured folders that act as a **knowledge base for Copilot**. During development sessions or when code changes occur, Copilot should propose updates to these files when new insights are discovered.

---

# Knowledge Architecture

Copilot should treat the following folders as the project's AI knowledge system.

.github/

* copilot-instructions.md → Global development rules
* instructions/ → Task-specific coding guidance
* agents/ → Multi-step development workflows
* patterns/ → Reusable architecture and coding patterns
* decision-log/ → Architectural decisions and reasoning
* knowledge/ → High-level understanding of the project
* feedback/ → Corrections and improvement notes
* prompts/ → Reusable prompts for development tasks

These files together form the **persistent memory of Copilot for this repository**.

---

# Continuous Learning Policy

During each Copilot session:

1. Observe patterns in the codebase.
2. Learn from developer feedback and corrections.
3. Identify repeated workflows or coding structures.
4. Capture architectural decisions.
5. Propose improvements to the Copilot knowledge system.

Copilot should suggest updates to the appropriate files whenever new patterns or conventions appear.

---

# Code Change Synchronization Policy

Whenever code is added, modified, refactored, or removed, Copilot should evaluate whether the change affects the knowledge base.

If relevant, Copilot should propose updates to the appropriate files so that the knowledge system remains accurate.

Examples:

### Architecture changes

Update:

.github/knowledge/

### New coding pattern appears

Update or create:

.github/patterns/

### A new recurring development workflow appears

Create or update:

.github/agents/

### New coding conventions appear

Update:

.github/instructions/

### Architectural decisions are made

Add a new file in:

.github/decision-log/

### Developer corrections or Copilot mistakes occur

Update:

.github/feedback/

This ensures the Copilot knowledge base stays **synchronized with the evolving codebase**.

---

# Global Rules

Copilot should prioritize information sources in the following order when generating code:

1. Existing code patterns in the repository
2. Rules defined in `.github/instructions/`
3. Architecture defined in `.github/knowledge/`
4. Patterns defined in `.github/patterns/`
5. Workflows defined in `.github/agents/`
6. Architectural reasoning in `.github/decision-log/`

Generated code must remain consistent with these sources.

---

# Pattern Detection

When Copilot observes repeated code structures in the repository:

1. Extract the reusable pattern.
2. Document it in `.github/patterns/`.
3. Provide a reusable template.
4. Describe when the pattern should be used.

Examples of patterns:

* Controller → Service → Repository architecture
* React functional component structure
* API request validation patterns
* Database query conventions
* Error handling patterns

Patterns should remain reusable across the repository.

---

# Instruction Management

The `.github/instructions/` folder stores **task-specific development guidance**.

Examples:

* backend-api.md
* generate-tests.md
* react-components.md
* database-patterns.md
* security-guidelines.md

Copilot should:

* Update existing instructions when conventions evolve
* Add new instructions when recurring tasks appear
* Remove outdated instructions if they no longer apply
* Avoid duplicating rules already present

---

# Agent Workflows

The `.github/agents/` folder defines multi-step workflows for complex development tasks.

Examples:

* feature-agent.md
* bug-fix-agent.md
* refactor-agent.md
* code-review-agent.md

Agents should contain:

Goal
Inputs
Steps
Expected output

If Copilot detects repeated multi-step development workflows, it should propose creating or updating an agent.

---

# Decision Logging

Architectural decisions should be documented in `.github/decision-log/`.

Each decision log should include:

Context
Decision
Reasoning
Tradeoffs

Copilot should suggest creating a decision log whenever major technical decisions are made.

---

# Project Knowledge

The `.github/knowledge/` folder contains high-level project understanding such as:

* system architecture
* technology stack
* domain model
* terminology
* integration details

Copilot should reference these files before generating complex features.

If architectural changes occur, Copilot should propose updating these files.

---

# Feedback Learning

The `.github/feedback/` folder captures mistakes and corrections.

Examples:

* copilot-mistakes.md
* improvement-notes.md

When developers correct Copilot-generated code, Copilot should treat it as feedback and propose documenting the correction.

This helps prevent repeating the same mistakes.

---

# Prompt Library

Reusable prompts can be stored in `.github/prompts/`.

Examples:

* implement-api.prompt.md
* write-tests.prompt.md
* debug-service.prompt.md

Copilot should reuse these prompts when similar tasks appear.

---

# When to Create or Update Files

Copilot should propose updates when:

* A coding pattern appears multiple times
* A workflow requires multiple steps
* A new architectural decision is made
* A developer corrects Copilot behavior
* The codebase structure changes
* New technologies are introduced
* Code changes invalidate existing documentation

Updates should be incremental and avoid unnecessary duplication.

---

# Code Generation Guidelines

When generating code, Copilot should:

1. Follow project architecture
2. Use existing patterns
3. Respect task-specific instructions
4. Align with documented decisions
5. Prefer consistency with the existing codebase
6. Ensure generated code remains maintainable and scalable

---

# Long-Term Objective

The objective is to build a **self-improving Copilot environment** where:

* Project conventions are automatically captured
* Architectural knowledge is preserved
* Development workflows are reusable
* Code changes automatically update the knowledge system
* Copilot becomes increasingly specialized for this repository

Over time this repository should function as a **continuously evolving knowledge base that improves Copilot accuracy and usefulness**.
