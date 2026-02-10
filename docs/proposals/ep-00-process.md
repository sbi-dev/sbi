# EP-00: Enhancement Proposal Process

Status: Discussion
Authors: @janfb

## Purpose

This document defines how `sbi` Enhancement Proposals (EPs) are created, discussed,
reviewed, and accepted. EPs are for substantial changes to APIs, architecture,
processes, or tooling that benefit from design discussion and broader visibility.

## When to write an EP

Write an EP when a change is:

- Affecting multiple modules or user-facing APIs.
- Architectural (e.g., training loop refactor, storage/logging backends).
- Process/governance-related (e.g., release, deprecation, contribution rules).
- Potentially controversial or with multiple viable designs.

Smaller changes (typo fixes, small bugfixes, doc tweaks) don’t need an EP.

## Structure of an EP

Recommended sections:

- Summary: One-paragraph overview of the proposal and motivation.
- Motivation: Context, problems with the status quo.
- Goals and Non-Goals: Clarify scope.
- Design: API changes, data flow, architecture.
- Alternatives Considered: Briefly describe alternatives and trade-offs.
- Backward Compatibility: Migration strategy and deprecations.
- Testing & Validation: How to verify correctness and performance.
- References: Related issues, PRs, or external work.

## Authoring and location

- Canonical source: sbi docs site under `mkdocs/docs/proposals/ep-XX-title.md`.
- Numbering: Incremental (EP-00, EP-01, …). Use a short kebab-case slug.
- Reviews happen via PRs to the repository, like other changes.

## Discussion and feedback

- Open a GitHub Discussion titled `EP-XX: <Title>` in the “Enhancement Proposals” category.
- Discussion post should include: brief summary, link to the EP doc, specific questions,
  and decision criteria if relevant.

## Decision process

- Aim for consensus. Maintainers make final calls after sufficient discussion.
- Acceptance criteria: clear benefits, feasibility, alignment with sbi design
  principles, and maintainability.
