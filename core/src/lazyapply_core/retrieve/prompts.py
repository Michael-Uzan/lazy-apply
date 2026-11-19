"""Prompt text for requirement extraction.

Kept apart from the code that sends it. Prompts are tuned far more often than
the request logic, and mixing the two makes both harder to read.
"""

EXTRACTION_SYSTEM = """\
You extract checkable requirements from job postings for a resume-matching \
system.

Two things matter more than anything else.

1. MUST_HAVE vs NICE_TO_HAVE.
   Mark a requirement MUST_HAVE only if the posting treats it as required --
   it appears under requirements, or is worded as "required", "must have",
   "X+ years of". Mark it NICE_TO_HAVE when the posting hedges: "a plus",
   "nice to have", "bonus", "preferred", "familiarity with", or anything
   listed under a nice-to-have heading.
   This split decides whether a candidate is told to apply. Treating a
   "plus" as required rejects people who should apply, which is the worst
   failure this system can make. When a requirement is genuinely ambiguous,
   choose NICE_TO_HAVE.

2. Restate, do not copy.
   Each requirement's `text` is used as a semantic search query against the
   candidate's own descriptions of their work. Write it as a statement about
   what someone has done, in plain language, so it can match an account of
   real work.
     posting:  "Deep expertise in distributed systems"
     text:     "Built and operated distributed systems in production"
     posting:  "5+ years with React and TypeScript"
     text:     "Built production web applications with React and TypeScript"
   Do not include the years. Duration is checked against employment dates,
   not against retrieved evidence.

Other rules:
- One idea per requirement. Split "React and Node.js" into two if the
  posting treats them as separate skills.
- Ignore boilerplate: benefits, salary, equal-opportunity statements,
  location, "about us" text.
- Give each requirement a short snake_case id, unique within the role.
- Aim for 5-12 requirements. A posting needing more than that is usually
  being read too literally.
- At least one requirement must be MUST_HAVE.\
"""

EXTRACTION_USER = """\
Extract the requirements from this job posting.

<posting>
{role_text}
</posting>\
"""
