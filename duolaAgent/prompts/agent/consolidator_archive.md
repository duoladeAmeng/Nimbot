Extract key facts from this conversation. For each fact, annotate its memory attributes.

Only SNIP facts deserve a non-[skip] mark:
- Signal: would the user need to repeat this if forgotten?
- Novel: not just a restatement of another fact in this same conversation chunk
- Important: prevents rework or captures preferences / rules
- Persistent: still relevant after 2 weeks

Output one fact per line in this format:
- [mark] fact content

Marks:
- [permanent] Core preferences, personal traits, habits
- [durable] Technical discoveries, project knowledge, config details
- [ephemeral] Active task state, temporary decisions
- [correction] Correction to a previous memory
- [skip] Conversational filler, repo-derivable facts, or audit-only breadcrumbs

Priority: user corrections and preferences > solutions > decisions > events > environment facts.
Output concise bullet points only. No preamble, no commentary.
If nothing noteworthy happened, output: (nothing)
