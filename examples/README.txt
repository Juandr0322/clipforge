Few-shot example library for highlight detection.

Each .txt file in this directory = one example block injected into the
system prompt. Fill these with clips that ACTUALLY performed well in your
niche. Suggested format per file:

  source: <what the long-form video was about>
  clip transcript: <the text of the winning clip>
  why it worked: <hook type, structure, payoff>
  performance: <e.g. 210k views vs channel avg 15k>

Then run:  python -m clipforge.cli highlight <job> --examples-dir examples/
Start with 3-5 strong examples; more than ~8 bloats the prompt.
