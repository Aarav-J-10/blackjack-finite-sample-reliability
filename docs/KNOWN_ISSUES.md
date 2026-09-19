# Known issues and publication checks

## Strict reversal versus nonpositive outcome

The manuscript formally defines reversal as a strict event,
`mean difference < 0`, with equality treated as a tie. The Bernoulli table in
the current manuscript PDF instead reports `mean difference <= 0`. Because the
Bernoulli model is discrete, this distinction is material.

For example, at `n=100`:

- exact strict paired reversal risk is approximately **19.997%**;
- exact tie probability is approximately **4.548%**; and
- exact nonpositive probability is approximately **24.545%**.

The repository's exact-benchmark CSV reports all three quantities. Before final
publication, Table 4 and its figure should either use strict reversal throughout
or explicitly rename the inclusive quantity as nonpositive/misclassification
risk.

## Manuscript completeness

The supplied PDF still contains a placeholder abstract, reserved figure boxes,
an unfinished conclusion/future-work/code-availability section, and incomplete
reference entries `[9]` through `[13]`. These are editorial issues, not software
failures, but should be resolved before public release.

## License

No open-source license has been selected. Choose one before advertising public
reuse rights.

