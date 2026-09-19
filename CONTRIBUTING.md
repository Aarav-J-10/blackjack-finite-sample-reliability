# Contributing

1. Create a branch for the proposed change.
2. Keep stochastic changes reproducible by recording all random seeds.
3. Add or update tests whenever simulator or probability logic changes.
4. Run `python -m pytest` before opening a pull request.
5. Do not replace archived full-run results with quick-mode outputs.
6. Document whether a reported probability is strict reversal, tie, or the
   combined nonpositive probability.

Changes to blackjack rules should include the rule configuration in both the
code and result metadata. Changes to manuscript claims should identify the CSV
or exact calculation that supports the claim.

