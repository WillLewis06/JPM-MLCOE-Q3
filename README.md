Repository overview

This repository contains three independent codebases grouped for review and reproducibility. They are not integrated and do not share code or runtime dependencies. Data files and trained checkpoints are excluded.

Full_Pipeline/

End-to-end implementation of the full modelling pipeline, including:
-model definitions
-training and evaluation entrypoints
-configuration files
-pipeline utilities and tests
This component is self-contained and intended to be run independently.
Detailed instructions are provided in Full_Pipeline/README.md.

Choice_Learn/
Minimal extensions to the existing choice_learn framework:
-newly implemented models and building blocks
-a single execution entrypoint
-unit tests for the added models
This directory contains only code introduced in this project and assumes these files will be added to a re-existing install.

ShuhanZhang_repo/
Frozen snapshot of the original Zhang et al. reference implementation.
Included for baseline comparison and long-term reproducibility
