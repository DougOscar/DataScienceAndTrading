# Phase 2 dry-run sandbox (DESIGN §10 Phase 2 exit test). Source before every quantlab command:
#   source research/dryrun/env.sh        (bash/zsh)
# Everything the dry run writes (ledger, trial store, system folder, Obsidian card) stays under
# research/dryrun/ -- the real research/ledger/, research/systems/ and DocumentationVault/ are untouched.
export QUANTLAB_LEDGER_DIR=research/dryrun/ledger
export QUANTLAB_STUDIES_DIR=research/dryrun/studies
export QUANTLAB_SYSTEMS_DIR=research/dryrun/systems
export QUANTLAB_VAULT_DIR=research/dryrun/vault
