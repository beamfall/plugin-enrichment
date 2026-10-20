# AGENTS.md — Beamfall Enrichment

Enrichment plugin for [Beamfall Core](https://github.com/beamfall/core): deterministic content
fingerprints for text and PNG images, with no model, no network, and no dependencies.

> **This file is canonical.** `CLAUDE.md` is a symlink to it — edit `AGENTS.md`, never maintain a
> second copy.

> **Documentation:** keep public docs in `README.md`, `CONTRIBUTING.md`, `SECURITY.md` and this file.
> When this repo is checked out inside the Beamfall multi-repo workspace, also follow that workspace's
> `DOCUMENTATION-GOVERNANCE.md` before adding, moving, or archiving any doc.

Use normal repo-local CLI and shell tools.

## Layout

- `enrichment.go` — text fingerprints; `image.go` — PNG image fingerprints.
- `enrichment_test.go`, `image_test.go` — provider tests (no parser or scorer vectors here).
- `script/gate.sh`, `.github/workflows/`, `testing/coverage-floor.json` — gate and CI.

## Build and test

- `go test ./...` — provider unit tests.
- `script/gate.sh` — full local gate: gofmt, vet, golangci-lint, test-depth contracts, exact
  per-package coverage, race, bounded image decoder fuzz. Requires `golangci-lint` and `python3`
  on `PATH`.
- `.github/workflows/conformance.yml` runs the same checks in CI, plus `assert-no-external-deps`
  and staticcheck. Keep the local gate and the workflow in step.
- Per-package coverage floors live in `testing/coverage-floor.json`; raise them with new tests, do
  not lower them to pass the gate.

## Conventions

- Standard library only: `go.mod` has no `require` block and CI asserts it. This plugin
  intentionally does not depend on `github.com/beamfall/plugin-sdk`; do not add it or any other
  module.
- Fingerprints must stay deterministic: the same input bytes always yield the same output.
- Format with `gofmt`; the lint rule set is `.golangci.yml`, shared verbatim across Beamfall plugin
  repos — do not fork it per plugin.

## Corvint context

Before editing, read `corvint context "<task>"` or `corvint impact <path>...`; before choosing tests,
run `corvint affected --base <origin/main sha>`. **Verify:** `script/gate.sh`.

## Contributing and license

- Licensed under the GNU AGPL v3 (`LICENSE`). Preserve license headers and third-party notices.
- External contributions need an accepted Beamfall Contributor License Agreement first: see
  `CONTRIBUTING.md` and `CLA.md`. Report vulnerabilities as described in `SECURITY.md`, not in
  public issues.
