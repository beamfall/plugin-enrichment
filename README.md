<p align="center">
  <img src=".github/brand/beamfall-logo.png" alt="Beamfall" width="96" height="96">
</p>

# Beamfall Enrichment Plugin

**Deterministic content fingerprints for your Beamfall library, with no model, no network, and no
dependencies.** The enrichment plugin gives Beamfall Core stable fingerprints of text and PNG
images, so the server can tell when content has changed and compare like with like.

It is honest about what it is. These fingerprints are hashes, not AI embeddings, and the plugin
says so: it never reports itself as a semantic search provider, and it refuses every request that
would store a hash where a semantic vector belongs. It is free and open source under
[AGPL-3.0](LICENSE).

## What you get

- **Text fingerprints.** `LexicalFingerprint` lowercases text, splits it into letter and digit
  tokens, hashes each token into one of 256 buckets, and L2-normalizes the result. The same text
  always gives the same fingerprint, on every machine.
- **Image fingerprints.** `PixelFingerprint` divides an image into a 16x16 grid, averages the
  color and luminance of each cell, and hashes the cells into a 256-value, L2-normalized
  fingerprint. It works on any decoded Go `image.Image`.
- **A careful PNG reader.** `PixelFingerprintReader` reads PNG files with its own streaming
  decoder. It checks every chunk and CRC, never builds a full decoded image in memory, and
  refuses input over hard limits: 16 MiB encoded, 16 million pixels, or 8,192 pixels on either
  side.
- **Cancellation that works.** Long work checks its context at least every 4,096 pixels. If the
  source is an `io.ReadCloser`, cancelling closes it, so a stalled read does not hold the server.
- **Batches with isolated failures.** `PixelFingerprintBatch` processes up to 1,024 images in one
  call. A bad file returns its own error and the rest of the batch still completes; an oversized
  batch is refused before any input is read.
- **No confusion with semantic search.** Text and image fingerprints carry their own kinds,
  `lexical-fingerprint-v1` and `pixel-fingerprint-v1`, and are kept apart from each other and
  from any semantic space. The semantic entry points (`Available`, `EmbedQuery`,
  `EmbedImageReader`, `EmbedImageBatch`) always report that no semantic provider is available.
- **A compatibility contract for real models.** `EmbeddingIdentity` records a semantic vector
  space by model, model version, weights digest, preprocessing, dimensions, normalization, and
  metric. `RequireCompatible` refuses to compare vectors unless every field matches, because equal
  dimensions alone do not make two spaces comparable.
- **Off unless you turn it on.** Each provider is built with an enabled switch and reports it
  through `FingerprintsAvailable`, so fingerprint work happens only when the operator enables it.
- **Pure Go standard library.** No third-party modules, no `go.sum`, no network access, and no
  dependency on other Beamfall modules. CI enforces this on every change.

## How it fits into Beamfall

The enrichment plugin ships with [Beamfall Core](https://github.com/beamfall/core), the
self-hosted media server. Core builds this module in directly and connects it to its own
embedding interface; you do not install it separately.

Most Beamfall plugins implement the bundle, scan, migrate, and scrobble seams from the
[Beamfall plugin SDK](https://github.com/beamfall/plugin-sdk). This one does not need them, so it
has no SDK dependency. The SDK's README and
[plugin-authoring guide](https://github.com/beamfall/plugin-sdk/blob/main/docs/plugin-authoring.md)
remain the reference for writing other plugins.

## Develop

You need the Go toolchain named in `go.mod`, Python 3, and `golangci-lint`.

```sh
go test ./...
script/gate.sh
```

`script/gate.sh` runs the full local gate: `gofmt`, `go vet`, the test-depth contracts, exact
per-package coverage against [testing/coverage-floor.json](testing/coverage-floor.json), and
`golangci-lint`. It ends with `ALL GATES GREEN`. Run `script/gate.sh race` for the race detector
and `script/gate.sh fuzz` for a bounded fuzz run of the PNG decoder.

| File | Contents |
| --- | --- |
| `enrichment.go` | Text fingerprints, the semantic identity contract, and `Provider`. |
| `image.go` | Image fingerprints, the streaming PNG decoder, and `ImageProvider`. |

## Contributing

Contributions require a signed Beamfall Contributor License Agreement. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md) before you open a pull request.

## Security

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Do not open a public
issue.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
