// Package enrichment contains deterministic, non-semantic content
// fingerprints and the identity contract a future semantic provider must
// satisfy.
//
// The bundled algorithms are deliberately dependency-free hashes. They are
// useful for stable change detection and compatibility fixtures, but they are
// not trained models and must never be advertised, stored, or queried as
// semantic embeddings.
package enrichment

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

// KindString is the storage kind of the bundled lexical fingerprint. Its name
// intentionally cannot be confused with a semantic embedding-space kind.
const KindString = "lexical-fingerprint-v1"

// Dims is the fixed width of the bundled lexical and pixel fingerprints.
// Fingerprint width is not an embedding-space identity: semantic providers
// must publish a complete EmbeddingIdentity.
const Dims = 256

// MaxQueryInputBytes bounds input sanitation even though the bundled provider
// refuses semantic queries.
const MaxQueryInputBytes = 512

var (
	// ErrEmptyInput reports text that contains no letter or digit.
	ErrEmptyInput = errors.New("fingerprint input is empty")

	// ErrSemanticProviderUnavailable is returned by the bundled hash provider's
	// semantic seam. Enabling enrichment cannot turn a hash into a model.
	ErrSemanticProviderUnavailable = errors.New("semantic embedding provider is unavailable")

	// ErrIncompleteEmbeddingIdentity reports a future semantic space whose
	// persisted identity omitted a required compatibility field.
	ErrIncompleteEmbeddingIdentity = errors.New("embedding identity is incomplete")

	// ErrIncompatibleEmbeddingIdentity reports an attempted query across two
	// different semantic vector spaces.
	ErrIncompatibleEmbeddingIdentity = errors.New("embedding identities are incompatible")
)

// EmbeddingIdentity is the complete persisted identity of one semantic vector
// space. Every field participates in compatibility: equal dimensions alone do
// not make vectors comparable.
type EmbeddingIdentity struct {
	Model         string `json:"model"`
	ModelVersion  string `json:"model_version"`
	WeightsDigest string `json:"weights_digest"`
	Preprocessing string `json:"preprocessing"`
	Dimensions    int    `json:"dimensions"`
	Normalization string `json:"normalization"`
	Metric        string `json:"metric"`
}

// Validate rejects partial semantic identities before a provider can write or
// query a vector. Whitespace-only values are missing, not distinct identities.
func (id EmbeddingIdentity) Validate() error {
	required := []struct {
		name  string
		value string
	}{
		{"model", id.Model},
		{"model_version", id.ModelVersion},
		{"weights_digest", id.WeightsDigest},
		{"preprocessing", id.Preprocessing},
		{"normalization", id.Normalization},
		{"metric", id.Metric},
	}
	for _, field := range required {
		if strings.TrimSpace(field.value) == "" {
			return errors.Join(ErrIncompleteEmbeddingIdentity, fmt.Errorf("%s is required", field.name))
		}
	}
	if id.Dimensions <= 0 {
		return errors.Join(ErrIncompleteEmbeddingIdentity, errors.New("dimensions must be positive"))
	}
	return nil
}

// RequireCompatible validates both identities and rejects any field mismatch.
// Callers must run this before admitting stored vectors to one query.
func RequireCompatible(query, stored EmbeddingIdentity) error {
	if err := query.Validate(); err != nil {
		return fmt.Errorf("query identity: %w", err)
	}
	if err := stored.Validate(); err != nil {
		return fmt.Errorf("stored identity: %w", err)
	}
	if query != stored {
		return ErrIncompatibleEmbeddingIdentity
	}
	return nil
}

// Provider is the bundled lexical-fingerprint provider. It retains the
// semantic-facing shape used by Core, but Available is intentionally always
// false and EmbedQuery always fails closed because no trained model ships.
//
// enabled is retained so configuration remains forward-compatible; it permits
// explicit fingerprint work through LexicalFingerprint but cannot enable a
// semantic capability.
type Provider struct {
	enabled bool
}

// New constructs the bundled non-semantic provider.
func New(enabled bool) *Provider {
	return &Provider{enabled: enabled}
}

// Available reports semantic availability. The bundled provider is a hash, so
// it is never available to semantic indexing or search.
func (p *Provider) Available() bool { return false }

// FingerprintsAvailable reports whether the operator enabled deterministic
// non-semantic fingerprint work.
func (p *Provider) FingerprintsAvailable() bool { return p != nil && p.enabled }

// EmbedQuery refuses to emit a hash through the semantic query seam.
func (p *Provider) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := compactEmbeddingInput(query, MaxQueryInputBytes); err != nil {
		return nil, err
	}
	return nil, ErrSemanticProviderUnavailable
}

// compactEmbeddingInput trims and collapses whitespace, caps the result at a
// rune boundary, and rejects input with no lexical token.
func compactEmbeddingInput(text string, maxBytes int) (string, error) {
	text = strings.Join(strings.Fields(text), " ")
	if maxBytes > 0 && len(text) > maxBytes {
		cut := 0
		for i, r := range text {
			next := i + utf8.RuneLen(r)
			if next > maxBytes {
				break
			}
			cut = next
		}
		text = strings.TrimSpace(text[:cut])
	}
	if len(tokenize(text)) == 0 {
		return "", ErrEmptyInput
	}
	return text, nil
}

// LexicalFingerprint returns a deterministic, L2-normalized hashed-token
// fingerprint. It is not semantically meaningful and its KindString must not
// be stored in a semantic embedding index.
func LexicalFingerprint(text string) []float32 {
	vec := make([]float32, Dims)
	h := fnv.New32a()
	for _, tok := range tokenize(text) {
		h.Reset()
		_, _ = h.Write([]byte(tok))
		vec[h.Sum32()%uint32(Dims)]++
	}
	l2Normalize(vec)
	return vec
}

func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func l2Normalize(vec []float32) {
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm == 0 {
		return
	}
	for i := range vec {
		vec[i] /= norm
	}
}
