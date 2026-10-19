package enrichment

import (
	"bytes"
	"compress/zlib"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// --- deterministic, fixed-dimension lexical fingerprint --------------------

func TestLexicalFingerprintFixedDimension(t *testing.T) {
	vec := LexicalFingerprint("alpha beta gamma with repeated terms")
	if len(vec) != Dims {
		t.Fatalf("fingerprint dims = %d, want %d", len(vec), Dims)
	}
}

func TestDimsValue(t *testing.T) {
	const want = 256
	if Dims != want {
		t.Fatalf("fingerprint Dims = %d, want %d", Dims, want)
	}
}

func TestLexicalFingerprintDeterministic(t *testing.T) {
	const text = "stable sample text, low variance"
	a := LexicalFingerprint(text)
	b := LexicalFingerprint(text)
	if len(a) != len(b) {
		t.Fatalf("len mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("fingerprint not deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestLexicalFingerprintNormalized(t *testing.T) {
	vec := LexicalFingerprint("plain sample tokens")
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := math.Sqrt(sumSq)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Fatalf("fingerprint not L2-normalized: norm = %v, want 1.0", norm)
	}
}

func TestLexicalFingerprintEmptyIsZero(t *testing.T) {
	// Tokenizes to nothing → all-zero vector (cosine treats it as score 0).
	vec := LexicalFingerprint("   ,.;  ")
	for i, v := range vec {
		if v != 0 {
			t.Fatalf("expected zero vector for token-free text, got %v at %d", v, i)
		}
	}
}

func TestEmbedQueryCompactEdgeCases(t *testing.T) {
	longPrefix := strings.Repeat("a", MaxQueryInputBytes-1)
	exactBoundary := strings.Repeat("é", MaxQueryInputBytes/len("é"))

	tests := []struct {
		name    string
		input   string
		max     int
		want    string
		wantErr bool
	}{
		{
			name:  "multi-byte leading rune",
			input: "éclair",
			max:   len("é"),
			want:  "é",
		},
		{
			name:  "multi-byte rune straddling MaxQueryInputBytes",
			input: longPrefix + "é tail",
			max:   MaxQueryInputBytes,
			want:  longPrefix,
		},
		{
			name:    "whitespace-after-cut collapsing to empty",
			input:   " \t é tail",
			max:     1,
			wantErr: true,
		},
		{
			name:    "punctuation-only has no letter/digit token",
			input:   "  ,.;  --- ...  ",
			max:     MaxQueryInputBytes,
			wantErr: true,
		},
		{
			name:  "exact-boundary case",
			input: exactBoundary,
			max:   MaxQueryInputBytes,
			want:  exactBoundary,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := compactEmbeddingInput(tt.input, tt.max)
			if tt.wantErr {
				if !errors.Is(err, ErrEmptyInput) {
					t.Fatalf("compactEmbeddingInput error = %v, want ErrEmptyInput", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("compactEmbeddingInput error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("compactEmbeddingInput = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestL2NormalizeZeroVectorStaysZero(t *testing.T) {
	// Mirrors the production zero-guard directly (rather than only
	// transitively via LexicalFingerprint): an all-zero vector must be left as zeros,
	// never divided by a zero norm.
	vec := make([]float32, Dims)
	l2Normalize(vec)
	assertFiniteVector(t, vec)
	for i, v := range vec {
		if v != 0 {
			t.Fatalf("l2Normalize mutated a zero vector: got %v at %d", v, i)
		}
	}
}

func TestL2NormalizeUnitLength(t *testing.T) {
	vec := []float32{3, 4} // 3-4-5 triangle: norm 5
	l2Normalize(vec)
	assertFiniteVector(t, vec)
	want := []float32{0.6, 0.8}
	for i := range vec {
		if math.Abs(float64(vec[i]-want[i])) > 1e-6 {
			t.Fatalf("l2Normalize[%d] = %v, want %v", i, vec[i], want[i])
		}
	}
}

func TestL2NormalizeSubnormalSumNeverInfNaN(t *testing.T) {
	vec := []float32{math.SmallestNonzeroFloat32}
	l2Normalize(vec)
	assertFiniteVector(t, vec)
	if vec[0] != 1 {
		t.Fatalf("l2Normalize subnormal = %v, want 1", vec[0])
	}
}

func TestLexicalFingerprintSharedTokensRankHigher(t *testing.T) {
	// Shared tokens raise this fingerprint's cosine score. That deterministic
	// hash property is not a semantic quality claim.
	query := LexicalFingerprint("alpha beta gamma")
	near := LexicalFingerprint("alpha beta gamma delta")
	far := LexicalFingerprint("kappa lambda mu")
	if cosine(query, near) <= cosine(query, far) {
		t.Fatalf("token-overlapping text did not rank higher: near=%v far=%v",
			cosine(query, near), cosine(query, far))
	}
}

func TestEmbedQuerySanitizes(t *testing.T) {
	p := New(true)
	vec, err := p.EmbedQuery(context.Background(), "  hello   world  ")
	if !errors.Is(err, ErrSemanticProviderUnavailable) {
		t.Fatalf("EmbedQuery error = %v, want ErrSemanticProviderUnavailable", err)
	}
	if vec != nil {
		t.Fatalf("EmbedQuery emitted %d hash dimensions through semantic seam", len(vec))
	}
	// Blank input is rejected by the sanitizer with the exported sentinel so
	// callers (including Core) can errors.Is against it.
	if _, err := p.EmbedQuery(context.Background(), "   "); !errors.Is(err, ErrEmptyInput) {
		t.Fatalf("EmbedQuery blank input error = %v, want errors.Is(err, ErrEmptyInput)", err)
	}
}

func TestEmbedQueryCompactsOversizedInput(t *testing.T) {
	p := New(true)
	vec, err := p.EmbedQuery(context.Background(), strings.Repeat("space ", 200))
	if !errors.Is(err, ErrSemanticProviderUnavailable) {
		t.Fatalf("EmbedQuery error = %v, want ErrSemanticProviderUnavailable", err)
	}
	if vec != nil {
		t.Fatalf("EmbedQuery emitted %d hash dimensions through semantic seam", len(vec))
	}
}

func TestProviderKindAndAvailability(t *testing.T) {
	if KindString != "lexical-fingerprint-v1" {
		t.Fatalf("KindString = %q, want lexical fingerprint kind", KindString)
	}
	if New(true).Available() {
		t.Fatal("enabled hash provider advertised semantic availability")
	}
	if New(false).Available() {
		t.Fatal("disabled provider reports available")
	}
	if !New(true).FingerprintsAvailable() {
		t.Fatal("enabled provider reports fingerprints unavailable")
	}
	if New(false).FingerprintsAvailable() {
		t.Fatal("disabled provider reports fingerprints available")
	}
	if (*Provider)(nil).FingerprintsAvailable() {
		t.Fatal("nil provider reports fingerprints available")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := New(true).EmbedQuery(ctx, "valid"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled query error = %v, want context.Canceled", err)
	}
}

func TestEmbeddingIdentityRejectsIncompleteAndMixedSpaces(t *testing.T) {
	complete := EmbeddingIdentity{
		Model:         "sentence-transformer",
		ModelVersion:  "2.1.0",
		WeightsDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Preprocessing: "unicode-nfc-lower-v1",
		Dimensions:    384,
		Normalization: "l2",
		Metric:        "cosine",
	}
	if err := complete.Validate(); err != nil {
		t.Fatalf("complete identity rejected: %v", err)
	}

	incomplete := complete
	incomplete.WeightsDigest = ""
	if err := incomplete.Validate(); !errors.Is(err, ErrIncompleteEmbeddingIdentity) {
		t.Fatalf("incomplete identity error = %v, want ErrIncompleteEmbeddingIdentity", err)
	}
	if err := RequireCompatible(complete, incomplete); !errors.Is(err, ErrIncompleteEmbeddingIdentity) {
		t.Fatalf("incomplete stored identity error = %v, want ErrIncompleteEmbeddingIdentity", err)
	}
	incomplete = complete
	incomplete.Dimensions = 0
	if err := incomplete.Validate(); !errors.Is(err, ErrIncompleteEmbeddingIdentity) {
		t.Fatalf("zero-dimensional identity error = %v, want ErrIncompleteEmbeddingIdentity", err)
	}
	if err := RequireCompatible(incomplete, complete); !errors.Is(err, ErrIncompleteEmbeddingIdentity) {
		t.Fatalf("incomplete query identity error = %v, want ErrIncompleteEmbeddingIdentity", err)
	}

	incompatible := complete
	incompatible.Preprocessing = "unicode-nfc-case-v2"
	if err := RequireCompatible(complete, incompatible); !errors.Is(err, ErrIncompatibleEmbeddingIdentity) {
		t.Fatalf("mixed-space error = %v, want ErrIncompatibleEmbeddingIdentity", err)
	}
	if err := RequireCompatible(complete, complete); err != nil {
		t.Fatalf("identical spaces rejected: %v", err)
	}
}

func assertFiniteVector(t *testing.T, vec []float32) {
	t.Helper()
	for i, v := range vec {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Fatalf("vector[%d] = %v, want finite", i, v)
		}
	}
}

func TestPixelFingerprintAtLimitPeakRSS(t *testing.T) {
	if testing.Short() {
		t.Skip("resource bound")
	}
	if raceDetectorEnabled() {
		t.Skip("RSS ceiling is calibrated for non-instrumented binaries")
	}
	if fixture := os.Getenv("BEAMFALL_ENRICHMENT_RSS_FIXTURE"); fixture != "" {
		data, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatalf("read at-limit fixture: %v", err)
		}
		vec, err := NewImageProvider(true).PixelFingerprintReader(context.Background(), bytes.NewReader(data))
		if err != nil {
			t.Fatalf("at-limit fingerprint: %v", err)
		}
		if len(vec) != Dims {
			t.Fatalf("at-limit dimensions = %d, want %d", len(vec), Dims)
		}
		return
	}

	t.Run("decoded pixel limit", func(t *testing.T) {
		assertFingerprintPeakRSS(t, atLimitPNG(t))
	})
	t.Run("hostile IDAT fanout", func(t *testing.T) {
		data := hostileIDATFanoutPNG(t)
		if len(data) > maxImageBytes {
			t.Fatalf("hostile fixture bytes = %d, limit %d", len(data), maxImageBytes)
		}
		if maxImageBytes-len(data) >= len(testPNGChunk("IDAT", nil)) {
			t.Fatalf("hostile fixture leaves room for another empty IDAT: bytes=%d", len(data))
		}
		assertFingerprintPeakRSS(t, data)
	})
}

func raceDetectorEnabled() bool {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return false
	}
	for _, setting := range info.Settings {
		if setting.Key == "-race" && setting.Value == "true" {
			return true
		}
	}
	return false
}

func assertFingerprintPeakRSS(t *testing.T, data []byte) {
	t.Helper()
	fixture := t.TempDir() + "/at-limit.png"
	if err := os.WriteFile(fixture, data, 0o600); err != nil {
		t.Fatalf("write at-limit fixture: %v", err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	cmd := exec.Command(executable, "-test.run=^TestPixelFingerprintAtLimitPeakRSS$")
	cmd.Env = append(os.Environ(), "BEAMFALL_ENRICHMENT_RSS_FIXTURE="+fixture)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("isolated at-limit fixture: %v\n%s", err, output)
	}
	peak, ok := processPeakRSS(cmd.ProcessState)
	if !ok {
		t.Skipf("peak RSS unavailable on %s", runtime.GOOS)
	}
	const maxPeakRSS = 96 << 20
	if peak >= maxPeakRSS {
		t.Fatalf("isolated peak RSS = %d bytes, want below %d", peak, maxPeakRSS)
	}
	t.Logf("isolated peak RSS = %d bytes (limit %d)", peak, maxPeakRSS)
}

func hostileIDATFanoutPNG(t *testing.T) []byte {
	t.Helper()
	base := idatFanoutPNG(t, 0)
	emptyChunkBytes := len(testPNGChunk("IDAT", nil))
	emptyChunks := (maxImageBytes - len(base)) / emptyChunkBytes
	if emptyChunks < 1_000_000 {
		t.Fatalf("hostile fixture IDAT fanout = %d, want over one million", emptyChunks)
	}
	t.Logf("hostile fixture IDAT chunks = %d", emptyChunks+1)
	return idatFanoutPNG(t, emptyChunks)
}

func idatFanoutPNG(t *testing.T, emptyChunks int) []byte {
	t.Helper()
	var compressed bytes.Buffer
	zw := zlib.NewWriter(&compressed)
	if _, err := zw.Write([]byte{0, 0}); err != nil {
		t.Fatalf("compress fanout fixture: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close fanout compressor: %v", err)
	}
	header := testPNGChunk("IHDR", testIHDR(1, 1, 8, 0))
	empty := testPNGChunk("IDAT", nil)
	imageData := testPNGChunk("IDAT", compressed.Bytes())
	end := testPNGChunk("IEND", nil)
	size := len(pngSignature) + len(header) + emptyChunks*len(empty) + len(imageData) + len(end)
	data := make([]byte, 0, size)
	data = append(data, pngSignature...)
	data = append(data, header...)
	for range emptyChunks {
		data = append(data, empty...)
	}
	data = append(data, imageData...)
	data = append(data, end...)
	return data
}

func atLimitPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, 8192, 2048))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatalf("encode at-limit fixture: %v", err)
	}
	return encoded.Bytes()
}

// Reflection keeps this test compilable where process usage has no Maxrss.
func processPeakRSS(state *os.ProcessState) (uint64, bool) {
	if state == nil || state.SysUsage() == nil {
		return 0, false
	}
	usage := reflect.ValueOf(state.SysUsage())
	for usage.Kind() == reflect.Pointer {
		if usage.IsNil() {
			return 0, false
		}
		usage = usage.Elem()
	}
	if usage.Kind() != reflect.Struct {
		return 0, false
	}
	field := usage.FieldByName("Maxrss")
	switch field.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if field.Int() < 0 {
			return 0, false
		}
		return normalizeMaxRSS(field.Int(), runtime.GOOS), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return normalizeMaxRSS(int64(field.Uint()), runtime.GOOS), true
	default:
		return 0, false
	}
}

func normalizeMaxRSS(maxRSS int64, goos string) uint64 {
	if goos == "darwin" {
		return uint64(maxRSS)
	}
	return uint64(maxRSS) * 1024
}

func TestNormalizeMaxRSSUnits(t *testing.T) {
	if got := normalizeMaxRSS(7, "darwin"); got != 7 {
		t.Fatalf("darwin maxrss bytes = %d, want 7", got)
	}
	if got := normalizeMaxRSS(7, "linux"); got != 7*1024 {
		t.Fatalf("linux maxrss bytes = %d, want %d", got, 7*1024)
	}
}

func TestPNGParserValidationMatrix(t *testing.T) {
	ihdr := testIHDR(1, 1, 8, 2)
	complete := func(header []byte, metadata ...[]byte) []byte {
		chunks := []byte{}
		chunks = append(chunks, testPNGChunk("IHDR", header)...)
		for _, chunk := range metadata {
			chunks = append(chunks, chunk...)
		}
		chunks = append(chunks, testPNGChunk("IDAT", []byte{1})...)
		chunks = append(chunks, testPNGChunk("IEND", nil)...)
		return append(append([]byte(nil), pngSignature...), chunks...)
	}
	valid := testPNG(
		testPNGChunk("IHDR", ihdr),
		testPNGChunk("aaAa", nil),
		testPNGChunk("IDAT", []byte{1}),
		testPNGChunk("IEND", nil),
	)
	badCRC := append([]byte(nil), valid...)
	badCRC[len(pngSignature)+8] ^= 1
	tests := []struct {
		name string
		data []byte
	}{
		{"truncated header", append(append([]byte(nil), pngSignature...), 1)},
		{"truncated payload", append(append([]byte(nil), pngSignature...), 0, 0, 0, 9, 'I', 'H', 'D', 'R')},
		{"bad CRC", badCRC},
		{"IHDR not first", testPNG(testPNGChunk("aaAa", nil), testPNGChunk("IHDR", ihdr))},
		{"duplicate IHDR", testPNG(testPNGChunk("IHDR", ihdr), testPNGChunk("IHDR", ihdr))},
		{"bad methods", testPNG(testPNGChunk("IHDR", testIHDR(1, 1, 8, 2, 1)))},
		{"bad color depth", testPNG(testPNGChunk("IHDR", testIHDR(1, 1, 4, 2)))},
		{"empty palette", testPNG(testPNGChunk("IHDR", ihdr), testPNGChunk("PLTE", nil))},
		{"palette partial entry", complete(ihdr, testPNGChunk("PLTE", []byte{0}))},
		{"palette exceeds entry limit", complete(ihdr, testPNGChunk("PLTE", make([]byte, 257*3)))},
		{"palette before IHDR", testPNG(testPNGChunk("PLTE", []byte{0, 0, 0}))},
		{"transparency before IHDR", testPNG(testPNGChunk("tRNS", nil))},
		{"data before IHDR", testPNG(testPNGChunk("IDAT", []byte{1}))},
		{"IEND payload", testPNG(testPNGChunk("IHDR", ihdr), testPNGChunk("IEND", []byte{1}))},
		{"unknown critical", testPNG(testPNGChunk("IHDR", ihdr), testPNGChunk("ABCD", nil))},
		{"trailing bytes", append(valid, 1)},
		{"incomplete", testPNG(testPNGChunk("IHDR", ihdr), testPNGChunk("IEND", nil))},
		{"indexed without palette", testPNG(
			testPNGChunk("IHDR", testIHDR(1, 1, 8, 3)),
			testPNGChunk("IDAT", []byte{1}),
			testPNGChunk("IEND", nil),
		)},
		{"duplicate palette", complete(ihdr,
			testPNGChunk("PLTE", []byte{0, 0, 0}),
			testPNGChunk("PLTE", []byte{1, 1, 1}),
		)},
		{"palette forbidden for grayscale", complete(testIHDR(1, 1, 8, 0),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
		)},
		{"palette forbidden for grayscale alpha", complete(testIHDR(1, 1, 8, 4),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
		)},
		{"indexed palette exceeds bit depth", complete(testIHDR(1, 1, 1, 3),
			testPNGChunk("PLTE", []byte{0, 0, 0, 1, 1, 1, 2, 2, 2}),
		)},
		{"palette follows transparency", complete(ihdr,
			testPNGChunk("tRNS", []byte{0, 0, 0, 0, 0, 0}),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
		)},
		{"duplicate transparency", complete(ihdr,
			testPNGChunk("tRNS", []byte{0, 0, 0, 0, 0, 0}),
			testPNGChunk("tRNS", []byte{0, 0, 0, 0, 0, 0}),
		)},
		{"transparency forbidden for grayscale alpha", complete(testIHDR(1, 1, 8, 4),
			testPNGChunk("tRNS", []byte{0, 0}),
		)},
		{"transparency forbidden for truecolor alpha", complete(testIHDR(1, 1, 8, 6),
			testPNGChunk("tRNS", []byte{0, 0, 0, 0, 0, 0}),
		)},
		{"grayscale transparency wrong length", complete(testIHDR(1, 1, 8, 0),
			testPNGChunk("tRNS", []byte{0}),
		)},
		{"grayscale transparency exceeds bit depth", complete(testIHDR(1, 1, 1, 0),
			testPNGChunk("tRNS", []byte{0, 2}),
		)},
		{"truecolor transparency wrong length", complete(ihdr,
			testPNGChunk("tRNS", []byte{0, 0}),
		)},
		{"truecolor transparency exceeds bit depth", complete(ihdr,
			testPNGChunk("tRNS", []byte{1, 0, 0, 0, 0, 0}),
		)},
		{"indexed transparency precedes palette", complete(testIHDR(1, 1, 8, 3),
			testPNGChunk("tRNS", []byte{0}),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
		)},
		{"indexed transparency empty", complete(testIHDR(1, 1, 8, 3),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
			testPNGChunk("tRNS", nil),
		)},
		{"indexed transparency exceeds palette", complete(testIHDR(1, 1, 8, 3),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
			testPNGChunk("tRNS", []byte{0, 1}),
		)},
		{"palette after image data", testPNG(
			testPNGChunk("IHDR", ihdr),
			testPNGChunk("IDAT", []byte{1}),
			testPNGChunk("PLTE", []byte{0, 0, 0}),
			testPNGChunk("IEND", nil),
		)},
		{"transparency after image data", testPNG(
			testPNGChunk("IHDR", ihdr),
			testPNGChunk("IDAT", []byte{1}),
			testPNGChunk("tRNS", []byte{0, 0, 0, 0, 0, 0}),
			testPNGChunk("IEND", nil),
		)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseFingerprintPNG(context.Background(), tt.data); err == nil {
				t.Fatal("malformed PNG accepted")
			}
		})
	}
	if _, err := parseFingerprintPNG(context.Background(), valid); err != nil {
		t.Fatalf("valid ancillary chunk rejected: %v", err)
	}
	withPalette := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 3)),
		testPNGChunk("PLTE", []byte{1, 2, 3}),
		testPNGChunk("tRNS", []byte{127}),
		testPNGChunk("IDAT", []byte{1}),
		testPNGChunk("IEND", nil),
	)
	if _, err := parseFingerprintPNG(context.Background(), withPalette); err != nil {
		t.Fatalf("valid palette metadata rejected: %v", err)
	}
	validMetadata := []struct {
		name string
		data []byte
	}{
		{
			name: "grayscale transparency",
			data: complete(testIHDR(1, 1, 1, 0),
				testPNGChunk("tRNS", []byte{0, 1}),
			),
		},
		{
			name: "truecolor palette and transparency",
			data: complete(ihdr,
				testPNGChunk("PLTE", []byte{0, 0, 0}),
				testPNGChunk("tRNS", []byte{0, 1, 0, 2, 0, 3}),
			),
		},
		{
			name: "truecolor alpha palette",
			data: complete(testIHDR(1, 1, 8, 6),
				testPNGChunk("PLTE", []byte{0, 0, 0}),
			),
		},
		{
			name: "sixteen bit truecolor transparency",
			data: complete(testIHDR(1, 1, 16, 2),
				testPNGChunk("tRNS", []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}),
			),
		},
	}
	for _, tt := range validMetadata {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseFingerprintPNG(context.Background(), tt.data); err != nil {
				t.Fatalf("valid PNG metadata rejected: %v", err)
			}
		})
	}
}

func TestPNGIDATFanoutStreamingAndMutations(t *testing.T) {
	const emptyChunks = pixelCancellationInterval + 1
	valid := idatFanoutPNG(t, emptyChunks)
	got, err := pixelFingerprintPNG(context.Background(), valid)
	if err != nil {
		t.Fatalf("valid IDAT fanout rejected: %v", err)
	}
	want, err := pixelFingerprintPNG(context.Background(), idatFanoutPNG(t, 0))
	if err != nil {
		t.Fatalf("single IDAT fixture rejected: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("IDAT framing changed the pixel fingerprint")
	}

	firstIDAT := len(pngSignature) + len(testPNGChunk("IHDR", testIHDR(1, 1, 8, 0)))
	badCRC := append([]byte(nil), valid...)
	badCRC[firstIDAT+8] ^= 1
	nonConsecutive := append([]byte(nil), valid...)
	secondIDAT := firstIDAT + len(testPNGChunk("IDAT", nil))
	copy(nonConsecutive[secondIDAT+4:secondIDAT+8], "aaAa")
	binary.BigEndian.PutUint32(
		nonConsecutive[secondIDAT+8:secondIDAT+12],
		crc32.ChecksumIEEE(nonConsecutive[secondIDAT+4:secondIDAT+8]),
	)
	badBoundary := append([]byte(nil), valid...)
	binary.BigEndian.PutUint32(badBoundary[firstIDAT:firstIDAT+4], 1)

	for _, mutation := range []struct {
		name string
		data []byte
	}{
		{name: "empty chunk CRC", data: badCRC},
		{name: "non-consecutive image data", data: nonConsecutive},
		{name: "empty chunk length", data: badBoundary},
	} {
		t.Run(mutation.name, func(t *testing.T) {
			if _, err := pixelFingerprintPNG(context.Background(), mutation.data); err == nil {
				t.Fatal("mutated IDAT fanout accepted")
			}
		})
	}
}

type cancelAfterContextChecks struct {
	context.Context
	checks   int
	cancelAt int
}

func (c *cancelAfterContextChecks) Err() error {
	c.checks++
	if c.checks >= c.cancelAt {
		return context.Canceled
	}
	return nil
}

func TestPNGIDATFanoutParserCancellationCheckpoint(t *testing.T) {
	ctx := &cancelAfterContextChecks{
		Context:  context.Background(),
		cancelAt: 2,
	}
	_, err := parseFingerprintPNG(ctx, idatFanoutPNG(t, pixelCancellationInterval+1))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("fanout parser cancellation error = %v, want context.Canceled", err)
	}
	if ctx.checks != 2 {
		t.Fatalf("fanout parser context checks = %d, want 2", ctx.checks)
	}
}

func TestPNGSingleIDATParserCancellationCheckpoint(t *testing.T) {
	data := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 0)),
		testPNGChunk("IDAT", make([]byte, 1<<20)),
		testPNGChunk("IEND", nil),
	)
	ctx := &cancelAfterContextChecks{
		Context:  context.Background(),
		cancelAt: 2,
	}
	_, err := parseFingerprintPNG(ctx, data)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("single-IDAT parser cancellation error = %v, want context.Canceled", err)
	}
	if ctx.checks != 2 {
		t.Fatalf("single-IDAT parser context checks = %d, want 2", ctx.checks)
	}
}

func TestPNGSingleIDATReaderCancellationCheckpoint(t *testing.T) {
	data := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 0)),
		testPNGChunk("IDAT", make([]byte, pixelCancellationInterval*2)),
		testPNGChunk("IEND", nil),
	)
	spec, err := parseFingerprintPNG(context.Background(), data)
	if err != nil {
		t.Fatalf("parse single-IDAT fixture: %v", err)
	}
	ctx := &cancelAfterContextChecks{
		Context:  context.Background(),
		cancelAt: 2,
	}
	reader := newPNGIDATReader(ctx, spec.data, spec.idatStart, spec.idatEnd)
	buf := make([]byte, pixelCancellationInterval)
	n, err := reader.Read(buf)
	if n == 0 || err != nil {
		t.Fatalf("first single-IDAT read = %d, %v", n, err)
	}
	if n > pixelCancellationInterval {
		t.Fatalf("first single-IDAT read = %d bytes, checkpoint %d", n, pixelCancellationInterval)
	}
	if _, err := reader.Read(buf); !errors.Is(err, context.Canceled) {
		t.Fatalf("second single-IDAT read error = %v, want context.Canceled", err)
	}
	if ctx.checks != 2 {
		t.Fatalf("single-IDAT reader context checks = %d, want 2", ctx.checks)
	}
}

func TestPNGIDATReaderCancellationAndEmptyReadEdges(t *testing.T) {
	chunk := testPNGChunk("IDAT", []byte{1})
	reader := newPNGIDATReader(context.Background(), chunk, 0, len(chunk))
	if n, err := reader.Read(nil); n != 0 || err != nil {
		t.Fatalf("empty IDAT read = %d, %v", n, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reader = newPNGIDATReader(ctx, chunk, 0, len(chunk))
	if _, err := reader.hasRemainingData(); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled remaining-data check = %v, want context.Canceled", err)
	}
}

func testIHDR(width, height uint32, bitDepth, colorType byte, badMethod ...byte) []byte {
	payload := make([]byte, 13)
	binary.BigEndian.PutUint32(payload[0:4], width)
	binary.BigEndian.PutUint32(payload[4:8], height)
	payload[8], payload[9] = bitDepth, colorType
	if len(badMethod) != 0 {
		payload[10] = badMethod[0]
	}
	return payload
}

func testPNG(chunks ...[]byte) []byte {
	data := append([]byte(nil), pngSignature...)
	for _, chunk := range chunks {
		data = append(data, chunk...)
	}
	return data
}

func testPNGChunk(kind string, payload []byte) []byte {
	chunk := make([]byte, 12+len(payload))
	binary.BigEndian.PutUint32(chunk[0:4], uint32(len(payload)))
	copy(chunk[4:8], kind)
	copy(chunk[8:], payload)
	binary.BigEndian.PutUint32(chunk[8+len(payload):], crc32.ChecksumIEEE(chunk[4:8+len(payload)]))
	return chunk
}

func TestPNGDecodeHelpers(t *testing.T) {
	for _, tt := range []struct {
		colorType byte
		depth     int
		channels  int
		ok        bool
	}{
		{0, 1, 1, true}, {0, 16, 1, true}, {2, 8, 3, true},
		{3, 4, 1, true}, {4, 8, 2, true}, {6, 16, 4, true},
		{2, 4, 3, false}, {5, 8, 0, false},
	} {
		got, ok := pngChannels(tt.colorType, tt.depth)
		if got != tt.channels || ok != tt.ok {
			t.Fatalf("pngChannels(%d,%d) = %d,%v", tt.colorType, tt.depth, got, ok)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := readPNGFilterByte(ctx, bytes.NewReader([]byte{0})); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled filter read = %v", err)
	}
	if _, err := readPNGFilterByte(context.Background(), bytes.NewReader(nil)); !errors.Is(err, io.EOF) {
		t.Fatalf("empty filter read = %v", err)
	}
	cancelReader := &cancelAfterFirstRead{cancel: func() {}}
	cancelCtx, cancelAfterRead := context.WithCancel(context.Background())
	cancelReader.cancel = cancelAfterRead
	if _, err := readPNGFilterByte(cancelCtx, cancelReader); !errors.Is(err, context.Canceled) {
		t.Fatalf("post-read cancellation = %v", err)
	}

	if err := readPNGPixelBytes(context.Background(), bytes.NewReader(nil), make([]byte, 1), 8); !errors.Is(err, io.EOF) {
		t.Fatalf("truncated pixel bytes = %v", err)
	}
	if err := readPNGPixelBytes(context.Background(), bytes.NewReader([]byte{1}), make([]byte, 1), 0); err != nil {
		t.Fatalf("zero-bit guarded chunk = %v", err)
	}
	cancelCtx, cancel = context.WithCancel(context.Background())
	cancel()
	if err := readPNGPixelBytes(cancelCtx, bytes.NewReader([]byte{1}), make([]byte, 1), 8); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled pixel bytes = %v", err)
	}

	for filter := byte(0); filter <= 4; filter++ {
		row := []byte{1, 2, 3, 4}
		if err := unfilterPNGRow(context.Background(), row, []byte{4, 3, 2, 1}, 1, filter); err != nil {
			t.Fatalf("filter %d: %v", filter, err)
		}
	}
	if err := unfilterPNGRow(context.Background(), []byte{1}, []byte{0}, 1, 5); err == nil {
		t.Fatal("invalid filter accepted")
	}
	if err := unfilterPNGRow(cancelCtx, []byte{1}, []byte{0}, 1, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled unfilter = %v", err)
	}
	if paeth(10, 2, 1) != 10 || paeth(2, 10, 1) != 10 || paeth(10, 100, 50) != 50 {
		t.Fatal("Paeth branch fixture mismatch")
	}
	if absInt(-3) != 3 || absInt(3) != 3 {
		t.Fatal("absolute value mismatch")
	}
}

func TestPNGPixelFormats(t *testing.T) {
	tests := []struct {
		name string
		spec pngFingerprintSpec
		row  []byte
		err  bool
	}{
		{"gray1 transparent", pngFingerprintSpec{bitDepth: 1, colorType: 0, channels: 1, transparency: []byte{0, 1}}, []byte{0x80}, false},
		{"gray16", pngFingerprintSpec{bitDepth: 16, colorType: 0, channels: 1}, []byte{0x12, 0x34}, false},
		{"rgb transparent", pngFingerprintSpec{bitDepth: 8, colorType: 2, channels: 3, transparency: []byte{0, 1, 0, 2, 0, 3}}, []byte{1, 2, 3}, false},
		{"palette alpha", pngFingerprintSpec{bitDepth: 2, colorType: 3, channels: 1, palette: []byte{1, 2, 3, 4, 5, 6}, transparency: []byte{255, 127}}, []byte{0x40}, false},
		{"bad palette", pngFingerprintSpec{bitDepth: 8, colorType: 3, channels: 1, palette: []byte{1, 2, 3}}, []byte{1}, true},
		{"gray alpha", pngFingerprintSpec{bitDepth: 8, colorType: 4, channels: 2}, []byte{100, 127}, false},
		{"rgba16", pngFingerprintSpec{bitDepth: 16, colorType: 6, channels: 4}, []byte{0xff, 0xff, 0, 0, 0, 0, 0x80, 0}, false},
		{"unknown", pngFingerprintSpec{bitDepth: 8, colorType: 5, channels: 1}, []byte{0}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := tt.spec.pixel(tt.row, 0)
			if (err != nil) != tt.err {
				t.Fatalf("pixel error = %v, want error %v", err, tt.err)
			}
		})
	}
}

func TestPNGStreamingFailureMatrix(t *testing.T) {
	zdata := func(raw []byte) []byte {
		var out bytes.Buffer
		zw := zlib.NewWriter(&out)
		if _, err := zw.Write(raw); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		return out.Bytes()
	}
	rgbPNG := func(idat []byte, extra ...[]byte) []byte {
		chunks := [][]byte{testPNGChunk("IHDR", testIHDR(1, 1, 8, 2))}
		chunks = append(chunks, extra...)
		chunks = append(chunks, testPNGChunk("IDAT", idat), testPNGChunk("IEND", nil))
		return testPNG(chunks...)
	}
	corruptChecksum := zdata([]byte{0, 1, 2, 3})
	corruptChecksum[len(corruptChecksum)-1] ^= 1
	validStream := zdata([]byte{0, 1, 2, 3})
	trailingCompressedJunk := append(append([]byte(nil), validStream...), 0xde, 0xad)
	secondZlibStream := append(append([]byte(nil), validStream...), validStream...)
	trailingJunkChunk := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 2)),
		testPNGChunk("IDAT", validStream),
		testPNGChunk("IDAT", []byte{0xde, 0xad}),
		testPNGChunk("IEND", nil),
	)
	secondStreamChunk := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 2)),
		testPNGChunk("IDAT", validStream),
		testPNGChunk("IDAT", validStream),
		testPNGChunk("IEND", nil),
	)
	paletteBadIndex := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 3)),
		testPNGChunk("PLTE", []byte{1, 2, 3}),
		testPNGChunk("IDAT", zdata([]byte{0, 1})),
		testPNGChunk("IEND", nil),
	)
	for _, tt := range []struct {
		name string
		data []byte
	}{
		{"bad zlib", rgbPNG([]byte{1})},
		{"missing filter", rgbPNG(zdata(nil))},
		{"truncated row", rgbPNG(zdata([]byte{0}))},
		{"bad filter", rgbPNG(zdata([]byte{5, 1, 2, 3}))},
		{"bad palette index", paletteBadIndex},
		{"trailing decoded byte", rgbPNG(zdata([]byte{0, 1, 2, 3, 4}))},
		{"bad zlib checksum", rgbPNG(corruptChecksum)},
		{"trailing compressed junk", rgbPNG(trailingCompressedJunk)},
		{"second zlib stream", rgbPNG(secondZlibStream)},
		{"trailing compressed junk in next IDAT", trailingJunkChunk},
		{"second zlib stream in next IDAT", secondStreamChunk},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := pixelFingerprintPNG(context.Background(), tt.data); err == nil {
				t.Fatal("invalid streaming PNG accepted")
			}
		})
	}
	trailingEmptyIDAT := testPNG(
		testPNGChunk("IHDR", testIHDR(1, 1, 8, 2)),
		testPNGChunk("IDAT", validStream),
		testPNGChunk("IDAT", nil),
		testPNGChunk("IEND", nil),
	)
	if _, err := pixelFingerprintPNG(context.Background(), trailingEmptyIDAT); err != nil {
		t.Fatalf("valid trailing empty IDAT rejected: %v", err)
	}
}

func TestPixelFingerprintTinyImageEmptyCells(t *testing.T) {
	vec, err := PixelFingerprintContext(context.Background(), image.NewGray(image.Rect(0, 0, 1, 1)))
	if err != nil || len(vec) != Dims {
		t.Fatalf("tiny fingerprint: dims=%d err=%v", len(vec), err)
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
