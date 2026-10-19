package enrichment

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"sync"
	"testing"
	"time"
)

func solidImage(w, h int, c color.Color) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func splitImage(w, h int, topC, botC color.Color) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		c := topC
		if y >= h/2 {
			c = botC
		}
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

func pngBytes(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return buf.Bytes()
}

func TestPixelFingerprintFixedDimension(t *testing.T) {
	vec := PixelFingerprint(splitImage(64, 64, color.White, color.Black))
	if len(vec) != Dims {
		t.Fatalf("pixel fingerprint dims = %d, want %d", len(vec), Dims)
	}
}

func TestPixelFingerprintDeterministic(t *testing.T) {
	img := splitImage(48, 48, color.RGBA{200, 30, 30, 255}, color.RGBA{20, 20, 90, 255})
	a := PixelFingerprint(img)
	b := PixelFingerprint(img)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("pixel fingerprint not deterministic at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestPixelFingerprintNormalized(t *testing.T) {
	vec := PixelFingerprint(splitImage(40, 40, color.White, color.RGBA{10, 10, 10, 255}))
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := math.Sqrt(sumSq)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Fatalf("pixel fingerprint not L2-normalized: norm = %v, want 1.0", norm)
	}
}

func TestFingerprintWidthsDoNotImplySemanticCompatibility(t *testing.T) {
	imgVec := PixelFingerprint(solidImage(32, 32, color.White))
	txtVec := LexicalFingerprint("beach sky sand")
	if len(imgVec) != len(txtVec) {
		t.Fatalf("pixel fingerprint dims %d != lexical fingerprint dims %d", len(imgVec), len(txtVec))
	}
}

func TestPixelFingerprintSimilarStructureRanksHigher(t *testing.T) {
	query := PixelFingerprint(splitImage(64, 64, color.White, color.Black))
	near := PixelFingerprint(splitImage(64, 64, color.RGBA{240, 240, 240, 255}, color.RGBA{16, 16, 16, 255}))
	far := PixelFingerprint(solidImage(64, 64, color.RGBA{200, 30, 30, 255}))
	if cosine(query, near) <= cosine(query, far) {
		t.Fatalf("structurally-similar image did not rank higher: near=%v far=%v",
			cosine(query, near), cosine(query, far))
	}
}

func TestPixelFingerprintRejectsZeroArea(t *testing.T) {
	vec, err := PixelFingerprintContext(context.Background(), image.NewRGBA(image.Rect(0, 0, 0, 0)))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("zero-area error = %v, want ErrNoImageData", err)
	}
	if vec != nil {
		t.Fatalf("zero-area image produced %d dimensions", len(vec))
	}
}

func TestPixelFingerprintReaderDecodesPNG(t *testing.T) {
	p := NewImageProvider(true)
	data := pngBytes(t, splitImage(32, 32, color.White, color.Black))
	vec, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader(data))
	if err != nil {
		t.Fatalf("PixelFingerprintReader: %v", err)
	}
	if len(vec) != Dims {
		t.Fatalf("PixelFingerprintReader dims = %d, want %d", len(vec), Dims)
	}
	// Decoded-then-embedded must equal directly-embedded (same pixels).
	want := PixelFingerprint(splitImage(32, 32, color.White, color.Black))
	for i := range vec {
		if vec[i] != want[i] {
			t.Fatalf("reader path diverged from PixelFingerprint at %d", i)
		}
	}
}

func TestPixelFingerprintReaderUndecodableIsSentinel(t *testing.T) {
	p := NewImageProvider(true)
	_, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader([]byte("not an image")))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("EmbedImageReader error = %v, want errors.Is(err, ErrNoImageData)", err)
	}
	_, err = p.PixelFingerprintReader(context.Background(), bytes.NewReader(nil))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("empty reader error = %v, want ErrNoImageData", err)
	}
	_, err = p.PixelFingerprintReader(context.Background(), errorReader{})
	if !errors.Is(err, errFixtureRead) {
		t.Fatalf("reader failure error = %v, want fixture error", err)
	}
}

var errFixtureRead = errors.New("fixture read failed")

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errFixtureRead }

func FuzzEmbedImageReader(f *testing.F) {
	var seed bytes.Buffer
	if err := png.Encode(&seed, solidImage(1, 1, color.White)); err != nil {
		f.Fatal(err)
	}
	f.Add(seed.Bytes())
	f.Add([]byte("not an image"))

	p := NewImageProvider(true)
	f.Fuzz(func(t *testing.T, data []byte) {
		vec, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader(data))
		if err != nil {
			return
		}
		if len(vec) != Dims {
			t.Fatalf("successful decode returned %d dimensions, want %d", len(vec), Dims)
		}
		assertFiniteVector(t, vec)
	})
}

func TestPixelFingerprintReaderNilReaderIsSentinel(t *testing.T) {
	p := NewImageProvider(true)
	_, err := p.PixelFingerprintReader(context.Background(), nil)
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("EmbedImageReader(nil) error = %v, want errors.Is(err, ErrNoImageData)", err)
	}
}

func TestPixelFingerprintReaderRejectsZeroDimensionHeader(t *testing.T) {
	p := NewImageProvider(true)
	zeroDims := []byte("GIF89a\x00\x00\x00\x00\x00\x00\x00")
	_, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader(zeroDims))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("EmbedImageReader error = %v, want errors.Is(err, ErrNoImageData)", err)
	}
}

func TestPixelFingerprintReaderRejectsTruncatedDecode(t *testing.T) {
	p := NewImageProvider(true)
	data := pngBytes(t, solidImage(64, 64, color.White))
	data = data[:len(data)/2]
	_, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader(data))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("truncated decode error = %v, want ErrNoImageData", err)
	}
}

func TestCheckDecodedDims(t *testing.T) {
	tests := []struct {
		name    string
		w, h    int
		wantErr bool
	}{
		{name: "typical preview", w: 1024, h: 768},
		{name: "at the pixel cap boundary", w: 8192, h: 2048}, // 8192*2048 == maxImagePixels
		{name: "just over the pixel cap", w: 8192, h: 2049, wantErr: true},
		{name: "just over the dimension cap", w: 8193, h: 1, wantErr: true},
		{name: "zero width", w: 0, h: 100, wantErr: true},
		{name: "zero height", w: 100, h: 0, wantErr: true},
		{name: "negative dimension", w: -1, h: 16, wantErr: true},
		{name: "int32-limit header must not overflow", w: math.MaxInt32, h: math.MaxInt32, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkDecodedDims(tt.w, tt.h)
			if (err != nil) != tt.wantErr {
				t.Fatalf("checkDecodedDims(%d, %d) error = %v, wantErr %v", tt.w, tt.h, err, tt.wantErr)
			}
		})
	}
}

func decodeBombPNG(t *testing.T, width, height uint32) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.White)
	data := pngBytes(t, img)

	// PNG framing: 8-byte signature, then chunks of
	// [length(4)][type(4)][data(length)][crc(4)]. The first chunk is IHDR,
	// whose 13-byte payload is width(4) height(4) + 5 more fields.
	const sigLen = 8
	lengthOff := sigLen
	typeOff := sigLen + 4
	dataOff := sigLen + 8
	length := binary.BigEndian.Uint32(data[lengthOff : lengthOff+4])
	if string(data[typeOff:typeOff+4]) != "IHDR" || length != 13 {
		t.Fatalf("unexpected PNG framing while building decode-bomb fixture")
	}

	binary.BigEndian.PutUint32(data[dataOff:dataOff+4], width)
	binary.BigEndian.PutUint32(data[dataOff+4:dataOff+8], height)

	crcOff := dataOff + int(length)
	sum := crc32.ChecksumIEEE(data[typeOff : typeOff+4+int(length)])
	binary.BigEndian.PutUint32(data[crcOff:crcOff+4], sum)

	return data
}

func TestPixelFingerprintReaderRejectsDecodeBombDimensions(t *testing.T) {
	p := NewImageProvider(true)
	bomb := decodeBombPNG(t, 100000, 100000)
	_, err := p.PixelFingerprintReader(context.Background(), bytes.NewReader(bomb))
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("EmbedImageReader error = %v, want errors.Is(err, ErrNoImageData)", err)
	}
}

func TestPixelFingerprintReaderRejectsOversizedByteInput(t *testing.T) {
	p := NewImageProvider(true)
	huge := io.LimitReader(zeroReader{}, maxImageBytes+1)
	_, err := p.PixelFingerprintReader(context.Background(), huge)
	if !errors.Is(err, ErrNoImageData) {
		t.Fatalf("EmbedImageReader error = %v, want errors.Is(err, ErrNoImageData)", err)
	}
}

type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func TestPixelFingerprintBatchPreservesOrder(t *testing.T) {
	p := NewImageProvider(true)
	a := pngBytes(t, solidImage(24, 24, color.White))
	b := pngBytes(t, solidImage(24, 24, color.Black))
	items := []BatchItem{
		{Key: "fp-a", Reader: bytes.NewReader(a)},
		{Key: "fp-b", Reader: bytes.NewReader(b)},
	}
	results, err := p.PixelFingerprintBatch(context.Background(), items)
	if err != nil {
		t.Fatalf("EmbedImageBatch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results = %d, want 2", len(results))
	}
	if results[0].Key != "fp-a" || results[1].Key != "fp-b" {
		t.Fatalf("results out of order: %q, %q", results[0].Key, results[1].Key)
	}
	for _, r := range results {
		if r.Err != nil {
			t.Fatalf("item %q errored: %v", r.Key, r.Err)
		}
		if len(r.Vec) != Dims {
			t.Fatalf("item %q dims = %d, want %d", r.Key, len(r.Vec), Dims)
		}
	}
}

func TestPixelFingerprintBatchIsolatesPerItemDecodeError(t *testing.T) {
	p := NewImageProvider(true)
	good := pngBytes(t, solidImage(16, 16, color.White))
	items := []BatchItem{
		{Key: "bad", Reader: bytes.NewReader([]byte("corrupt"))},
		{Key: "good", Reader: bytes.NewReader(good)},
	}
	results, err := p.PixelFingerprintBatch(context.Background(), items)
	if err != nil {
		t.Fatalf("EmbedImageBatch returned batch error for a per-item failure: %v", err)
	}
	if !errors.Is(results[0].Err, ErrNoImageData) {
		t.Fatalf("bad item error = %v, want ErrNoImageData", results[0].Err)
	}
	if results[1].Err != nil || len(results[1].Vec) != Dims {
		t.Fatalf("good item after a bad one: err=%v dims=%d", results[1].Err, len(results[1].Vec))
	}
}

func TestPixelFingerprintBatchIsolatesNilReader(t *testing.T) {
	p := NewImageProvider(true)
	good := pngBytes(t, solidImage(16, 16, color.White))
	items := []BatchItem{
		{Key: "nil", Reader: nil},
		{Key: "good", Reader: bytes.NewReader(good)},
	}
	results, err := p.PixelFingerprintBatch(context.Background(), items)
	if err != nil {
		t.Fatalf("EmbedImageBatch returned batch error for a nil-reader item: %v", err)
	}
	if !errors.Is(results[0].Err, ErrNoImageData) {
		t.Fatalf("nil-reader item error = %v, want ErrNoImageData", results[0].Err)
	}
	if results[1].Err != nil || len(results[1].Vec) != Dims {
		t.Fatalf("good item after a nil-reader one: err=%v dims=%d", results[1].Err, len(results[1].Vec))
	}
}

func TestPixelFingerprintBatchHonorsCancellation(t *testing.T) {
	p := NewImageProvider(true)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	good := pngBytes(t, solidImage(16, 16, color.White))
	items := []BatchItem{{Key: "x", Reader: bytes.NewReader(good)}}
	results, err := p.PixelFingerprintBatch(ctx, items)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("EmbedImageBatch cancelled error = %v, want context.Canceled", err)
	}
	if len(results) != 0 {
		t.Fatalf("cancelled batch produced %d results, want 0", len(results))
	}
}

func TestPixelFingerprintBatchRejectsOversizedCallBeforeIO(t *testing.T) {
	p := NewImageProvider(true)
	reader := &countingReader{Reader: bytes.NewReader(pngBytes(t, solidImage(1, 1, color.White)))}
	items := make([]BatchItem, MaxFingerprintBatchItems+1)
	for i := range items {
		items[i] = BatchItem{Key: "oversized", Reader: reader}
	}
	results, err := p.PixelFingerprintBatch(context.Background(), items)
	if !errors.Is(err, ErrFingerprintBatchTooLarge) {
		t.Fatalf("oversized batch error = %v, want ErrFingerprintBatchTooLarge", err)
	}
	if results != nil {
		t.Fatalf("oversized batch allocated %d results, want nil", len(results))
	}
	if reader.reads != 0 {
		t.Fatalf("oversized batch performed %d reads before rejection", reader.reads)
	}
}

func TestImageProviderKindAndAvailability(t *testing.T) {
	if ImageKindString != "pixel-fingerprint-v1" {
		t.Fatalf("ImageKindString = %q, want pixel fingerprint kind", ImageKindString)
	}
	if ImageKindString == KindString {
		t.Fatalf("image kind must differ from text kind %q so both coexist per Media", KindString)
	}
	if NewImageProvider(true).Available() {
		t.Fatal("enabled pixel hash provider advertised semantic availability")
	}
	if NewImageProvider(false).Available() {
		t.Fatal("disabled image provider reports available")
	}
	if !NewImageProvider(true).FingerprintsAvailable() {
		t.Fatal("enabled image provider reports fingerprints unavailable")
	}
	if (*ImageProvider)(nil).FingerprintsAvailable() {
		t.Fatal("nil image provider reports fingerprints available")
	}
}

func TestImageSemanticSeamsFailClosedWithoutReading(t *testing.T) {
	p := NewImageProvider(true)
	reader := &countingReader{Reader: bytes.NewReader(pngBytes(t, solidImage(1, 1, color.White)))}
	vec, err := p.EmbedImageReader(context.Background(), reader)
	if !errors.Is(err, ErrSemanticProviderUnavailable) {
		t.Fatalf("semantic reader error = %v, want ErrSemanticProviderUnavailable", err)
	}
	if vec != nil || reader.reads != 0 {
		t.Fatalf("semantic reader emitted vec=%v or consumed %d reads", vec != nil, reader.reads)
	}
	results, err := p.EmbedImageBatch(context.Background(), []BatchItem{{Key: "x", Reader: reader}})
	if !errors.Is(err, ErrSemanticProviderUnavailable) {
		t.Fatalf("semantic batch error = %v, want ErrSemanticProviderUnavailable", err)
	}
	if results != nil || reader.reads != 0 {
		t.Fatalf("semantic batch emitted results=%v or consumed %d reads", results != nil, reader.reads)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.EmbedImageReader(ctx, reader); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled semantic reader error = %v, want context.Canceled", err)
	}
	if _, err := p.EmbedImageBatch(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled semantic batch error = %v, want context.Canceled", err)
	}
}

type countingReader struct {
	io.Reader
	reads int
}

func (r *countingReader) Read(p []byte) (int, error) {
	r.reads++
	return r.Reader.Read(p)
}

func TestPixelFingerprintCancellationCheckpoint(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	img := &cancelOnFirstPixel{
		Image:  image.NewGray(image.Rect(0, 0, 8192, 2048)),
		cancel: cancel,
	}
	vec, err := PixelFingerprintContext(ctx, img)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("fingerprint cancellation error = %v, want context.Canceled", err)
	}
	if vec != nil {
		t.Fatalf("cancelled fingerprint emitted %d dimensions", len(vec))
	}
	if img.calls > pixelCancellationInterval {
		t.Fatalf("cancellation observed after %d pixels, limit %d", img.calls, pixelCancellationInterval)
	}
}

func TestPNGDecodeCancellationCheckpoint(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader := &cancelAfterFirstRead{cancel: cancel}
	const bitsPerPixel = 24
	row := make([]byte, 8192*bitsPerPixel/8)
	err := readPNGPixelBytes(ctx, reader, row, bitsPerPixel)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("decode cancellation error = %v, want context.Canceled", err)
	}
	const maxDecodeBytes = pixelCancellationInterval * bitsPerPixel / 8
	if reader.maxRead > maxDecodeBytes {
		t.Fatalf("decoder requested %d bytes, representing more than %d pixels", reader.maxRead, pixelCancellationInterval)
	}
	if reader.reads != 1 {
		t.Fatalf("decoder performed %d reads after cancellation, want 1", reader.reads)
	}
}

type cancelAfterFirstRead struct {
	cancel  context.CancelFunc
	maxRead int
	reads   int
}

func (r *cancelAfterFirstRead) Read(p []byte) (int, error) {
	r.reads++
	r.maxRead = max(r.maxRead, len(p))
	for i := range p {
		p[i] = 0
	}
	r.cancel()
	return len(p), nil
}

func TestPixelFingerprintPreflightCancellationAndNil(t *testing.T) {
	if _, err := PixelFingerprintContext(context.Background(), nil); !errors.Is(err, ErrNoImageData) {
		t.Fatalf("nil image error = %v, want ErrNoImageData", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	img := image.NewGray(image.Rect(0, 0, 1, 1))
	if _, err := PixelFingerprintContext(ctx, img); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled fingerprint error = %v, want context.Canceled", err)
	}
	if _, err := NewImageProvider(true).PixelFingerprintReader(ctx, bytes.NewReader(pngBytes(t, img))); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled reader error = %v, want context.Canceled", err)
	}
	reader := &contextReader{ctx: ctx, r: bytes.NewReader([]byte("x"))}
	if _, err := reader.Read(make([]byte, 1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("context reader error = %v, want context.Canceled", err)
	}
}

func TestPixelFingerprintReaderCancellationClosesBlockedSource(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader := newBlockingReadCloser()
	errCh := make(chan error, 1)
	go func() {
		_, err := NewImageProvider(true).PixelFingerprintReader(ctx, reader)
		errCh <- err
	}()
	select {
	case <-reader.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("reader did not enter blocked Read")
	}
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked reader cancellation error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation did not close and unblock Reader")
	}
}

type blockingReadCloser struct {
	entered chan struct{}
	closed  chan struct{}
	once    sync.Once
}

func newBlockingReadCloser() *blockingReadCloser {
	return &blockingReadCloser{
		entered: make(chan struct{}),
		closed:  make(chan struct{}),
	}
}

func (r *blockingReadCloser) Read([]byte) (int, error) {
	r.once.Do(func() { close(r.entered) })
	<-r.closed
	return 0, errors.New("reader closed")
}

func (r *blockingReadCloser) Close() error {
	select {
	case <-r.closed:
	default:
		close(r.closed)
	}
	return nil
}

type cancelOnFirstPixel struct {
	image.Image
	cancel context.CancelFunc
	calls  int
}

func (i *cancelOnFirstPixel) At(x, y int) color.Color {
	i.calls++
	if i.calls == 1 {
		i.cancel()
	}
	return i.Image.At(x, y)
}
