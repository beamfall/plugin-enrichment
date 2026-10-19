package enrichment

import (
	"bytes"
	"compress/zlib"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"hash/fnv"
	"image"
	"io"
)

const (
	// maxImageBytes is the hard encoded-input ceiling.
	maxImageBytes = 16 << 20
	// maxImagePixels is the hard decoded-pixel ceiling.
	maxImagePixels = 16 << 20
	// maxImageDimension rejects pathological aspect ratios independently of
	// the aggregate pixel ceiling.
	maxImageDimension = 8192
	// pixelCancellationInterval is the maximum number of pixels traversed
	// between context checks.
	pixelCancellationInterval = 4096
	// MaxFingerprintBatchItems bounds output allocation and aggregate work for
	// one public call. Inputs beyond the ceiling fail before allocation or I/O.
	MaxFingerprintBatchItems = 1024

	gridSide = 16
)

// ImageKindString is the storage kind of the bundled pixel fingerprint. It is
// deliberately segregated from every semantic embedding kind.
const ImageKindString = "pixel-fingerprint-v1"

var ErrNoImageData = errors.New("image input is empty or undecodable")

// ErrFingerprintBatchTooLarge reports a batch rejected before allocation or
// input reads because it exceeds MaxFingerprintBatchItems.
var ErrFingerprintBatchTooLarge = errors.New("pixel fingerprint batch exceeds item limit")

// ImageProvider provides explicitly non-semantic pixel fingerprints. It is
// never available through the semantic provider seam.
type ImageProvider struct {
	enabled bool
}

func NewImageProvider(enabled bool) *ImageProvider {
	return &ImageProvider{enabled: enabled}
}

// Available reports semantic availability. A pixel hash is never a semantic
// image model.
func (p *ImageProvider) Available() bool { return false }

func (p *ImageProvider) FingerprintsAvailable() bool { return p != nil && p.enabled }

// EmbedImageReader is the legacy semantic seam. It fails closed so a caller
// cannot persist a pixel hash as a semantic vector.
func (p *ImageProvider) EmbedImageReader(ctx context.Context, r io.Reader) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, ErrSemanticProviderUnavailable
}

// PixelFingerprintReader decodes one bounded, non-interlaced PNG and computes
// its explicit non-semantic fingerprint. The streaming decoder checks ctx
// before consuming each group of at most 4,096 decoded pixels.
//
// An io.Reader has no cancellation operation. Potentially blocking sources
// should implement io.ReadCloser; cancellation then closes the source to
// interrupt a blocked Read. For a Reader without Close, cancellation is
// observed immediately before and after each Read but cannot preempt the Read
// itself.
func (p *ImageProvider) PixelFingerprintReader(ctx context.Context, r io.Reader) ([]float32, error) {
	if r == nil {
		return nil, errors.Join(ErrNoImageData, errors.New("nil image reader"))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	data, err := readBoundedImage(ctx, r)
	if err != nil {
		return nil, err
	}
	vec, err := pixelFingerprintPNG(ctx, data)
	if err != nil {
		return nil, errors.Join(ErrNoImageData, err)
	}
	return vec, nil
}

// readBoundedImage retains at most maxImageBytes and checks cancellation around
// every reader interaction. Closing a ReadCloser on cancellation is the only
// general way io.Reader permits a blocked read to be interrupted.
func readBoundedImage(ctx context.Context, r io.Reader) ([]byte, error) {
	var stopClose func() bool
	if closer, ok := r.(io.ReadCloser); ok {
		stopClose = context.AfterFunc(ctx, func() {
			_ = closer.Close()
		})
		defer stopClose()
	}
	lr := &io.LimitedReader{R: &contextReader{ctx: ctx, r: r}, N: maxImageBytes + 1}
	data, err := io.ReadAll(lr)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, ErrNoImageData
	}
	if len(data) > maxImageBytes {
		return nil, errors.Join(ErrNoImageData, fmt.Errorf("image input exceeds %d byte limit", maxImageBytes))
	}
	return data, nil
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r *contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	n, err := r.r.Read(p)
	if contextErr := r.ctx.Err(); contextErr != nil {
		return 0, contextErr
	}
	return n, err
}

func checkDecodedDims(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("image dimensions %dx%d contain no pixels", width, height)
	}
	if width > maxImageDimension || height > maxImageDimension {
		return fmt.Errorf("image dimensions %dx%d exceed %d pixel dimension limit", width, height, maxImageDimension)
	}
	if int64(width)*int64(height) > maxImagePixels {
		return fmt.Errorf("image dimensions %dx%d exceed %d pixel limit", width, height, maxImagePixels)
	}
	return nil
}

type BatchItem struct {
	Key    string
	Reader io.Reader
}

type BatchResult struct {
	Key string
	Vec []float32
	Err error
}

// EmbedImageBatch is the legacy semantic seam. It fails closed without reading
// any item.
func (p *ImageProvider) EmbedImageBatch(ctx context.Context, items []BatchItem) ([]BatchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, ErrSemanticProviderUnavailable
}

// PixelFingerprintBatch fingerprints bounded inputs sequentially. Per-item
// decode failures remain isolated; cancellation stops the batch immediately.
func (p *ImageProvider) PixelFingerprintBatch(ctx context.Context, items []BatchItem) ([]BatchResult, error) {
	if len(items) > MaxFingerprintBatchItems {
		return nil, errors.Join(
			ErrFingerprintBatchTooLarge,
			fmt.Errorf("got %d items, limit is %d", len(items), MaxFingerprintBatchItems),
		)
	}
	results := make([]BatchResult, 0, len(items))
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return results, err
		}
		vec, err := p.PixelFingerprintReader(ctx, item.Reader)
		results = append(results, BatchResult{Key: item.Key, Vec: vec, Err: err})
	}
	return results, nil
}

var pngSignature = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

type pngFingerprintSpec struct {
	width, height int
	bitDepth      int
	colorType     byte
	channels      int
	palette       []byte
	transparency  []byte
	data          []byte
	idatStart     int
	idatEnd       int
}

// pixelFingerprintPNG decodes scanlines directly into the 16x16 accumulator.
// It never allocates a full decoded image, and each compressed read is capped
// to the bytes representing at most pixelCancellationInterval output pixels.
func pixelFingerprintPNG(ctx context.Context, data []byte) ([]float32, error) {
	spec, err := parseFingerprintPNG(ctx, data)
	if err != nil {
		return nil, err
	}

	idat := newPNGIDATReader(ctx, spec.data, spec.idatStart, spec.idatEnd)
	zr, err := zlib.NewReader(idat)
	if err != nil {
		return nil, fmt.Errorf("open PNG image data: %w", err)
	}
	defer func() {
		_ = zr.Close()
	}()

	bitsPerPixel := spec.channels * spec.bitDepth
	rowBytes := (spec.width*bitsPerPixel + 7) / 8
	filterBytes := (bitsPerPixel + 7) / 8
	current := make([]byte, rowBytes)
	previous := make([]byte, rowBytes)
	var red, green, blue, count [gridSide * gridSide]uint64

	for y := 0; y < spec.height; y++ {
		filter, err := readPNGFilterByte(ctx, zr)
		if err != nil {
			return nil, fmt.Errorf("read PNG row %d filter: %w", y, err)
		}
		if err := readPNGPixelBytes(ctx, zr, current, bitsPerPixel); err != nil {
			return nil, fmt.Errorf("read PNG row %d: %w", y, err)
		}
		if err := unfilterPNGRow(ctx, current, previous, filterBytes, filter); err != nil {
			return nil, fmt.Errorf("unfilter PNG row %d: %w", y, err)
		}

		cy := ((y+1)*gridSide - 1) / spec.height
		for x := 0; x < spec.width; x++ {
			if x%pixelCancellationInterval == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			r, g, b, err := spec.pixel(current, x)
			if err != nil {
				return nil, fmt.Errorf("decode PNG pixel (%d,%d): %w", x, y, err)
			}
			cx := ((x+1)*gridSide - 1) / spec.width
			cell := cy*gridSide + cx
			red[cell] += uint64(r)
			green[cell] += uint64(g)
			blue[cell] += uint64(b)
			count[cell]++
		}
		current, previous = previous, current
	}
	var trailing [1]byte
	if n, err := zr.Read(trailing[:]); n != 0 || (err != nil && !errors.Is(err, io.EOF)) {
		if err == nil {
			err = errors.New("unexpected decompressed data after final row")
		}
		return nil, fmt.Errorf("finish PNG image data: %w", err)
	}
	hasTrailing, err := idat.hasRemainingData()
	if err != nil {
		return nil, fmt.Errorf("finish PNG image data: %w", err)
	}
	if hasTrailing {
		return nil, errors.New("unused compressed data after PNG zlib stream")
	}

	vec := make([]float32, Dims)
	h := fnv.New32a()
	for cell := range count {
		if count[cell] == 0 {
			continue
		}
		r := byte(red[cell] / count[cell] >> 8)
		g := byte(green[cell] / count[cell] >> 8)
		b := byte(blue[cell] / count[cell] >> 8)
		lum := byte((uint32(r)*299 + uint32(g)*587 + uint32(b)*114) / 1000)
		cx, cy := cell%gridSide, cell/gridSide
		h.Reset()
		_, _ = h.Write([]byte{byte(cx), byte(cy), lum, r, g, b})
		vec[h.Sum32()%uint32(Dims)]++
	}
	l2Normalize(vec)
	return vec, nil
}

// pngIDATReader concatenates validated, consecutive IDAT payloads without
// retaining per-chunk slices, interfaces, or heap readers. Its state remains
// constant even when a valid encoded-limit PNG contains millions of empty
// IDAT chunks.
type pngIDATReader struct {
	budget  contextWorkBudget
	data    []byte
	payload []byte
	offset  int
	end     int
}

func newPNGIDATReader(ctx context.Context, data []byte, start, end int) *pngIDATReader {
	return &pngIDATReader{
		budget: contextWorkBudget{ctx: ctx},
		data:   data,
		offset: start,
		end:    end,
	}
}

func (r *pngIDATReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for len(r.payload) == 0 {
		if r.offset == r.end {
			return 0, io.EOF
		}
		if err := r.budget.before(12); err != nil {
			return 0, err
		}
		length := int(binary.BigEndian.Uint32(r.data[r.offset : r.offset+4]))
		payloadEnd := r.offset + 8 + length
		r.payload = r.data[r.offset+8 : payloadEnd]
		r.offset = payloadEnd + 4
	}
	n, err := r.budget.next(min(len(p), len(r.payload)))
	if err != nil {
		return 0, err
	}
	n = copy(p[:n], r.payload[:n])
	r.payload = r.payload[n:]
	return n, nil
}

func (r *pngIDATReader) ReadByte() (byte, error) {
	var data [1]byte
	if _, err := r.Read(data[:]); err != nil {
		return 0, err
	}
	return data[0], nil
}

func (r *pngIDATReader) hasRemainingData() (bool, error) {
	_, err := r.ReadByte()
	if errors.Is(err, io.EOF) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

type contextWorkBudget struct {
	ctx       context.Context
	remaining int
}

func (b *contextWorkBudget) before(work int) error {
	if b.remaining >= work {
		b.remaining -= work
		return nil
	}
	if err := b.ctx.Err(); err != nil {
		return err
	}
	b.remaining = pixelCancellationInterval - work
	return nil
}

func (b *contextWorkBudget) next(work int) (int, error) {
	if b.remaining == 0 {
		if err := b.ctx.Err(); err != nil {
			return 0, err
		}
		b.remaining = pixelCancellationInterval
	}
	work = min(work, b.remaining)
	b.remaining -= work
	return work, nil
}

func parseFingerprintPNG(ctx context.Context, data []byte) (pngFingerprintSpec, error) {
	spec := pngFingerprintSpec{data: data}
	if len(data) < len(pngSignature) || !bytes.Equal(data[:len(pngSignature)], pngSignature) {
		return spec, errors.New("only PNG fingerprint input is supported")
	}
	offset := len(pngSignature)
	seenIHDR := false
	seenPLTE := false
	seenTRNS := false
	seenIDAT := false
	endedIDAT := false
	seenIEND := false
	budget := contextWorkBudget{ctx: ctx}
	for offset < len(data) {
		if err := budget.before(12); err != nil {
			return spec, err
		}
		if len(data)-offset < 12 {
			return spec, errors.New("truncated PNG chunk header")
		}
		length := int64(binary.BigEndian.Uint32(data[offset : offset+4]))
		if length > int64(len(data)-offset-12) {
			return spec, errors.New("truncated PNG chunk payload")
		}
		chunkEnd := offset + 8 + int(length)
		crcEnd := chunkEnd + 4
		kind := string(data[offset+4 : offset+8])
		payload := data[offset+8 : chunkEnd]
		wantCRC := binary.BigEndian.Uint32(data[chunkEnd:crcEnd])
		gotCRC, err := pngChunkCRC(&budget, data[offset+4:chunkEnd])
		if err != nil {
			return spec, err
		}
		if gotCRC != wantCRC {
			return spec, fmt.Errorf("%s chunk CRC mismatch", kind)
		}

		switch kind {
		case "IHDR":
			if seenIHDR || offset != len(pngSignature) || len(payload) != 13 {
				return spec, errors.New("invalid PNG IHDR")
			}
			width := binary.BigEndian.Uint32(payload[0:4])
			height := binary.BigEndian.Uint32(payload[4:8])
			spec.width, spec.height = int(width), int(height)
			if err := checkDecodedDims(spec.width, spec.height); err != nil {
				return spec, err
			}
			spec.bitDepth = int(payload[8])
			spec.colorType = payload[9]
			if payload[10] != 0 || payload[11] != 0 || payload[12] != 0 {
				return spec, errors.New("unsupported PNG compression, filter, or interlace method")
			}
			channels, ok := pngChannels(spec.colorType, spec.bitDepth)
			if !ok {
				return spec, fmt.Errorf("unsupported PNG color type %d at bit depth %d", spec.colorType, spec.bitDepth)
			}
			spec.channels = channels
			seenIHDR = true
		case "PLTE":
			if !validPNGPalette(spec, payload, seenIHDR, seenPLTE, seenTRNS, seenIDAT) {
				return spec, errors.New("invalid PNG palette")
			}
			spec.palette = payload
			seenPLTE = true
		case "tRNS":
			if !seenIHDR || seenTRNS || seenIDAT || !validPNGTransparency(spec, payload, seenPLTE) {
				return spec, errors.New("invalid PNG transparency")
			}
			spec.transparency = payload
			seenTRNS = true
		case "IDAT":
			if !seenIHDR {
				return spec, errors.New("PNG image data precedes IHDR")
			}
			if endedIDAT {
				return spec, errors.New("PNG image data chunks are not consecutive")
			}
			if !seenIDAT {
				spec.idatStart = offset
			}
			seenIDAT = true
			spec.idatEnd = crcEnd
		case "IEND":
			if len(payload) != 0 {
				return spec, errors.New("invalid PNG IEND")
			}
			seenIEND = true
		default:
			if kind[0]&0x20 == 0 {
				return spec, fmt.Errorf("unsupported critical PNG chunk %q", kind)
			}
			if seenIDAT {
				endedIDAT = true
			}
		}
		offset = crcEnd
		if seenIEND {
			if offset != len(data) {
				return spec, errors.New("data follows PNG IEND")
			}
			break
		}
	}
	if !seenIHDR || !seenIEND || !seenIDAT {
		return spec, errors.New("incomplete PNG")
	}
	if spec.colorType == 3 && len(spec.palette) == 0 {
		return spec, errors.New("indexed PNG has no palette")
	}
	if err := ctx.Err(); err != nil {
		return spec, err
	}
	return spec, nil
}

func pngChunkCRC(budget *contextWorkBudget, data []byte) (uint32, error) {
	var sum uint32
	for len(data) != 0 {
		n, err := budget.next(len(data))
		if err != nil {
			return 0, err
		}
		sum = crc32.Update(sum, crc32.IEEETable, data[:n])
		data = data[n:]
	}
	return sum, nil
}

func validPNGPalette(spec pngFingerprintSpec, payload []byte, seenIHDR, seenPLTE, seenTRNS, seenIDAT bool) bool {
	if !seenIHDR || seenPLTE || seenTRNS || seenIDAT {
		return false
	}
	if spec.colorType == 0 || spec.colorType == 4 {
		return false
	}
	if len(payload) == 0 || len(payload)%3 != 0 || len(payload) > 256*3 {
		return false
	}
	entries := len(payload) / 3
	return spec.colorType != 3 || entries <= 1<<spec.bitDepth
}

func validPNGTransparency(spec pngFingerprintSpec, payload []byte, seenPLTE bool) bool {
	switch spec.colorType {
	case 0:
		if len(payload) != 2 {
			return false
		}
		return spec.bitDepth == 16 || int(binary.BigEndian.Uint16(payload)) < 1<<spec.bitDepth
	case 2:
		if len(payload) != 6 {
			return false
		}
		if spec.bitDepth == 16 {
			return true
		}
		for offset := 0; offset < len(payload); offset += 2 {
			if binary.BigEndian.Uint16(payload[offset:offset+2]) >= 1<<spec.bitDepth {
				return false
			}
		}
		return true
	case 3:
		return seenPLTE && len(payload) != 0 && len(payload) <= len(spec.palette)/3
	default:
		return false
	}
}

func pngChannels(colorType byte, bitDepth int) (int, bool) {
	switch colorType {
	case 0:
		return 1, bitDepth == 1 || bitDepth == 2 || bitDepth == 4 || bitDepth == 8 || bitDepth == 16
	case 2:
		return 3, bitDepth == 8 || bitDepth == 16
	case 3:
		return 1, bitDepth == 1 || bitDepth == 2 || bitDepth == 4 || bitDepth == 8
	case 4:
		return 2, bitDepth == 8 || bitDepth == 16
	case 6:
		return 4, bitDepth == 8 || bitDepth == 16
	default:
		return 0, false
	}
}

func readPNGFilterByte(ctx context.Context, r io.Reader) (byte, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	var filter [1]byte
	if _, err := io.ReadFull(r, filter[:]); err != nil {
		return 0, err
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return filter[0], nil
}

func readPNGPixelBytes(ctx context.Context, r io.Reader, row []byte, bitsPerPixel int) error {
	maxChunk := (pixelCancellationInterval*bitsPerPixel + 7) / 8
	if maxChunk < 1 {
		maxChunk = 1
	}
	for offset := 0; offset < len(row); {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(offset+maxChunk, len(row))
		if _, err := io.ReadFull(r, row[offset:end]); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		offset = end
	}
	return nil
}

func unfilterPNGRow(ctx context.Context, row, previous []byte, bytesPerPixel int, filter byte) error {
	const maxBytesPerCancellationCheck = pixelCancellationInterval / 8
	for i := range row {
		// One byte represents at most eight decoded pixels (1-bit grayscale),
		// so 512 bytes is the conservative checkpoint for every PNG mode.
		if i%maxBytesPerCancellationCheck == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		var left, up, upperLeft byte
		if i >= bytesPerPixel {
			left = row[i-bytesPerPixel]
			upperLeft = previous[i-bytesPerPixel]
		}
		up = previous[i]
		switch filter {
		case 0:
		case 1:
			row[i] += left
		case 2:
			row[i] += up
		case 3:
			row[i] += byte((uint16(left) + uint16(up)) / 2)
		case 4:
			row[i] += paeth(left, up, upperLeft)
		default:
			return fmt.Errorf("invalid PNG filter %d", filter)
		}
	}
	return nil
}

func paeth(a, b, c byte) byte {
	p := int(a) + int(b) - int(c)
	pa, pb, pc := absInt(p-int(a)), absInt(p-int(b)), absInt(p-int(c))
	if pa <= pb && pa <= pc {
		return a
	}
	if pb <= pc {
		return b
	}
	return c
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func (s pngFingerprintSpec) pixel(row []byte, x int) (uint32, uint32, uint32, error) {
	rawSample := func(channel int) uint16 {
		sampleIndex := x*s.channels + channel
		switch s.bitDepth {
		case 16:
			offset := sampleIndex * 2
			return binary.BigEndian.Uint16(row[offset : offset+2])
		case 8:
			return uint16(row[sampleIndex])
		default:
			bitOffset := sampleIndex * s.bitDepth
			shift := 8 - s.bitDepth - bitOffset%8
			mask := byte((1 << s.bitDepth) - 1)
			return uint16((row[bitOffset/8] >> shift) & mask)
		}
	}
	sample := func(channel int) uint16 {
		raw := rawSample(channel)
		if s.bitDepth == 16 {
			return raw
		}
		return uint16(uint32(raw) * 0xffff / uint32((1<<s.bitDepth)-1))
	}
	opaque := uint16(0xffff)
	var r, g, b, a uint16
	switch s.colorType {
	case 0:
		r = sample(0)
		g, b, a = r, r, opaque
		if len(s.transparency) == 2 && rawSample(0) == binary.BigEndian.Uint16(s.transparency) {
			a = 0
		}
	case 2:
		r, g, b, a = sample(0), sample(1), sample(2), opaque
		if len(s.transparency) == 6 &&
			rawSample(0) == binary.BigEndian.Uint16(s.transparency[0:2]) &&
			rawSample(1) == binary.BigEndian.Uint16(s.transparency[2:4]) &&
			rawSample(2) == binary.BigEndian.Uint16(s.transparency[4:6]) {
			a = 0
		}
	case 3:
		index := int(rawSample(0))
		if index*3+2 >= len(s.palette) {
			return 0, 0, 0, errors.New("palette index out of range")
		}
		r = uint16(s.palette[index*3]) * 0x101
		g = uint16(s.palette[index*3+1]) * 0x101
		b = uint16(s.palette[index*3+2]) * 0x101
		a = opaque
		if index < len(s.transparency) {
			a = uint16(s.transparency[index]) * 0x101
		}
	case 4:
		r, a = sample(0), sample(1)
		g, b = r, r
	case 6:
		r, g, b, a = sample(0), sample(1), sample(2), sample(3)
	default:
		return 0, 0, 0, errors.New("unsupported PNG color type")
	}
	return uint32(r) * uint32(a) / 0xffff,
		uint32(g) * uint32(a) / 0xffff,
		uint32(b) * uint32(a) / 0xffff,
		nil
}

// PixelFingerprint returns a deterministic pixel fingerprint for callers that
// already own a decoded image. It is not a semantic embedding.
func PixelFingerprint(img image.Image) []float32 {
	vec, _ := PixelFingerprintContext(context.Background(), img)
	return vec
}

// PixelFingerprintContext computes the pixel fingerprint while checking ctx
// before traversal and after every pixelCancellationInterval pixels.
func PixelFingerprintContext(ctx context.Context, img image.Image) ([]float32, error) {
	if img == nil {
		return nil, ErrNoImageData
	}
	bounds := img.Bounds()
	if err := checkDecodedDims(bounds.Dx(), bounds.Dy()); err != nil {
		return nil, errors.Join(ErrNoImageData, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	vec := make([]float32, Dims)
	h := fnv.New32a()
	for cy := 0; cy < gridSide; cy++ {
		for cx := 0; cx < gridSide; cx++ {
			r, g, b, ok, err := cellAverage(ctx, img, bounds, cx, cy)
			if err != nil {
				return nil, err
			}
			if !ok {
				continue
			}
			lum := byte((uint32(r)*299 + uint32(g)*587 + uint32(b)*114) / 1000)
			h.Reset()
			_, _ = h.Write([]byte{byte(cx), byte(cy), lum, r, g, b})
			vec[h.Sum32()%uint32(Dims)]++
		}
	}
	l2Normalize(vec)
	return vec, nil
}

func cellAverage(
	ctx context.Context,
	img image.Image,
	bounds image.Rectangle,
	cx, cy int,
) (r, g, b byte, ok bool, err error) {
	x0 := bounds.Min.X + cx*bounds.Dx()/gridSide
	x1 := bounds.Min.X + (cx+1)*bounds.Dx()/gridSide
	y0 := bounds.Min.Y + cy*bounds.Dy()/gridSide
	y1 := bounds.Min.Y + (cy+1)*bounds.Dy()/gridSide
	var sr, sg, sb, n uint64
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			if n%pixelCancellationInterval == 0 {
				if err := ctx.Err(); err != nil {
					return 0, 0, 0, false, err
				}
			}
			pr, pg, pb, _ := img.At(x, y).RGBA()
			sr += uint64(pr)
			sg += uint64(pg)
			sb += uint64(pb)
			n++
		}
	}
	if n == 0 {
		return 0, 0, 0, false, nil
	}
	return byte(sr / n >> 8), byte(sg / n >> 8), byte(sb / n >> 8), true, nil
}
