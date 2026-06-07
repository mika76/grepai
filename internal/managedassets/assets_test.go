package managedassets

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func setTestHomeDir(t *testing.T, dir string) func() {
	t.Helper()
	if runtime.GOOS == "windows" {
		original := os.Getenv("USERPROFILE")
		os.Setenv("USERPROFILE", dir)
		return func() { os.Setenv("USERPROFILE", original) }
	}
	original := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	return func() { os.Setenv("HOME", original) }
}

func TestManagedPaths(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	binDir, err := GetManagedBinDir()
	if err != nil {
		t.Fatalf("GetManagedBinDir failed: %v", err)
	}
	modelDir, err := GetManagedModelsDir()
	if err != nil {
		t.Fatalf("GetManagedModelsDir failed: %v", err)
	}
	if filepath.Base(binDir) != "bin" {
		t.Fatalf("expected bin dir, got %s", binDir)
	}
	if filepath.Base(modelDir) != "models" {
		t.Fatalf("expected models dir, got %s", modelDir)
	}
}

func TestManagedRuntimeBinaryPathIsVersioned(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	def := RuntimeDefinition{
		Version:  "btest",
		Platform: "darwin",
		Arch:     "arm64",
		Binary:   "llama-server",
	}
	path, err := ManagedRuntimeBinaryPath(def)
	if err != nil {
		t.Fatalf("ManagedRuntimeBinaryPath failed: %v", err)
	}
	want := filepath.Join(tmpDir, ".grepai", "runtimes", "llamacpp", "btest", "darwin-arm64", "llama-server")
	if path != want {
		t.Fatalf("runtime path = %q, want %q", path, want)
	}
}

func TestSaveAndLoadInstalledModels(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	models := []InstalledModel{{
		ID:         DefaultModelID,
		FileName:   "test.gguf",
		Path:       filepath.Join(tmpDir, "test.gguf"),
		SourceURL:  "https://example.com/test.gguf",
		Dimensions: 768,
	}}
	if err := SaveInstalledModels(models); err != nil {
		t.Fatalf("SaveInstalledModels failed: %v", err)
	}
	loaded, err := LoadInstalledModels()
	if err != nil {
		t.Fatalf("LoadInstalledModels failed: %v", err)
	}
	if len(loaded) != 1 || loaded[0].ID != DefaultModelID {
		t.Fatalf("unexpected loaded models: %+v", loaded)
	}
}

func TestResolveModelPathRequiresExistingRegularFile(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	missing := filepath.Join(tmpDir, "missing.gguf")
	if err := SaveInstalledModels([]InstalledModel{{
		ID:         DefaultModelID,
		FileName:   "missing.gguf",
		Path:       missing,
		Dimensions: 384,
	}}); err != nil {
		t.Fatalf("SaveInstalledModels failed: %v", err)
	}

	if _, _, err := ResolveModelPath(DefaultModelID, ""); err == nil {
		t.Fatal("expected missing installed model file to fail")
	}

	modelPath := filepath.Join(tmpDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o600); err != nil {
		t.Fatalf("failed to create model file: %v", err)
	}
	if got, dims, err := ResolveModelPath("", modelPath); err != nil {
		t.Fatalf("ResolveModelPath override failed: %v", err)
	} else if got != modelPath || dims != defaultEmbeddingDimSize {
		t.Fatalf("override path/dims = %q/%d", got, dims)
	}
}

func TestLookupCurrentRuntime(t *testing.T) {
	if _, err := LookupCurrentRuntime(); err != nil {
		t.Fatalf("LookupCurrentRuntime failed for %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
}

func TestExtractZipRejectsPathTraversal(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "bad.zip")

	f, err := os.Create(archivePath)
	if err != nil {
		t.Fatalf("failed to create zip: %v", err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("../escape")
	if err != nil {
		t.Fatalf("failed to create zip entry: %v", err)
	}
	if _, err := w.Write([]byte("bad")); err != nil {
		t.Fatalf("failed to write zip entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("failed to close zip: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("failed to close file: %v", err)
	}

	if err := extractZip(archivePath, filepath.Join(tmpDir, "out")); err == nil {
		t.Fatal("expected path traversal zip to fail")
	}
}

func TestExtractTarGzRejectsPathTraversal(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "bad.tar.gz")

	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)
	if err := tw.WriteHeader(&tar.Header{
		Name:    "../escape",
		Mode:    0o600,
		Size:    int64(len("bad")),
		ModTime: time.Now(),
	}); err != nil {
		t.Fatalf("failed to write tar header: %v", err)
	}
	if _, err := tw.Write([]byte("bad")); err != nil {
		t.Fatalf("failed to write tar content: %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("failed to close tar: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("failed to close gzip: %v", err)
	}
	if err := os.WriteFile(archivePath, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("failed to write archive: %v", err)
	}

	if err := extractTarGz(archivePath, filepath.Join(tmpDir, "out")); err == nil {
		t.Fatal("expected path traversal tar.gz to fail")
	}
}

func TestExtractTarGzPreservesSafeSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires additional privileges on Windows")
	}

	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "runtime.tar.gz")

	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)
	content := []byte("library")
	if err := tw.WriteHeader(&tar.Header{
		Name:    "runtime/libexample.1.dylib",
		Mode:    0o755,
		Size:    int64(len(content)),
		ModTime: time.Now(),
	}); err != nil {
		t.Fatalf("failed to write tar header: %v", err)
	}
	if _, err := tw.Write(content); err != nil {
		t.Fatalf("failed to write tar content: %v", err)
	}
	if err := tw.WriteHeader(&tar.Header{
		Name:     "runtime/libexample.dylib",
		Typeflag: tar.TypeSymlink,
		Linkname: "libexample.1.dylib",
		ModTime:  time.Now(),
	}); err != nil {
		t.Fatalf("failed to write tar symlink: %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("failed to close tar: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("failed to close gzip: %v", err)
	}
	if err := os.WriteFile(archivePath, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("failed to write archive: %v", err)
	}

	outDir := filepath.Join(tmpDir, "out")
	if err := extractTarGz(archivePath, outDir); err != nil {
		t.Fatalf("extractTarGz failed: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(outDir, "runtime", "libexample.dylib"))
	if err != nil {
		t.Fatalf("failed to read symlink: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("symlink content = %q, want %q", got, content)
	}
}

func TestLookupRuntime_KnownCrossPlatformTargets(t *testing.T) {
	targets := [][2]string{
		{"darwin", "arm64"},
		{"darwin", "amd64"},
		{"linux", "amd64"},
		{"windows", "amd64"},
	}

	for _, target := range targets {
		def, err := LookupRuntime(target[0], target[1])
		if err != nil {
			t.Fatalf("LookupRuntime(%s, %s) failed: %v", target[0], target[1], err)
		}
		if def.URL == "" || def.Binary == "" {
			t.Fatalf("incomplete runtime definition for %s/%s: %+v", target[0], target[1], def)
		}
		if def.Archive == "" || def.SHA256 == "" {
			t.Fatalf("runtime definition for %s/%s must pin archive type and checksum: %+v", target[0], target[1], def)
		}
	}
}

func TestEnsureRuntimeIgnoresLegacyGlobalBinary(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	legacyDir := filepath.Join(tmpDir, ".grepai", "bin")
	if err := os.MkdirAll(legacyDir, 0o755); err != nil {
		t.Fatalf("failed to create legacy bin dir: %v", err)
	}
	legacyPath := filepath.Join(legacyDir, "llama-server")
	if runtime.GOOS == "windows" {
		legacyPath += ".exe"
	}
	if err := os.WriteFile(legacyPath, []byte("legacy"), 0o755); err != nil {
		t.Fatalf("failed to create legacy binary: %v", err)
	}

	binary := "llama-server"
	if runtime.GOOS == "windows" {
		binary = "llama-server.exe"
	}
	archive := makeRuntimeZip(t, binary, []byte("current"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(archive)
	}))
	defer server.Close()

	key := runtime.GOOS + "/" + runtime.GOARCH
	original, hadOriginal := runtimeDefinitions[key]
	runtimeDefinitions[key] = RuntimeDefinition{
		Version:  "btest",
		Platform: runtime.GOOS,
		Arch:     runtime.GOARCH,
		URL:      server.URL + "/runtime.zip",
		Archive:  "zip",
		Binary:   binary,
	}
	defer func() {
		if hadOriginal {
			runtimeDefinitions[key] = original
		} else {
			delete(runtimeDefinitions, key)
		}
	}()

	path, def, err := EnsureRuntime(context.Background(), nil)
	if err != nil {
		t.Fatalf("EnsureRuntime failed: %v", err)
	}
	if def.Version != "btest" {
		t.Fatalf("runtime version = %q, want btest", def.Version)
	}
	if path == legacyPath {
		t.Fatalf("EnsureRuntime reused legacy binary path %s", path)
	}
	wantPath := filepath.Join(tmpDir, ".grepai", "runtimes", "llamacpp", "btest", runtime.GOOS+"-"+runtime.GOARCH, binary)
	if path != wantPath {
		t.Fatalf("runtime path = %q, want %q", path, wantPath)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read installed runtime: %v", err)
	}
	if string(data) != "current" {
		t.Fatalf("installed runtime = %q, want current", data)
	}
}

func TestEnsureRuntimeRepairsIncompleteVersionedInstall(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	binary := "llama-server"
	if runtime.GOOS == "windows" {
		binary = "llama-server.exe"
	}

	def := RuntimeDefinition{
		Version:  "btest",
		Platform: runtime.GOOS,
		Arch:     runtime.GOARCH,
		Archive:  "zip",
		Binary:   binary,
	}
	binPath, err := ManagedRuntimeBinaryPath(def)
	if err != nil {
		t.Fatalf("ManagedRuntimeBinaryPath failed: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(binPath), 0o755); err != nil {
		t.Fatalf("failed to create incomplete runtime dir: %v", err)
	}
	if err := os.WriteFile(binPath, []byte("partial"), 0o755); err != nil {
		t.Fatalf("failed to create incomplete runtime binary: %v", err)
	}

	var downloads int
	archive := makeRuntimeZip(t, binary, []byte("complete"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		downloads++
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(archive)
	}))
	defer server.Close()
	def.URL = server.URL + "/runtime.zip"

	key := runtime.GOOS + "/" + runtime.GOARCH
	original, hadOriginal := runtimeDefinitions[key]
	runtimeDefinitions[key] = def
	defer func() {
		if hadOriginal {
			runtimeDefinitions[key] = original
		} else {
			delete(runtimeDefinitions, key)
		}
	}()

	path, _, err := EnsureRuntime(context.Background(), nil)
	if err != nil {
		t.Fatalf("EnsureRuntime failed: %v", err)
	}
	if downloads != 1 {
		t.Fatalf("downloads = %d, want 1", downloads)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read installed runtime: %v", err)
	}
	if string(data) != "complete" {
		t.Fatalf("installed runtime = %q, want complete", data)
	}
	marker := filepath.Join(filepath.Dir(path), runtimeInstallFileName)
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("runtime install marker missing: %v", err)
	}
}

func TestRuntimeStateRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	cleanup := setTestHomeDir(t, tmpDir)
	defer cleanup()

	state := RuntimeState{
		Version:  DefaultRuntimeVersion,
		Platform: "darwin",
		Arch:     "arm64",
		Binary:   "/tmp/llama-server",
		Endpoint: DefaultSidecarEndpoint(),
		PID:      12345,
	}
	if err := SaveRuntimeState(state); err != nil {
		t.Fatalf("SaveRuntimeState failed: %v", err)
	}
	loaded, err := LoadRuntimeState()
	if err != nil {
		t.Fatalf("LoadRuntimeState failed: %v", err)
	}
	if loaded == nil || loaded.PID != state.PID || loaded.Endpoint != state.Endpoint {
		t.Fatalf("unexpected runtime state: %+v", loaded)
	}
	if err := ClearRuntimeState(); err != nil {
		t.Fatalf("ClearRuntimeState failed: %v", err)
	}
	loaded, err = LoadRuntimeState()
	if err != nil {
		t.Fatalf("LoadRuntimeState after clear failed: %v", err)
	}
	if loaded != nil {
		t.Fatalf("expected nil runtime state after clear, got %+v", loaded)
	}
}

func makeRuntimeZip(t *testing.T, binary string, content []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create(binary)
	if err != nil {
		t.Fatalf("failed to create zip entry: %v", err)
	}
	if _, err := w.Write(content); err != nil {
		t.Fatalf("failed to write zip entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("failed to close zip: %v", err)
	}
	return buf.Bytes()
}
