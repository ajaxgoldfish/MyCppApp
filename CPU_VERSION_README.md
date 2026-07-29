# Vision Library Usage

## API

1. `bs_yzx_init(bool isDebug)` initializes the GPU pipeline, discovers every Lanxin camera, and opens all camera streams.
2. `bs_yzx_object_detection_lanxin(zzb::Box boxArr[], const char *cameraIp, const char *intrinsicExtrinsicPath)` reads `intrinsics` and `extrinsics` from `intrinsicExtrinsicPath`, captures from the Lanxin camera at `cameraIp`, runs GPU detection, and writes results.

`taskId` is no longer part of the public API. Each run creates a timestamp directory:

```text
res/<timestamp>/
  rgb_orig.jpg
  vis_on_orig.jpg
```

The local file mode and CPU mode have been removed. During initialization, every discovered Lanxin camera is opened and kept streaming. Detection only selects an already-open camera by `cameraIp`; switching IP does not stop, close, or reopen any camera. All camera connections close when the DLL/process exits. `cameraIp` and `intrinsicExtrinsicPath` are required. Legacy calibration node names `intrinsicRGB` and `extrinsicRGB` are still accepted.

`cnn.ini` can set `capture_retry_timeout_ms` to control how long each camera capture call retries before returning `-22` or `-23`.

## Return Values

- `< 0`: error code
- `>= 0`: number of detected boxes written to `boxArr`

Common error codes:

- `-10`: pipeline not initialized
- `-11`: camera discovery/initialization failed, or requested camera IP is not ready
- `-12`: null box array
- `-13`: empty camera IP
- `-14`: empty intrinsic/extrinsic path
- `-22`: failed to read or capture RGB image
- `-23`: failed to read or capture aligned depth image
- `-25`: failed to open calibration config
- `-26`: invalid intrinsic config
- `-28`: missing extrinsic config
- `-29`: invalid extrinsic config
- `-31`: failed to enable CUDA provider
