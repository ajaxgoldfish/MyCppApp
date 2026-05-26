# Vision Library Usage

## API

1. `bs_yzx_init(bool isDebug)` initializes the camera + GPU pipeline and CNN settings from `cnn.ini`.
2. `bs_yzx_object_detection_lanxin(zzb::Box boxArr[], const char *cameraIp, const char *intrinsicExtrinsicPath)` reads `intrinsicRGB` and `extrinsicRGB` from `intrinsicExtrinsicPath`, captures from the Lanxin camera at `cameraIp`, runs GPU detection, and writes results.

`taskId` is no longer part of the public API. Each run creates a timestamp directory:

```text
res/<timestamp>/
  rgb_orig.jpg
  cloud_orig.pcd
  vis_on_orig.jpg
```

The local file mode and CPU mode have been removed. `cameraIp` and `intrinsicExtrinsicPath` are required. The program opens the Lanxin camera by that IP address, reads calibration from the supplied path, captures data, and writes raw data plus results to `res/<timestamp>/`. The example executables accept the camera IP as the first command-line argument and the calibration path as the second.

## Return Values

- `< 0`: error code
- `>= 0`: number of detected boxes written to `boxArr`

Common error codes:

- `-10`: pipeline not initialized
- `-11`: camera not initialized or not open
- `-12`: null box array
- `-13`: empty camera IP
- `-14`: empty intrinsic/extrinsic path
- `-22`: failed to read or capture RGB image
- `-23`: failed to read or capture point cloud
- `-25`: failed to open calibration config
- `-26`: invalid intrinsic config
- `-28`: missing extrinsic config
- `-29`: invalid extrinsic config
- `-31`: failed to enable CUDA provider
