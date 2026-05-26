# Vision Library Usage

## API

1. `bs_yzx_init(bool isDebug)` initializes the camera + GPU pipeline from `config/params.xml` and `cnn.ini`.
2. `bs_yzx_object_detection_lanxin(zzb::Box boxArr[], const char *cameraIp)` captures from the Lanxin camera at `cameraIp`, runs GPU detection, and writes results.

`taskId` is no longer part of the public API. Each run creates a timestamp directory:

```text
res/<timestamp>/
  rgb_orig.jpg
  cloud_orig.pcd
  vis_on_orig.jpg
```

The local file mode and CPU mode have been removed. `cameraIp` is required. The program opens the Lanxin camera by that IP address, captures data, and writes raw data plus results to `res/<timestamp>/`. The example executables accept the camera IP as the first command-line argument.

## Return Values

- `< 0`: error code
- `>= 0`: number of detected boxes written to `boxArr`

Common error codes:

- `-10`: pipeline not initialized
- `-11`: camera not initialized or not open
- `-12`: null box array
- `-13`: empty camera IP
- `-22`: failed to read or capture RGB image
- `-23`: failed to read or capture point cloud
- `-25`: failed to open calibration config
- `-31`: failed to enable CUDA provider
