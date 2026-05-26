# Vision Library Usage

## API

1. `bs_yzx_init(bool isDebug)` initializes the pipeline from `config/params.xml` and `cnn.ini`.
2. `bs_yzx_object_detection_lanxin(zzb::Box boxArr[])` runs detection and writes results.

`taskId` is no longer part of the public API. Each run creates a timestamp directory:

```text
res/<timestamp>/
  rgb_orig.jpg
  cloud_orig.pcd
  vis_on_orig.jpg
  boxes.json
```

In file mode (`RunMode=0`), the program reads the latest directory under `res/` that contains both `rgb_orig.jpg` and `cloud_orig.pcd`, then writes new results to `res/<timestamp>/`.

In camera mode (`RunMode=1`), the program captures data from the Lanxin camera and writes raw data plus results to `res/<timestamp>/`.

## Return Values

- `< 0`: error code
- `>= 0`: number of detected boxes written to `boxArr`

Common error codes:

- `-10`: pipeline not initialized
- `-11`: camera not initialized or not open
- `-12`: null box array
- `-21`: no valid input directory in file mode
- `-22`: failed to read or capture RGB image
- `-23`: failed to read or capture point cloud
- `-25`: failed to open calibration config
