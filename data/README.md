# CVO Dataset

CVO dataset is a large optical flow dataset, consisting of synthetic video sequences and comprehensive optical flows (including adjacent frame and cross-frame flows).

### Dataset Details

- Video Information:
    - Resolution: 512x512
    - FPS: 30
    - Duration: 0.2s
- Number of Training Videos: 11406
- Number of Testing Videos: 536

### Data Format

We dump all the data in LMDB format, where we can index keys for correspoinding values. For each data sample, we provide five data groups as follows:

- `imgs`: List of video frames, [I0, I1, I2, ..., I6].
- `imgs_blur`: List of blurred video frames, random motion blur is added.
- `fflows`: List of forward long-range optical flows, [F02, F03, ..., F06].
- `bflows`: List of backward long-range optical flows, [F20, F30, ...,F60].
- `delta_fflows`: List of forward local optical flows, [F01, F12, ..., F56].
- `delta_bflows`: List of backward local optical flows, [F10, F21, ..., F65].

Here shows the structure of the training file:

    CVO_train.lmdb
    ├── __samples__: [0,1,2,...,11405]
    ├── 00000_imgs: [...] # 7 RGB frames
    ├── 00000_imgs_blur: [...] # 7 RGB frames (random blurred)
    ├── 00000_fflows: [...] # 5 optical flows (cross-frame)
    ├── 00000_bflows: [...] # 5 optical flows (cross-frame)
    ├── 00000_delta_fflows: [...] # 6 optical flows (adjacent frame)
    ├── 00000_delta_bflows: [...] # 6 optical flows (adjacent frame)
    ├── 00001_xxx
    ...
    ├── 00002_xxx
    ...
    ...
    ├── 11405_xxx
    ...
    ├── __valid_keys__: [imgs, imgs_blur, ..., delta_bflows]
    └── __keys__: [...] # All the data keys in this file

The test file `CVO_test.lmdb` shares similar structure. 
All the arrays in a list is concatenated along the channel dimension.

### Download Links:
*Curretly, we only provide Baidu Netdisk links, other platforms will be included later*

**Split Compressed files:**

[CVO-test](https://pan.baidu.com/s/1nHwJiqlXMwPyD4DRPVioUw?pwd=7w27) (*Extracting Code: 7w27*)

[CVO-train](https://pan.baidu.com/s/1vyc7ZAwfvWwmrXjmXmixDw?pwd=9s6c) (*Extracting Code: 9s6c*)

[OneDrive Link](https://1drv.ms/f/s!AkDWlTIB7xY4ga9iQ-i5cvY-Fm0UlA?e=SQnb9a)

**Uncompressed LMDB files:**

[CVO-test](https://pan.baidu.com/s/1oTfVUfdgXemve3ocrKhL_A?pwd=vii9) (*Extracting Code: vii9*)

[CVO-train](https://pan.baidu.com/s/1Nz2lH3r4y5iC__PQKY6ioA?pwd=akep) (*Extracting Code: akep*)
