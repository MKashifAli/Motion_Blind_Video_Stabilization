# Deep Motion Blind Video Stabilization
The Code for the paper "Deep Motion Blind Video Stabilization" will be uploaded soon.

The Dataset can be downloaded from [this link](https://hyu-my.sharepoint.com/:u:/g/personal/kashifali_hanyang_ac_kr/EcHrM-0xmmpNiyrlZeUhDj8B6mRAVpSlSWH2jM6twbI7CQ?e=1rcrVE)


### Test scripts
Download the model checkpoints (all stages) from [this link](https://hyu-my.sharepoint.com/:u:/g/personal/kashifali_hanyang_ac_kr/EYJI4NLx_DFNtbC2wZkucjYBXI8MpAUfm5JDVnsCTZePQQ?e=wgeQ8o)

1. Extract the model checkpoints to the root directory.
2. Place the frames for videos in folders in the following format and pass the path to the folder containing videos (split into frames) to ```root```.
## Directory Structure

``` text
root
└───videos_to_test
│   └───Video_1
|   |   │   │   xxxx.png
|   |   │   │   xxxx.png
|   |   │   └───...
│   └───Video_2
|   |   │   │   xxxx.png
|   |   │   │   xxxx.png
|   |   │   └───...
│   └───Video_3
|   |   │   │   xxxx.png
|   |   │   │   xxxx.png
|   |   │   └───...
```
3. Modify the ```test\_only.py```  according to the stage that you would like to test. (both checkpoint name and the function name to load the models).
4. The folder for output will be generated according to the checkpoint name.
