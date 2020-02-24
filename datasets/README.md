We only support coco dataset now, more datasets will be supported in the future.  
The datasets are assumed to exist in a directory called
"datasets/", under the directory where you launch the program.
They need to have the following directory structure:

## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.  
If you have COCO dataset in another directory, soft link also works.
```shell
ln -s path_of_your_dataset coco
```
