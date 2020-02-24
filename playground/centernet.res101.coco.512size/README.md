# centernet.res101.coco.512size  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.250
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.822 | 55.444 | 38.948 | 16.094 | 42.418 | 55.546 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 47.599 | bicycle      | 26.318 | car            | 34.760 |  
| motorcycle    | 34.987 | airplane     | 58.181 | bus            | 60.491 |  
| train         | 59.755 | truck        | 31.674 | boat           | 21.044 |  
| traffic light | 20.269 | fire hydrant | 63.556 | stop sign      | 61.995 |  
| parking meter | 41.975 | bench        | 20.782 | bird           | 29.717 |  
| cat           | 65.989 | dog          | 60.865 | horse          | 58.227 |  
| sheep         | 49.664 | cow          | 50.273 | elephant       | 62.381 |  
| bear          | 75.254 | zebra        | 65.308 | giraffe        | 66.469 |  
| backpack      | 12.751 | umbrella     | 34.768 | handbag        | 11.160 |  
| tie           | 25.070 | suitcase     | 35.482 | frisbee        | 53.366 |  
| skis          | 19.810 | snowboard    | 31.453 | sports ball    | 31.564 |  
| kite          | 36.350 | baseball bat | 23.447 | baseball glove | 27.932 |  
| skateboard    | 47.818 | surfboard    | 33.947 | tennis racket  | 41.908 |  
| bottle        | 29.796 | wine glass   | 28.649 | cup            | 35.280 |  
| fork          | 32.150 | knife        | 13.279 | spoon          | 13.807 |  
| bowl          | 34.606 | banana       | 19.621 | apple          | 16.669 |  
| sandwich      | 31.665 | orange       | 27.548 | broccoli       | 19.957 |  
| carrot        | 16.677 | hot dog      | 30.329 | pizza          | 47.912 |  
| donut         | 43.314 | cake         | 34.797 | chair          | 25.633 |  
| couch         | 45.630 | potted plant | 24.237 | bed            | 36.002 |  
| dining table  | 17.468 | toilet       | 62.687 | tv             | 52.205 |  
| laptop        | 56.512 | mouse        | 52.113 | remote         | 19.695 |  
| keyboard      | 47.411 | cell phone   | 27.388 | microwave      | 55.441 |  
| oven          | 33.335 | toaster      | 22.460 | sink           | 32.417 |  
| refrigerator  | 49.899 | book         | 7.776  | clock          | 42.461 |  
| vase          | 33.686 | scissors     | 27.542 | teddy bear     | 46.708 |  
| hair drier    | 0.682  | toothbrush   | 15.991 |                |        |
