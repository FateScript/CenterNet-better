# centernet.res50.coco.512size  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.723
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.996 | 53.532 | 37.222 | 15.005 | 40.426 | 52.151 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 46.488 | bicycle      | 24.646 | car            | 33.950 |  
| motorcycle    | 34.192 | airplane     | 57.787 | bus            | 59.410 |  
| train         | 57.824 | truck        | 29.154 | boat           | 19.716 |  
| traffic light | 20.152 | fire hydrant | 56.706 | stop sign      | 60.076 |  
| parking meter | 40.947 | bench        | 19.448 | bird           | 28.366 |  
| cat           | 62.626 | dog          | 57.599 | horse          | 54.194 |  
| sheep         | 47.858 | cow          | 49.406 | elephant       | 60.487 |  
| bear          | 65.966 | zebra        | 64.194 | giraffe        | 65.829 |  
| backpack      | 11.921 | umbrella     | 35.818 | handbag        | 10.447 |  
| tie           | 23.445 | suitcase     | 32.642 | frisbee        | 54.998 |  
| skis          | 18.959 | snowboard    | 30.049 | sports ball    | 30.998 |  
| kite          | 34.593 | baseball bat | 23.977 | baseball glove | 27.653 |  
| skateboard    | 45.796 | surfboard    | 31.309 | tennis racket  | 40.058 |  
| bottle        | 28.510 | wine glass   | 27.281 | cup            | 33.019 |  
| fork          | 26.461 | knife        | 12.096 | spoon          | 13.060 |  
| bowl          | 32.051 | banana       | 18.736 | apple          | 14.459 |  
| sandwich      | 33.101 | orange       | 26.227 | broccoli       | 18.156 |  
| carrot        | 16.218 | hot dog      | 27.507 | pizza          | 43.866 |  
| donut         | 40.909 | cake         | 32.701 | chair          | 23.469 |  
| couch         | 39.975 | potted plant | 22.998 | bed            | 34.893 |  
| dining table  | 16.043 | toilet       | 59.697 | tv             | 50.760 |  
| laptop        | 54.441 | mouse        | 51.537 | remote         | 17.960 |  
| keyboard      | 44.958 | cell phone   | 24.642 | microwave      | 49.011 |  
| oven          | 28.429 | toaster      | 24.910 | sink           | 30.723 |  
| refrigerator  | 49.575 | book         | 6.824  | clock          | 39.898 |  
| vase          | 31.487 | scissors     | 25.936 | teddy bear     | 39.742 |  
| hair drier    | 2.970  | toothbrush   | 14.782 |                |        |
