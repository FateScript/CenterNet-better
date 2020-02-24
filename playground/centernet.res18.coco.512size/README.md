# centernet.res18.coco.512size  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 29.843 | 46.608 | 31.559 | 10.854 | 33.146 | 44.896 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 42.136 | bicycle      | 20.436 | car            | 27.887 |  
| motorcycle    | 29.500 | airplane     | 53.821 | bus            | 53.821 |  
| train         | 55.510 | truck        | 23.589 | boat           | 15.162 |  
| traffic light | 15.166 | fire hydrant | 54.358 | stop sign      | 54.304 |  
| parking meter | 37.132 | bench        | 16.381 | bird           | 23.243 |  
| cat           | 56.821 | dog          | 51.461 | horse          | 47.794 |  
| sheep         | 39.360 | cow          | 43.659 | elephant       | 55.730 |  
| bear          | 67.505 | zebra        | 59.171 | giraffe        | 62.983 |  
| backpack      | 7.683  | umbrella     | 28.071 | handbag        | 5.416  |  
| tie           | 18.392 | suitcase     | 26.153 | frisbee        | 43.985 |  
| skis          | 12.904 | snowboard    | 20.594 | sports ball    | 27.206 |  
| kite          | 28.847 | baseball bat | 15.583 | baseball glove | 24.149 |  
| skateboard    | 38.321 | surfboard    | 25.889 | tennis racket  | 34.148 |  
| bottle        | 22.082 | wine glass   | 21.968 | cup            | 27.059 |  
| fork          | 16.727 | knife        | 6.432  | spoon          | 4.362  |  
| bowl          | 28.422 | banana       | 17.347 | apple          | 9.697  |  
| sandwich      | 27.608 | orange       | 24.074 | broccoli       | 17.818 |  
| carrot        | 14.870 | hot dog      | 24.158 | pizza          | 40.461 |  
| donut         | 34.080 | cake         | 26.088 | chair          | 18.486 |  
| couch         | 37.648 | potted plant | 16.580 | bed            | 33.362 |  
| dining table  | 15.311 | toilet       | 56.160 | tv             | 44.237 |  
| laptop        | 47.111 | mouse        | 43.463 | remote         | 10.357 |  
| keyboard      | 40.784 | cell phone   | 19.719 | microwave      | 46.002 |  
| oven          | 29.418 | toaster      | 5.418  | sink           | 28.024 |  
| refrigerator  | 45.025 | book         | 3.947  | clock          | 38.130 |  
| vase          | 25.934 | scissors     | 17.424 | teddy bear     | 31.751 |  
| hair drier    | 0.069  | toothbrush   | 5.522  |                |        |
