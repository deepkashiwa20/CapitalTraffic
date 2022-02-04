# Capital Traffic and Accident
Working projects for forecasting traffic flow at accidents and incidents

* At first, please run the following commands:  
  cd ./dataplus
  <br>python gen_dataplus.py
  
* Our MMGCRN model is saved at ./model_ours/save/tokyo202112_MMGCRN_c1to1_20220204030801_history
  
  It can be reproduced by running the following commands:
  <br>cd ./model_ours
  <br>python retest_MMGCRN.py
  
  <br>Other ablation models (GCRN, MetaGCRN, MemGCRN) can also be reproduced in a similar way.
  <br>python retest_GCRN.py
  <br>python retest_MetaGCRN.py
  <br>python retest_MemGCRN.py


