# Capital Traffic and Accident
Working projects for forecasting traffic flow at accidents and incidents

* At first, please run the following commands:  
  cd ./dataplus
  <br>python gen_dataplus.py
  
* Our MMGCRN model is saved at ./model_ours/save/tokyo202112_MMGCRN_c1to1_20220204030801_history
  
  It can be reproduced by running the following commands:
  <br>cd ./model_ours
  <br>python retest_MMGCRN.py
  
  Other ablation models (GCRN, MetaGCRN, MemGCRN) can also be reproduced in a similar way.
  <br>python retest_GCRN.py
  <br>python retest_MetaGCRN.py
  <br>python retest_MemGCRN.py

* GW-Net model is saved at ./baseline/save/
  <br> GW-Net (x only): tokyo202112_GraphWaveNet_c1to1_20220204015721
  <br> GW-Net (x + tcov): tokyo202112_GraphWaveNet_c2to1_20220204015728_time
  <br> GW-Net (x + hcov): tokyo202112_GraphWaveNet_c2to1_20220204015732_history
  <br> GW-Net (x + tcov) is the default. 
  
  It can be reproduced by running the following commands:
  <br>cd ./baseline
  <br>python retest_GraphWaveNet.py
  <br>where the current default target is (to reproduce) GW-Net (x + tcov): tokyo202112_GraphWaveNet_c2to1_20220204015728_time
