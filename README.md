# Capital Traffic and Accident
Working projects for forecasting traffic flow at accidents and incidents
* Goolge Sheets
 https://docs.google.com/spreadsheets/d/1BxwzAauyfT6aYyDYXhZ0VrCxO-tV94iYX8KYBjrIwrs/edit?usp=sharing

* At first, please run the following commands:  
  cd dataplus
  <br>python gen_dataplus.py
  
* MMGCRN（tcov）: save/tokyo202112_MMGCRN_c1to1_20220208044416_time
  
  It can be reproduced by running the following commands:
  <br>cd model_ours
  <br>python retest_MMGCRN.py

* GW-Net (tcov): save/tokyo202112_GraphWaveNet_c2to1_20220208044941_time
  
  It can be reproduced by running the following commands:
  <br>cd baseline
  <br>python retest_GraphWaveNet.py
