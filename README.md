# KAIST Pre_URP: Reinfocement Learning-based Self Driving Algirthm in Unity Virtual Environment

<strong>Reward Function Design</strong><br>
<li>Situation 1. Euclidean Distance Between Initial Position and Target Position</li>
<li>Situation 2. Lane Centering (no obstacles nearby)</li>
<li>Situation 3. Reached Target Position</li>
<li>Situation 4. Reached Target Position with lower time</li>

<br><br>

<strong>Model Architecture</strong>
<li>Implemented Deep Q-Network with 30 actions(-45~45 degree steering angle)</li>
<li>Epsilon Decay (starts from 40, 0.01)</li>

