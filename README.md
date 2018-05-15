# car_detect_competition
a game that detect car  relative velocity and  relative distance

# summary

1. align 能够提高小目标的检测精度，RFCN版本的align,正常可用。https://github.com/haoyang11/R-FCN-PSROIAlign

2. 图片输入的大小对结果又影响，不同目标resize不同大小应该更合理，但是不是很本质。

3. 误差带来的思考，从投影到拟合。bias带来的思路的转换。

4. 不同噪声程度的滤波，小波滤波尝试。

5. 其他尝试：canny边缘检测，KCF做跟踪，kalman做滤波。

6. 其他工具：coco转pascon_voc格式，json可视化
