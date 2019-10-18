# Fast-and-Coherent-Video-Object-Segmentation

-------------------------------------------------------------------------
Graph cut based Video Object Segmentation in Bilateral Space
README (13/10/2019)
-------------------------------------------------------------------------

This code is a simplified implementation of the video object segmentation method described in the following paper: 
Yan Gui, Ying Tan, Dao-Jian Zeng, Zhi-Feng Xie and Yi-Yu Cai: "Reliable and Dynamic Appearance Modeling and Label Consistency Enforcing for Fast and Coherent Video Object Segmentation with the Bilateral Grid", Aug. 2019.

1. We implemented our prototype system with  C++ and the OpenCV 3.1.0 library. 
   
2. Clone "VOS - user interaction" to your local, and run "Bilateral.sln". Here, we allow the user to manually give sparse cues in the keyframe(s). These cues are a sparse set of labels specified by user scribbles, which are propagated to unlabeled image regions according to propagation principle based on pixel similarity. The segmentation process is triggered once the keyframe(s) are annotated.

3. Clone "VOS - user-specified object masks" to your local, and run "Bilateral.sln". Here, we require the user to provide a pixel-wise segmentation on a few keyframes for initialization.

4. Clone "Ablation Study" to your local. Here, the codes are the implementation of the ablation variants of our method.

5. Clone "Evaluation" to your local. Here, this Package contains the Matlab implementation of the code behind: 
*A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation* 
[DAVIS](https://graphics.ethz.ch/~perazzif/davis/index.html). 
Please cite 'DAVIS' in your publications if it helps your research.


You can use these codes for scientific purposes only. Use in commercial projects and redistribution are not allowed without author's permission. Please cite (https://github.com/guiyan2018/Fast-and-Coherent-Video-Object-Segmentation.git) when using this code. 

====================================================

Personal Contact Information

====================================================

Email:
	（tiany2033@163.com）		(Ying Tan)
