# FaceGenderID: Exploiting Gender Information in DCNNs Face Recognition Systems

Here we provide the FaceGenderID pretrained DCNNs models for face recognition. There are gender specific models (one for males and one for females) and one gender balanced model. These models are adapted for VGG-Face and ResNet-50 face DCNNs models to perform better for each specific gender. The proposed system is the following: 

![](http://atvs.ii.uam.es/atvs/FaceGenderID_1.png )
Results from our paper [1] show significant improvements of performance when using these models compared to a general face recognition DCNN model such as VGG-Face or ResNet-50.

Main results from [1]:

![](http://atvs.ii.uam.es/atvs/FaceGenderID_2.png )
This figure shows ROC curves for face verification showing AUC(Area under the curve) for each case. 

If you use these pretrained models, make sure you cite the work:

[1] Ruben Vera-Rodriguez, Marta Blazquez, Aythami Morales, Ester Gonzalez-Sosa, João C. Neves, and Hugo Proença, “FaceGenderID: Exploiting Gender Information in DCNNs Face Recognition Systems”, in Proc. CVPR workshop on Bias Estimation in Face Analytics, 2019.
