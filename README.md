# Fashion-AttGAN: Attribute-Aware Fashion Editing with Multi-Objective GAN
This repo provides the dataset used in our paper "Fashion-AttGAN: Attribute-Aware Fashion Editing with Multi-Objective GAN" (https://arxiv.org/abs/1904.07460). 

## Dataset
The dataset contains 14,221 images and corresponding predicted attribute values. Currently we publish two attribute categories among others: clothing colors and sleeve lengths (data/data.json). 22 attribute values in total are selected in this published dataset. Therefore each image is associated with 1 color value and 1 sleeve-length value (these two values can be empty since there are other values of these two categories that are not covered in this dataset).

The image data is adapted from the VITON dataset, and can be downloaded [here](https://drive.google.com/file/d/1DBJY3wPyEDvcSvQjZkX8Kjg6nuRDRIa7/view?usp=sharing)

## Result
The results of our proposed Fashion-AttGAN and the original AttGAN are as follows:

![alt text](https://github.com/ChanningPing/Fashion_Attribute_Editing/blob/master/images/base_result.jpg)

![alt text](https://github.com/ChanningPing/Fashion_Attribute_Editing/blob/master/images/our_result.jpg)
###### Figure 1:  Clothing Attribute-Editing Results.  Top: Attribute-Editing Results of AttGAN. Bottom: Attribute-Editing Results of Fashion-AttGAN. From left to right columns:(1) original image, (2) reconstructed image, (3-6)varied sleeve lengths, (7-24) varied colors.

## Train
### Prerequisites
Tensorflow
### Data Preparation
Put all images under /data.
### Training
```
bash train.sh
```

## Cite
To use or cite this dataset, please use the following:
```
Qing Ping, Bing Wu, Wanying Ding, Jiangbo Yuan. 2019. Fashion-AttGAN: Attribute-Aware Fashion Editing with Multi-Objective GAN. FFSS-USAD Workshop at CVPR 2019. 
```

## Statistics
The statistics of different attribute values are summarized as follows:
<table>
  <tr>
    <td>Attribute Category</td>
    <td>Attribute Values</td>
    <td>Frequency</td>
  </tr>
  <tr>
    <td rowspan="18">Colors</td>
    <td>black</td>
    <td>3581</td>
  </tr>
  <tr>
    <td>white</td>
    <td>3264</td>
  </tr>
    <tr>
    <td>red</td>
    <td>1870</td>
  </tr>
  <tr>
    <td>grey</td>
    <td>1649</td>
  </tr>
  <tr>
    <td>navy-blue</td>
    <td>1291</td>
  </tr>
    <tr>
    <td>blue</td>
    <td>945</td>
  </tr>
    <tr>
    <td>pink</td>
    <td>778</td>
  </tr>
    <tr>
    <td>green</td>
    <td>696</td>
  </tr>
    <tr>
    <td>orange</td>
    <td>252</td>
  </tr>
    <tr>
    <td>purple</td>
    <td>217</td>
  </tr>
    <tr>
    <td>yellow</td>
    <td>153</td>
  </tr>
    <tr>
    <td>dark-brown</td>
    <td>85</td>
  </tr>
    <tr>
    <td>beige</td>
    <td>54</td>
  </tr>
    <tr>
    <td>apricot</td>
    <td>45</td>
  </tr>
    <tr>
    <td>camel-brown</td>
    <td>41</td>
  </tr>
    <tr>
    <td>sky-blue</td>
    <td>31</td>
  </tr>
    <tr>
    <td>medium-brown</td>
    <td>26</td>
  </tr>
  <tr>
    <td>light-brown</td>
    <td>15</td>
  </tr>
  <tr>
    <td rowspan="4">Sleeve Length</td>
    <td>long sleeve</td>
    <td>4151</td>
  </tr>
  </tr>
  <tr>
    <td>no sleeve</td>
    <td>2137</td>
  </tr>
  <tr>
    <td>cap sleeve</td>
    <td>468</td>
  </tr>
  <tr>
    <td>below elbow/three-quarter/seven-eighths sleeve</td>
    <td>152</td>
  </tr>
  </table>
  
  
  


