# A Geometry-based Stochastic Wireless Channel Model using Channel Images

 ### In this work, we obtain channel parameters from the ray-tracing simulation in a specific area and process them in the form of images to train any generative model.
### For the detailed descriptions of this implementation, we kindly suggest to read the paper, [A Geometry-based Stochastic Wireless Channel Model using Channel Images](https://arxiv.org/abs/2312.06637)
### Extended decription -> [Geometry-based Stochastic Wireless Channel Model using Generative Neural Networks](https://www.techrxiv.org/doi/full/10.36227/techrxiv.172114994.43396874)

# Channel Image Generation and Model Training

### Firstly, we obtain channel parameters from ray-tracing simulate and create the matrices $\boldsymbol{D}$ corresponding to each Tx-Rx link. 
![ray_matrix](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/6322f387-f304-459d-ab16-583da47d369c)



### After then, we normalize the matrices with Min-Max scaling and create images by enlarging the matrices horizontally and vertically. 
![enlarging process](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/e139e848-21bf-4b0a-8d85-b0f6024849dc)

### Next, we train WGAN-GP with the channel images. The followings show the output images at each epoch of training. 
![gan_training_process](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/06a123f7-3304-41b9-ad30-b23a33d8f36e)

### After training the WGAN-GP, we sample data from the outputs of the model.  
![sampling_process](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/ffc2cd81-1265-4521-a81b-4dc6023860e8)


# Performance Evaluation

### CDFs of pathloss and delay obtained from the trained model are compared with those from the original data. 
![path_loss](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/8b9cf7d4-e89f-4335-800e-ffb67ffc296a)
![delay](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/e2108d9d-226f-41c5-be13-d38aef931f10)

### In addition, we compare the link state probabilities (LOS and outages). 

![los_prob_new](https://github.com/sk8053/GeoStochasticChanModel/assets/59175938/6d872cac-fcd0-4bf5-ab55-751183596338)
