**About the solution**

***Tell us about your model. How many classifiers did you build? Which architecture had each of them? How were they different from one another? Did you use transfer learning? If they were pre-trained, in which dataset? How did you ensemble the different model's predictions?*** 

The architecture I chose is [SE_ResNext50](https://arxiv.org/abs/1709.01507). It has a good balance between accuracy and computation. More importantly, it fits on a single GPU with acceptable training time for experiments.

In the training phase, I split the training images into subsets, then train 8 model on each subset. They have the same input image size, same data augmentation, batch size, learning rate and epochs. The technique is called k-fold cross-validation (k=8 here). Each fold has its train and validation subset.

![k-fold cv](https://raw.githubusercontent.com/LiberiFatali/cassava_and_snake/master/K-fold_cross_validation_EN.jpg)

To initialize network weights, I applied transfer learning technique. Models were pre-trained on ImageNet.

In the test phase, I used simple averaging on the probabilities, in which all classes have equal priority to vote.

***Which tricks did you use to train the model that others may not have used and put you on the top?***

Train with a higher solution (448x448) and 8-fold cross-validation. More epochs on training until no improvement.

***Which was your best idea?***

In round 1, I used a single InceptionResnetV2 model (best F1_score 0.802). In round 2, I switched to SE_ResNext50 with k-fold cross-validation (best F1_score 0.847). It’s my first time to train SE_ResNext50 models and make use of k-fold cross-validation

***How did you deal with the data imbalance problem?***

Stratified k-fold to split data randomly into subsets while preserving the percentage of images for each class. Data augmentation in train and validation.

***How did you deal with the fact that images had very different sizes? To which size did you resize the images to make it work best?***

Round 1, I use image size 299x299. Round 2, I resize to 448x448, which is a higher resolution.

***After submitting the first time, which things did you change in your model to achieve the performance you finally had.***

In round 1, I used a single InceptionResnetV2 model that was fine-tuned on whole training images with the default image size of the model. In round 2, I built SE_ResNext50 with 8-fold cross-validation higher resolution images; in other words, I trained 8 models on 8 subsets of training data.

***How did you reduce the loss? e.g. make your model more confident about its predictions.***

At the inference phase, I applied a simple Test Time Augmentation: averaging predictions on the original image, flipped left-right and flipped up-down images. This trick increased accuracy to about 0.2 percent, which is just enough to get me into the top 3.

***Which aspects of the challenge were the most challenging and how did you solve it?***

The natural characteristics of snakes make it very challenging to separate them from the environment. Many images lack visual feature on the snake, while there are many background noises. The training set has over 82k images that are rather big for a single GPU. The challenge has a long timeline so we may lose focus. The submission protocol in round 2 is complicated, time-consuming, error-prone and inconvenient to debug.
