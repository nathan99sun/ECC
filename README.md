# ECC
eulerian: This project was made in PROMYS 2018 to generate data for q-Eulerian polynomials proposed by Paul Gunnells (https://arxiv.org/pdf/1702.02446.pdf). See https://arxiv.org/pdf/1809.07398.pdf Definition 2.3, 2.6 for algorithm, also Appendix A for data. As there is no  previously known efficient way to generate q-Eulerian polynomials, this program is able to generate data for n=10 with given data restraints, allowing us to make important conjectures about the properties of these polynomials. 

Secondary project: CIFARtest, seer1, and resnet are a CNN, DenseNet, and ResNet, respectively, trained on CIFAR100. Final commented accuracies are ran on 10 epochs each to compare the three models. 

IMPORTANT: when running these three programs, change the paths, as they save the best model for test and validation.

Densely Connected Neural Network(DenseNet) is a machine learning architecture inspired by CNN(Convolutional Neural Networks). Different from CNN, DenseNet concatenates layers of output to make sure components of inputs are connected. In doing so, intuitively speaking, DenseNets optimize their own operation layers in addition to optimizing output and thus become more and more efficient as training happens. DenseNets can deal with large data inputs without vanishing gradient and static learning rate phenomenons. We also compare the model to previous models of CNN and ResNet, to test this claim.
