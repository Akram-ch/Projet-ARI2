# Projet ARI 
## Objectif du projet :
Ce Projet a pour objectif de partir d'une base qui est le classificateur d'images fourni dans le tutoriel, et essayer d'améliorer ses performances en introduisant des modifications dans les tailles de ces couches, en ajoutant des nouvelles couches etc.

**Nb : Les entraînements ont été effectués sur mon GPU personnel.**

## Instrumentation et évaluation "en continu" du système :
### Evaluation de performance :
On introduit 2 fonctions spécialisées pour mesurer la precision du modèle, une pour la précision globale,  et une pour mesurer la précision d'une classe à l'autre
```python
# Precision globale
def model_acc():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```
```python
# Precision par classe
def class_acc():
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
```

Ces deux fonctions sont d'abord appelés avant la première itération (avant l'aprentissage) pour mesurer les performances du "hasard".
en suite, model_acc() est appelée à la fin de chaque itération, pour mesurer l'évolution de la précision. class_acc() est appelée à la fin de l'entraînement pour voir le résultat final.

### Nombre d'operations flottantes :
On a introduit une fonction specialisée pour calculer le nombre d'opérations flottantes effectuées par le modèle, séparément pour les additions, les multiplications, les maximums et le total.
```python
def count_ops(model, input_shape):
    add_ops, mul_ops, max_ops, total_ops = 0, 0, 0, 0

    # Calculating the number of operations for the Conv2d layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.groups, layer.bias is not None
            input_shape = (input_shape[0], in_channels, input_shape[1], input_shape[2])
            output_shape = (input_shape[0], out_channels, (input_shape[2] + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1)//stride[0]+1, (input_shape[3] + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1)//stride[1]+1)
            n_ops = output_shape[1] * (2*kernel_size[0]*kernel_size[1]*input_shape[1] - 1) * output_shape[2] * output_shape[3]
            if bias:
                n_ops += output_shape[1] * output_shape[2] * output_shape[3]
            add_ops += n_ops
            mul_ops += n_ops

            input_shape = output_shape

    # Calculating the number of operations for the MaxPool2d layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.MaxPool2d):
            kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
            output_shape = (input_shape[0], input_shape[1], (input_shape[2] + 2*padding - dilation*(kernel_size-1)-1)//stride+1, (input_shape[3] + 2*padding - dilation*(kernel_size-1)-1)//stride+1)
            n_ops = output_shape[1] * kernel_size * kernel_size * output_shape[2] * output_shape[3]
            max_ops += n_ops
            total_ops += n_ops

            input_shape = output_shape

    # Calculating the number of operations for the Linear layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            in_features, out_features, bias = layer.in_features, layer.out_features, layer.bias is not None
            n_ops = in_features * out_features
            add_ops += n_ops
            mul_ops += n_ops * 2
            total_ops += n_ops * 2
            if bias:
                add_ops += out_features
                total_ops += out_features
        
    print("Nombre d'operations d'additions :" , add_ops  ,"\n")
    print("Nombre d'operations de multiplication :" , mul_ops  ,"\n")
    print("Nombre d'operations de maximisation :" , max_ops  ,"\n")
    print("Nombre total d'operations  :" , total_ops  ,"\n")

    return add_ops, mul_ops, max_ops, total_ops
```
### Nombre de paramètres :
```python
total_params = sum(p.numel() for p in net.parameters())
print(f"Total number of parameters: {total_params}")
```
### Sommaire du modèle :
```python
from torchsummary import summary
summary(net, input_size=(3, 32, 32))
```
## Modèle initial :

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Caractéristiques du réseau :
#### **Liste des fonctions de chaque couche et leurs caractéristiques :**
* Conv2d (couche 1) : 3 entrées, 6 sorties, taille du noyau 5x5.
* MaxPool2d (couche 1) : taille du noyau 2x2, stride 2.
* Conv2d (couche 2) : 6 entrées, 16 sorties, taille du noyau 5x5.
* Linear (couche 3) : 16x5x5 entrées, 120 sorties.
* Linear (couche 4) : 120 entrées, 84 sorties.
* Linear (couche 5) : 84 entrées, 10 sorties.

#### **données et leurs tailles :**
* Le dataset CIFAR-10 est composé d'un ensemble de 60000 images couleur de 32x32 pixels, réparties en 10 classes. Chaque image est donc représentée par un tenseur de taille 3x32x32, correspondant à 3 canaux de couleurs (rouge, vert, bleu) et une résolution de 32x32 pixels.

#### **liste des Wn (paramètres) :**
* conv1.weight : 6x3x5x5 poids (6 filtres de 5x5 pour chaque canal d'entrée).
* conv1.bias : 6 biais pour chaque filtre de la couche 1.
* conv2.weight : 16x6x5x5 poids (16 filtres de 5x5 pour chaque canal d'entrée).
* conv2.bias : 16 biais pour chaque filtre de la couche 2.
* fc1.weight : 120x400 poids (120 neurones avec 400 entrées).
* fc1.bias : 120 biais pour chaque neurone de la couche 3.
* fc2.weight : 84x120 poids (84 neurones avec 120 entrées).
* fc2.bias : 84 biais pour chaque neurone de la couche 4.
* fc3.weight : 10x84 poids (10 neurones avec 84 entrées).
* fc3.bias : 10 biais pour chaque neurone de la couche 5.
  
#### **Nombre de paramètres :** 62006
#### **Nombre d'operations :** 
```
Nombre d'operations d'additions : 995134 
Nombre d'operations de multiplication : 1053840 
Nombre d'operations de maximisation : 768 
Nombre total d'operations  : 118822 
```


#### **Sommaire du modèle :** 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             456
         MaxPool2d-2            [-1, 6, 14, 14]               0
            Conv2d-3           [-1, 16, 10, 10]           2,416
         MaxPool2d-4             [-1, 16, 5, 5]               0
            Linear-5                  [-1, 120]          48,120
            Linear-6                   [-1, 84]          10,164
            Linear-7                   [-1, 10]             850
================================================================
Total params: 62,006
Trainable params: 62,006
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.06
Params size (MB): 0.24
Estimated Total Size (MB): 0.31
----------------------------------------------------------------
```

#### **Resultats du hasard :**
```
Accuracy of the network on the 10000 test images: 10 %
Accuracy for class: plane is 0.0 %
Accuracy for class: car   is 0.0 %
Accuracy for class: bird  is 0.0 %
Accuracy for class: cat   is 100.0 %
Accuracy for class: deer  is 0.0 %
Accuracy for class: dog   is 0.0 %
Accuracy for class: frog  is 0.0 %
Accuracy for class: horse is 0.0 %
Accuracy for class: ship  is 0.0 %
Accuracy for class: truck is 0.1 %
```

#### **Resultats aprés entraînement :**
```
[1,  2000] loss: 2.261
[1,  4000] loss: 1.937
[1,  6000] loss: 1.733
[1,  8000] loss: 1.616
[1, 10000] loss: 1.526
[1, 12000] loss: 1.471
Time taken for epoch 1: 53.64 seconds
Accuracy of the network on the 10000 test images: 50 %
[2,  2000] loss: 1.404
[2,  4000] loss: 1.377
[2,  6000] loss: 1.353
[2,  8000] loss: 1.319
[2, 10000] loss: 1.323
[2, 12000] loss: 1.285
Time taken for epoch 2: 51.85 seconds
Accuracy of the network on the 10000 test images: 54 %
Accuracy for class: plane is 42.6 %
Accuracy for class: car   is 72.7 %
Accuracy for class: bird  is 48.7 %
Accuracy for class: cat   is 44.3 %
Accuracy for class: deer  is 37.9 %
Accuracy for class: dog   is 30.4 %
Accuracy for class: frog  is 75.8 %
Accuracy for class: horse is 56.6 %
Accuracy for class: ship  is 74.4 %
Accuracy for class: truck is 61.6 %
Finished Training
```

En se basant sur ce modèles et ces résultats, on va essayer en suite de l'améliorer à travers quelques modifications.

### 1 - Taille de Batch (Batch-size) : 

Le batch_size est un hyperparamètre important dans l'entraînement des modèles de machine learning. Il correspond au nombre d'échantillons qui sont présentés en même temps à l'algorithme lors d'une étape de mise à jour des poids du modèle.

Le choix du batch_size a un impact significatif sur l'entraînement du modèle. Un batch_size trop petit peut conduire à une convergence plus lente et moins stable du modèle, car l'algorithme de descente de gradient stochastique effectue des mises à jour plus fréquentes des poids. En revanche, un batch_size trop grand peut conduire à une utilisation inefficace de la mémoire du GPU, ce qui peut ralentir l'entraînement et empêcher la convergence.

**train_batch :** taille de batch d'entraînement.

**test_batch :** taille de batch de test.

#### **1-1 :**
```
train_batch = 20
test_batch = 400
```
```
[1,  2000] loss: 2.236
Time taken for epoch 1: 16.30 seconds
Accuracy of the network on the 10000 test images: 29 %
[2,  2000] loss: 1.805
Time taken for epoch 2: 15.50 seconds
Accuracy of the network on the 10000 test images: 41 %
Accuracy for class: plane is 46.7 %
Accuracy for class: car   is 65.5 %
Accuracy for class: bird  is 10.9 %
Accuracy for class: cat   is 21.5 %
Accuracy for class: deer  is 19.9 %
Accuracy for class: dog   is 42.0 %
Accuracy for class: frog  is 71.3 %
Accuracy for class: horse is 50.6 %
Accuracy for class: ship  is 38.5 %
Accuracy for class: truck is 51.3 %
Finished Training
```
On remarque que bien que le temps d'entraînement a été reduit d'une manière significative, la précision évolue moins rapidement et ne donne pas un resultat acceptable.

#### **1-2 :**

```
train_batch = 16
test_batch = 120
```

```
[1,  2000] loss: 2.211
Time taken for epoch 1: 17.56 seconds
Accuracy of the network on the 10000 test images: 34 %
[2,  2000] loss: 1.638
Time taken for epoch 2: 17.99 seconds
Accuracy of the network on the 10000 test images: 46 %
Accuracy for class: plane is 67.0 %
Accuracy for class: car   is 69.1 %
Accuracy for class: bird  is 17.3 %
Accuracy for class: cat   is 19.7 %
Accuracy for class: deer  is 27.5 %
Accuracy for class: dog   is 39.9 %
Accuracy for class: frog  is 65.7 %
Accuracy for class: horse is 55.5 %
Accuracy for class: ship  is 48.7 %
Accuracy for class: truck is 53.3 %
Finished Training
```
Amélioration legère en terme de précision

#### **1-3 :**
```
train_batch = 8
test_batch = 120
```
```
[1,  2000] loss: 2.231
[1,  4000] loss: 1.881
[1,  6000] loss: 1.643
Time taken for epoch 1: 25.20 seconds
Accuracy of the network on the 10000 test images: 42 %
[2,  2000] loss: 1.500
[2,  4000] loss: 1.442
[2,  6000] loss: 1.392
Time taken for epoch 2: 25.71 seconds
Accuracy of the network on the 10000 test images: 52 %
Accuracy for class: plane is 66.1 %
Accuracy for class: car   is 69.7 %
Accuracy for class: bird  is 45.1 %
Accuracy for class: cat   is 49.1 %
Accuracy for class: deer  is 44.0 %
Accuracy for class: dog   is 31.4 %
Accuracy for class: frog  is 62.8 %
Accuracy for class: horse is 50.3 %
Accuracy for class: ship  is 56.0 %
Accuracy for class: truck is 50.0 %
Finished Training
```

#### **conclusion :**
 En nous limitant à 2 itérations, dans notre cas, garder une taille de batch de 8 pour l'entraînement présente le meilleur compromis entre la précision et le temps d'entraînement.

 On garde donc cette version pour les prochaines modifications

 ### 2 - Tailles des couches :
 La modification suivante qu'on va apporter au modèle et de changer les tailles des differentes couches

 Le changement de la taille des couches peut avoir un impact significatif sur les performances du modèle. En effet, la taille des couches affecte la capacité du modèle à apprendre des motifs dans les données. Une couche trop petite peut ne pas être en mesure de capturer suffisamment d'informations pour effectuer une prédiction précise, tandis qu'une couche trop grande peut conduire à un surapprentissage et à une baisse de la généralisation.

 #### **2-1 :**

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]           2,432
         MaxPool2d-2           [-1, 32, 14, 14]               0
            Conv2d-3           [-1, 64, 10, 10]          51,264
         MaxPool2d-4             [-1, 64, 5, 5]               0
            Linear-5                  [-1, 128]         204,928
            Linear-6                   [-1, 64]           8,256
            Linear-7                   [-1, 10]             650
================================================================
Total params: 267,530
Trainable params: 267,530
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.30
Params size (MB): 1.02
Estimated Total Size (MB): 1.33
----------------------------------------------------------------
```
```
Nombre d'operations d'additions : 72789834 
Nombre d'operations de multiplication : 73003264 
Nombre d'operations de maximisation : 43008 
Nombre total d'operations  : 470474 
```
```
Accuracy of the network on the 10000 test images: 58 %
Accuracy for class: plane is 57.0 %
Accuracy for class: car   is 60.5 %
Accuracy for class: bird  is 44.0 %
Accuracy for class: cat   is 20.8 %
Accuracy for class: deer  is 53.1 %
Accuracy for class: dog   is 49.8 %
Accuracy for class: frog  is 76.7 %
Accuracy for class: horse is 81.7 %
Accuracy for class: ship  is 72.7 %
Accuracy for class: truck is 66.1 %
```

**nombre de paramètres :** 267530

On remarque une amélioration dans la précision du modèle, mais notre modèle dépasse les 256k paramètres donc on ne peut pas garder cette version.

#### **2-2 :**

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 25, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(25, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
Sommaire : 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 25, 28, 28]           1,900
         MaxPool2d-2           [-1, 25, 14, 14]               0
            Conv2d-3           [-1, 50, 10, 10]          31,300
         MaxPool2d-4             [-1, 50, 5, 5]               0
            Linear-5                  [-1, 100]         125,100
            Linear-6                   [-1, 50]           5,050
            Linear-7                   [-1, 10]             510
================================================================
Total params: 163,860
Trainable params: 163,860
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.24
Params size (MB): 0.63
Estimated Total Size (MB): 0.87
----------------------------------------------------------------
```
Nombre d'opérations :
```
Nombre d'operations d'additions : 34570660 
Nombre d'operations de multiplication : 34701000 
Nombre d'operations de maximisation : 24000 
Nombre total d'operations  : 285160 
```

Resultats :
```
[1,  2000] loss: 2.129
[1,  4000] loss: 1.677
[1,  6000] loss: 1.490
Time taken for epoch 1: 26.94 seconds
Accuracy of the network on the 10000 test images: 51 %
[2,  2000] loss: 1.341
[2,  4000] loss: 1.253
[2,  6000] loss: 1.182
Time taken for epoch 2: 32.12 seconds
Accuracy of the network on the 10000 test images: 61 %
Accuracy for class: plane is 53.9 %
Accuracy for class: car   is 73.0 %
Accuracy for class: bird  is 35.5 %
Accuracy for class: cat   is 48.1 %
Accuracy for class: deer  is 52.5 %
Accuracy for class: dog   is 49.5 %
Accuracy for class: frog  is 79.8 %
Accuracy for class: horse is 64.9 %
Accuracy for class: ship  is 83.1 %
Accuracy for class: truck is 72.0 %
Training time :  73.10 seconds
Finished Training
```

On remarque une bonne amélioration en précision par rapport au modèle initial, avec un temps d'entraînement de 73.10 secondes pour 2 itérations.
Et on est bien encore en dessous des bugets en nombre de paramètres, et on peut laisser tourner l'entraînement jusqu'à 49 iterations pour atteindre le temps d'entraînement maximal.

On peut alors proceder avec cette version et essayer de l'ameliorer encore plus.

### 3 - Ajouter des nouvelles couches :


#### **1-Batch Normalisation :**

L'ajout de couches de normalisation par lots dans un réseau de neurones peut accélérer l'apprentissage, régulariser le modèle, améliorer la généralisation et réduire les effets de la dépendance entre les activations de chaque couche.


Les avantages de l'utilisation de la normalisation par lots (Batch Normalization) sont :

* La réduction du surapprentissage (overfitting) en stabilisant les activations de chaque couche, ce qui permet une meilleure propagation du signal dans le réseau.

* L'amélioration de la vitesse de convergence de l'apprentissage en diminuant les problèmes de saturation des poids et en permettant une utilisation plus efficace des fonctions d'activation.

Les inconvénients peuvent être :

* La nécessité d'ajuster les hyperparamètres de la couche Batch Normalization pour optimiser la performance du modèle.
  
* L'augmentation de la complexité du modèle en ajoutant des paramètres supplémentaires pour chaque couche.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 25, 5),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, 5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 5 * 5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 25, 28, 28]           1,900
       BatchNorm2d-2           [-1, 25, 28, 28]              50
              ReLU-3           [-1, 25, 28, 28]               0
         MaxPool2d-4           [-1, 25, 14, 14]               0
            Conv2d-5           [-1, 50, 10, 10]          31,300
       BatchNorm2d-6           [-1, 50, 10, 10]             100
              ReLU-7           [-1, 50, 10, 10]               0
         MaxPool2d-8             [-1, 50, 5, 5]               0
            Linear-9                  [-1, 100]         125,100
      BatchNorm1d-10                  [-1, 100]             200
             ReLU-11                  [-1, 100]               0
           Linear-12                   [-1, 50]           5,050
      BatchNorm1d-13                   [-1, 50]             100
             ReLU-14                   [-1, 50]               0
           Linear-15                   [-1, 10]             510
================================================================
Total params: 164,310
Trainable params: 164,310
Non-trainable params: 0
```


Resultats :
```
[1,  2000] loss: 1.136
[1,  4000] loss: 1.110
[1,  6000] loss: 1.113
Time taken for epoch 1: 35.38 seconds
Accuracy of the network on the 10000 test images: 67 %
[2,  2000] loss: 1.038
[2,  4000] loss: 1.037
[2,  6000] loss: 1.045
Time taken for epoch 2: 35.79 seconds
Accuracy of the network on the 10000 test images: 69 %
Accuracy for class: plane is 73.8 %
Accuracy for class: car   is 85.3 %
Accuracy for class: bird  is 58.3 %
Accuracy for class: cat   is 46.0 %
Accuracy for class: deer  is 68.0 %
Accuracy for class: dog   is 53.2 %
Accuracy for class: frog  is 79.3 %
Accuracy for class: horse is 74.6 %
Accuracy for class: ship  is 82.3 %
Accuracy for class: truck is 78.7 %
Training time :  83.94 seconds
Finished Training
```

On constate qu'ajouter des couches de batch_normalization a permis d'ameliorer les performances du modèle en terme de précision avec un coût minimal en nombre de paramètres ajoutés et en temps d'entraînement.
On garde ainsi ces modifications.

#### **2-modules Dropout :**

L'ajout de couches Dropout est une autre technique couramment utilisée pour améliorer la performance et la généralisation des réseaux de neurones. Dropout est une technique de régularisation qui consiste à désactiver aléatoirement un certain pourcentage de neurones lors de chaque propagation avant (forward pass) pendant l'entraînement. Le taux de désactivation est généralement compris entre 0,2 et 0,5.

Les avantages de l'utilisation des couches Dropout sont :

* La réduction du surapprentissage (overfitting) en diminuant   l'interdépendance des neurones d'une couche.

* L'amélioration de la capacité de généralisation du modèle en évitant qu'une partie des neurones ne soit spécialisée dans l'apprentissage de 

Les inconvénients peuvent être :

* La diminution de la performance du modèle sur certaines tâches complexes, car une partie importante de l'information apprise par le réseau peut être perdue.

* L'augmentation du temps d'apprentissage, car les couches Dropout ralentissent l'apprentissage en forçant le modèle à s'adapter à des sous-ensembles de données différents à chaque époque.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 25, 5),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, 5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 5 * 5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 25, 28, 28]           1,900
       BatchNorm2d-2           [-1, 25, 28, 28]              50
              ReLU-3           [-1, 25, 28, 28]               0
         MaxPool2d-4           [-1, 25, 14, 14]               0
            Conv2d-5           [-1, 50, 10, 10]          31,300
       BatchNorm2d-6           [-1, 50, 10, 10]             100
              ReLU-7           [-1, 50, 10, 10]               0
         MaxPool2d-8             [-1, 50, 5, 5]               0
            Linear-9                  [-1, 100]         125,100
      BatchNorm1d-10                  [-1, 100]             200
             ReLU-11                  [-1, 100]               0
          Dropout-12                  [-1, 100]               0
           Linear-13                   [-1, 50]           5,050
      BatchNorm1d-14                   [-1, 50]             100
             ReLU-15                   [-1, 50]               0
          Dropout-16                   [-1, 50]               0
           Linear-17                   [-1, 10]             510
================================================================
Total params: 164,310
Trainable params: 164,310
Non-trainable params: 0
----------------------------------------------------------------
```
Resultat :
```
[1,  2000] loss: 1.811
[1,  4000] loss: 1.572
[1,  6000] loss: 1.496
Time taken for epoch 1: 43.39 seconds
Accuracy of the network on the 10000 test images: 55 %
[2,  2000] loss: 1.403
[2,  4000] loss: 1.354
[2,  6000] loss: 1.321
Time taken for epoch 2: 43.49 seconds
Accuracy of the network on the 10000 test images: 60 %
Accuracy for class: plane is 61.6 %
Accuracy for class: car   is 77.5 %
Accuracy for class: bird  is 43.5 %
Accuracy for class: cat   is 31.6 %
Accuracy for class: deer  is 48.7 %
Accuracy for class: dog   is 55.4 %
Accuracy for class: frog  is 79.8 %
Accuracy for class: horse is 67.4 %
Accuracy for class: ship  is 78.0 %
Accuracy for class: truck is 71.2 %
Training time :  100.37 seconds
Finished Training
```

L'ajout des couches dropout a augmenté le temps d'entraînement sans ameliorer la précision, on ne garde pas alors ces modifications.

### 4 - variantes des fonctions de perte et de l'optimiseur 
Quelques variantes qu'on peut utiliser sont :
*   nn.NLLLoss(), nn.BCEWithLogitsLoss() pour les fonctions de coûts
*   optim.Adam(), optim.Adagrad()  pour l'optimisateur

Dans notre cas, aucune de ces variante n'a apporté d'amélioration sur la précision du modèle, donc on garde les fonctions initiales.

### 5 - Augmentation des données 
L'augmentation des données est une technique qui consiste à créer de nouvelles données à partir des données d'entraînement existantes en appliquant des transformations telles que la rotation, le zoom, le décalage, la translation, etc. Cette technique permet d'augmenter le nombre de données d'entraînement et de réduire le risque de surapprentissage en permettant au modèle d'apprendre des variations des données existantes.

```python
transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     #transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```

```
[1,  2000] loss: 1.778
[1,  4000] loss: 1.558
[1,  6000] loss: 1.459
Time taken for epoch 1: 39.48 seconds
Accuracy of the network on the 10000 test images: 56 %
[2,  2000] loss: 1.352
[2,  4000] loss: 1.302
[2,  6000] loss: 1.274
Time taken for epoch 2: 41.05 seconds
Accuracy of the network on the 10000 test images: 62 %
Accuracy for class: plane is 61.5 %
Accuracy for class: car   is 79.1 %
Accuracy for class: bird  is 47.8 %
Accuracy for class: cat   is 39.5 %
Accuracy for class: deer  is 49.9 %
Accuracy for class: dog   is 45.7 %
Accuracy for class: frog  is 78.7 %
Accuracy for class: horse is 69.8 %
Accuracy for class: ship  is 81.8 %
Accuracy for class: truck is 68.9 %
Training time :  93.01 seconds
Finished Training
```
On n'observe pas de gain en précision sur les 2 itérations, mais on peut tester sur plus d'itérations à la fin du projet.

pour le moment, on procède sans augmentation des données.


### 6 - Description du réseau final.

```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 25, 5),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, 5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 5 * 5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 25, 28, 28]           1,900
       BatchNorm2d-2           [-1, 25, 28, 28]              50
              ReLU-3           [-1, 25, 28, 28]               0
         MaxPool2d-4           [-1, 25, 14, 14]               0
            Conv2d-5           [-1, 50, 10, 10]          31,300
       BatchNorm2d-6           [-1, 50, 10, 10]             100
              ReLU-7           [-1, 50, 10, 10]               0
         MaxPool2d-8             [-1, 50, 5, 5]               0
            Linear-9                  [-1, 100]         125,100
      BatchNorm1d-10                  [-1, 100]             200
             ReLU-11                  [-1, 100]               0
           Linear-12                   [-1, 50]           5,050
      BatchNorm1d-13                   [-1, 50]             100
             ReLU-14                   [-1, 50]               0
           Linear-15                   [-1, 10]             510
================================================================
Total params: 164,310
Trainable params: 164,310
Non-trainable params: 0
```
Resultats :
```
[1,  2000] loss: 1.136
[1,  4000] loss: 1.110
[1,  6000] loss: 1.113
Time taken for epoch 1: 35.38 seconds
Accuracy of the network on the 10000 test images: 67 %
[2,  2000] loss: 1.038
[2,  4000] loss: 1.037
[2,  6000] loss: 1.045
Time taken for epoch 2: 35.79 seconds
Accuracy of the network on the 10000 test images: 69 %
Accuracy for class: plane is 73.8 %
Accuracy for class: car   is 85.3 %
Accuracy for class: bird  is 58.3 %
Accuracy for class: cat   is 46.0 %
Accuracy for class: deer  is 68.0 %
Accuracy for class: dog   is 53.2 %
Accuracy for class: frog  is 79.3 %
Accuracy for class: horse is 74.6 %
Accuracy for class: ship  is 82.3 %
Accuracy for class: truck is 78.7 %
Training time :  83.94 seconds
Finished Training
```
Nombre d'operations :
```
Nombre d'operations d'additions : 34570660 
Nombre d'operations de multiplication : 34701000 
Nombre d'operations de maximisation : 30000 
Nombre total d'operations  : 69,301,660 
```

On va mainenant laisser tourner l'entraînement de ce modèle pour une dizaine d'itérations, et puis voir si l'ajout des couches dropout ou l'augmentation des données aura un effet plus significatif avec ce nombre d'itérations.


```
[1,  2000] loss: 1.729
[1,  4000] loss: 1.471
[1,  6000] loss: 1.365
Time taken for epoch 1: 37.36 seconds
Accuracy of the network on the 10000 test images: 58 %
[2,  2000] loss: 1.259
[2,  4000] loss: 1.224
[2,  6000] loss: 1.177
Time taken for epoch 2: 41.66 seconds
Accuracy of the network on the 10000 test images: 65 %
[3,  2000] loss: 1.108
[3,  4000] loss: 1.103
[3,  6000] loss: 1.096
Time taken for epoch 3: 40.28 seconds
Accuracy of the network on the 10000 test images: 68 %
[4,  2000] loss: 1.025
[4,  4000] loss: 1.034
[4,  6000] loss: 1.016
Time taken for epoch 4: 37.58 seconds
Accuracy of the network on the 10000 test images: 69 %
[5,  2000] loss: 0.956
[5,  4000] loss: 0.968
[5,  6000] loss: 0.961
Time taken for epoch 5: 38.85 seconds
Accuracy of the network on the 10000 test images: 71 %
[6,  2000] loss: 0.915
[6,  4000] loss: 0.917
[6,  6000] loss: 0.922
Time taken for epoch 6: 36.76 seconds
Accuracy of the network on the 10000 test images: 72 %
[7,  2000] loss: 0.876
[7,  4000] loss: 0.872
[7,  6000] loss: 0.883
Time taken for epoch 7: 38.23 seconds
Accuracy of the network on the 10000 test images: 72 %
[8,  2000] loss: 0.838
[8,  4000] loss: 0.849
[8,  6000] loss: 0.832
Time taken for epoch 8: 37.21 seconds
Accuracy of the network on the 10000 test images: 73 %
[9,  2000] loss: 0.797
[9,  4000] loss: 0.812
[9,  6000] loss: 0.819
Time taken for epoch 9: 37.74 seconds
Accuracy of the network on the 10000 test images: 73 %
[10,  2000] loss: 0.779
[10,  4000] loss: 0.785
[10,  6000] loss: 0.799
Time taken for epoch 10: 39.04 seconds
Accuracy of the network on the 10000 test images: 73 %
[11,  2000] loss: 0.746
[11,  4000] loss: 0.763
[11,  6000] loss: 0.769
Time taken for epoch 11: 36.42 seconds
Accuracy of the network on the 10000 test images: 73 %
[12,  2000] loss: 0.741
[12,  4000] loss: 0.722
[12,  6000] loss: 0.755
Time taken for epoch 12: 37.42 seconds
Accuracy of the network on the 10000 test images: 73 %
```
On remarque bien que l'évolution de la précision commence à stagner à partir de la 8ème itération. On va alors se limiter à 8 itérations.

essayons de voir si l'ajout des couches dropout peut améliorer la précision

```
[1,  2000] loss: 1.853
[1,  4000] loss: 1.650
[1,  6000] loss: 1.542
Time taken for epoch 1: 40.19 seconds
Accuracy of the network on the 10000 test images: 55 %
[2,  2000] loss: 1.462
[2,  4000] loss: 1.403
[2,  6000] loss: 1.379
Time taken for epoch 2: 40.80 seconds
Accuracy of the network on the 10000 test images: 61 %
[3,  2000] loss: 1.317
[3,  4000] loss: 1.302
[3,  6000] loss: 1.256
Time taken for epoch 3: 40.40 seconds
Accuracy of the network on the 10000 test images: 65 %
[4,  2000] loss: 1.223
[4,  4000] loss: 1.237
[4,  6000] loss: 1.199
Time taken for epoch 4: 42.53 seconds
Accuracy of the network on the 10000 test images: 67 %
[5,  2000] loss: 1.160
[5,  4000] loss: 1.168
[5,  6000] loss: 1.151
Time taken for epoch 5: 39.57 seconds
Accuracy of the network on the 10000 test images: 69 %
[6,  2000] loss: 1.102
[6,  4000] loss: 1.109
[6,  6000] loss: 1.110
Time taken for epoch 6: 38.40 seconds
Accuracy of the network on the 10000 test images: 70 %
[7,  2000] loss: 1.079
[7,  4000] loss: 1.066
[7,  6000] loss: 1.077
Time taken for epoch 7: 41.44 seconds
Accuracy of the network on the 10000 test images: 71 %
[8,  2000] loss: 1.037
[8,  4000] loss: 1.052
[8,  6000] loss: 1.035
Time taken for epoch 8: 43.11 seconds
Accuracy of the network on the 10000 test images: 71 %
[9,  2000] loss: 1.026
[9,  4000] loss: 1.015
[9,  6000] loss: 1.010
Time taken for epoch 9: 38.81 seconds
Accuracy of the network on the 10000 test images: 71 %
[10,  2000] loss: 0.979
[10,  4000] loss: 0.991
[10,  6000] loss: 0.996
Time taken for epoch 10: 38.62 seconds
Accuracy of the network on the 10000 test images: 72 %
[11,  2000] loss: 0.963
[11,  4000] loss: 0.970
[11,  6000] loss: 0.959
Time taken for epoch 11: 37.35 seconds
Accuracy of the network on the 10000 test images: 73 %
[12,  2000] loss: 0.955
[12,  4000] loss: 0.952
[12,  6000] loss: 0.953
Time taken for epoch 12: 38.88 seconds
Accuracy of the network on the 10000 test images: 73 %
[13,  2000] loss: 0.924
[13,  4000] loss: 0.938
[13,  6000] loss: 0.941
Time taken for epoch 13: 40.07 seconds
Accuracy of the network on the 10000 test images: 73 %
[14,  2000] loss: 0.895
[14,  4000] loss: 0.913
[14,  6000] loss: 0.934
Time taken for epoch 14: 39.00 seconds
Accuracy of the network on the 10000 test images: 73 %
[15,  2000] loss: 0.895
[15,  4000] loss: 0.903
[15,  6000] loss: 0.890
Time taken for epoch 15: 40.29 seconds
Accuracy of the network on the 10000 test images: 74 %
Accuracy for class: plane is 75.2 %
Accuracy for class: car   is 85.6 %
Accuracy for class: bird  is 58.4 %
Accuracy for class: cat   is 52.1 %
Accuracy for class: deer  is 77.1 %
Accuracy for class: dog   is 62.0 %
Accuracy for class: frog  is 81.0 %
Accuracy for class: horse is 80.5 %
Accuracy for class: ship  is 82.8 %
Accuracy for class: truck is 86.8 %
Training time :  701.96 seconds
Finished Training
```
Avec les couches dropout, la précision évolue plus lentement, et n'offre pas une grande amélioration sur le modèle précedent.

