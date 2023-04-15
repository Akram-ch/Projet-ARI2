# Rapport Projet ARI


## Caractéristiques du réseau initial
### Liste des fonctions de chaque couche et leurs caractéristiques :
* Conv2d (couche 1) : 3 entrées, 6 sorties, taille du noyau 5x5.
* MaxPool2d (couche 1) : taille du noyau 2x2, stride 2.
* Conv2d (couche 2) : 6 entrées, 16 sorties, taille du noyau 5x5.
* Linear (couche 3) : 16x5x5 entrées, 120 sorties.
* Linear (couche 4) : 120 entrées, 84 sorties.
* Linear (couche 5) : 84 entrées, 10 sorties.

### données et leurs tailles :
* Le dataset CIFAR-10 est composé d'un ensemble de 60000 images couleur de 32x32 pixels, réparties en 10 classes. Chaque image est donc représentée par un tenseur de taille 3x32x32, correspondant à 3 canaux de couleurs (rouge, vert, bleu) et une résolution de 32x32 pixels.

 ### liste des Wn (paramètres) :
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

## Temps d'éxecution par époque :
dans cette étape on a ajouté la mesure du temps d'éxecuion pour chaque époque pour le réseau initial 
```
Time taken for epoch 1: 56.41 seconds
Time taken for epoch 2: 58.54 seconds
```
## mesures des précision du modèle initial:
### Performance du hasard
```
Accuracy of the network on the 10000 test images: 9 %
Accuracy for class: plane is 0.1 %
Accuracy for class: car   is 99.8 %
Accuracy for class: bird  is 0.0 %
Accuracy for class: cat   is 0.0 %
Accuracy for class: deer  is 0.0 %
Accuracy for class: dog   is 0.0 %
Accuracy for class: frog  is 0.0 %
Accuracy for class: horse is 0.0 %
Accuracy for class: ship  is 0.0 %
Accuracy for class: truck is 0.0 %
```
### Performance du modèle :
```
[1,  2000] loss: 2.218
[1,  4000] loss: 1.857
[1,  6000] loss: 1.704
[1,  8000] loss: 1.605
[1, 10000] loss: 1.545
[1, 12000] loss: 1.492
Time taken for epoch 1: 52.11 seconds
Accuracy of the network on the 10000 test images: 46 %
[2,  2000] loss: 1.422
[2,  4000] loss: 1.405
[2,  6000] loss: 1.359
[2,  8000] loss: 1.336
[2, 10000] loss: 1.310
[2, 12000] loss: 1.281
Time taken for epoch 2: 55.58 seconds
Accuracy of the network on the 10000 test images: 54 %
Accuracy for class: plane is 47.5 %
Accuracy for class: car   is 52.8 %
Accuracy for class: bird  is 43.0 %
Accuracy for class: cat   is 19.1 %
Accuracy for class: deer  is 43.1 %
Accuracy for class: dog   is 65.9 %
Accuracy for class: frog  is 64.8 %
Accuracy for class: horse is 60.3 %
Accuracy for class: ship  is 84.5 %
Accuracy for class: truck is 63.9 %
Finished Training
```
<<<<<<< HEAD


### entraînement sur 12 itérations : 
```
[1,  2000] loss: 2.208
[1,  4000] loss: 1.870
[1,  6000] loss: 1.727
[1,  8000] loss: 1.589
[1, 10000] loss: 1.533
[1, 12000] loss: 1.468
Time taken for epoch 1: 49.36 seconds
Accuracy of the network on the 10000 test images: 49 %
[2,  2000] loss: 1.376
[2,  4000] loss: 1.376
[2,  6000] loss: 1.336
[2,  8000] loss: 1.303
[2, 10000] loss: 1.300
[2, 12000] loss: 1.267
Time taken for epoch 2: 49.03 seconds
Accuracy of the network on the 10000 test images: 54 %
[3,  2000] loss: 1.196
[3,  4000] loss: 1.194
[3,  6000] loss: 1.177
[3,  8000] loss: 1.184
[3, 10000] loss: 1.164
[3, 12000] loss: 1.144
Time taken for epoch 3: 46.27 seconds
Accuracy of the network on the 10000 test images: 60 %
[4,  2000] loss: 1.071
[4,  4000] loss: 1.093
[4,  6000] loss: 1.091
[4,  8000] loss: 1.092
[4, 10000] loss: 1.098
[4, 12000] loss: 1.070
Time taken for epoch 4: 46.02 seconds
Accuracy of the network on the 10000 test images: 61 %
[5,  2000] loss: 1.005
[5,  4000] loss: 1.018
[5,  6000] loss: 1.017
[5,  8000] loss: 1.004
[5, 10000] loss: 1.011
[5, 12000] loss: 1.050
Time taken for epoch 5: 49.13 seconds
Accuracy of the network on the 10000 test images: 61 %
[6,  2000] loss: 0.937
[6,  4000] loss: 0.955
[6,  6000] loss: 0.972
[6,  8000] loss: 0.960
[6, 10000] loss: 0.967
[6, 12000] loss: 0.967
Time taken for epoch 6: 47.49 seconds
Accuracy of the network on the 10000 test images: 63 %
[7,  2000] loss: 0.853
[7,  4000] loss: 0.898
[7,  6000] loss: 0.901
[7,  8000] loss: 0.922
[7, 10000] loss: 0.927
[7, 12000] loss: 0.945
Time taken for epoch 7: 46.63 seconds
Accuracy of the network on the 10000 test images: 61 %
[8,  2000] loss: 0.828
[8,  4000] loss: 0.885
[8,  6000] loss: 0.854
[8,  8000] loss: 0.880
[8, 10000] loss: 0.906
[8, 12000] loss: 0.896
Time taken for epoch 8: 49.99 seconds
Accuracy of the network on the 10000 test images: 61 %
[9,  2000] loss: 0.797
[9,  4000] loss: 0.833
[9,  6000] loss: 0.843
[9,  8000] loss: 0.842
[9, 10000] loss: 0.853
[9, 12000] loss: 0.877
Time taken for epoch 9: 49.59 seconds
Accuracy of the network on the 10000 test images: 64 %
[10,  2000] loss: 0.755
[10,  4000] loss: 0.775
[10,  6000] loss: 0.816
[10,  8000] loss: 0.830
[10, 10000] loss: 0.842
[10, 12000] loss: 0.853
Time taken for epoch 10: 46.97 seconds
Accuracy of the network on the 10000 test images: 62 %
[11,  2000] loss: 0.723
[11,  4000] loss: 0.755
[11,  6000] loss: 0.794
[11,  8000] loss: 0.793
[11, 10000] loss: 0.799
[11, 12000] loss: 0.814
Time taken for epoch 11: 47.27 seconds
Accuracy of the network on the 10000 test images: 63 %
[12,  2000] loss: 0.718
[12,  4000] loss: 0.729
[12,  6000] loss: 0.744
[12,  8000] loss: 0.766
[12, 10000] loss: 0.794
[12, 12000] loss: 0.801
Time taken for epoch 12: 49.25 seconds
Accuracy of the network on the 10000 test images: 64 %
Accuracy for class: plane is 66.2 %
Accuracy for class: car   is 84.2 %
Accuracy for class: bird  is 56.1 %
Accuracy for class: cat   is 44.0 %
Accuracy for class: deer  is 54.9 %
Accuracy for class: dog   is 48.9 %
Accuracy for class: frog  is 77.3 %
Accuracy for class: horse is 64.0 %
Accuracy for class: ship  is 77.2 %
Accuracy for class: truck is 71.7 %
Finished Training
```
On remarque que les performances du modèle commencent à stagner à partir de la 7ème itération.

## Modification du réseau :
### Taille du batch 
#### **version 1 :** 
```
train_batch = 20
test_batch = 400
```
Résultat:
```
[1,  2000] loss: 2.113
Time taken for epoch 1: 15.74 seconds
Accuracy of the network on the 10000 test images: 36 %
[2,  2000] loss: 1.605
Time taken for epoch 2: 14.98 seconds
Accuracy of the network on the 10000 test images: 45 %
Accuracy for class: plane is 51.3 %
Accuracy for class: car   is 48.4 %
Accuracy for class: bird  is 11.6 %
Accuracy for class: cat   is 14.3 %
Accuracy for class: deer  is 41.9 %
Accuracy for class: dog   is 50.6 %
Accuracy for class: frog  is 67.8 %
Accuracy for class: horse is 60.2 %
Accuracy for class: ship  is 44.9 %
Accuracy for class: truck is 62.4 %
Finished Training
```

On peut remarquer qu'on augmentant la taille du batch, le temps d'entraînement est plus rapide, mais la convergence en précision d'une époque à l'autre est plus lente  en raison de l'affaiblissement de l'effet stochastique dans la descente de gradient.

#### **version 2 :**
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

#### **version3 :**
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
 En nous limitant à 2 itérations, dans notre cas, garder une taille de batch de 8 présente le meilleur compromis entre la précision et le temps d'entraînement

### Tailles des couches :

#### **version 1 :** 
```
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
* Conv2d (couche 1) : 3 entrées, 32 sorties, taille du noyau 5x5.
* MaxPool2d (couche 1) : taille du noyau 2x2, stride 2.
* Conv2d (couche 2) : 32 entrées, 64 sorties, taille du noyau 5x5.
* Linear (couche 3) : 64x5x5 entrées, 128 sorties.
* Linear (couche 4) : 128 entrées, 64 sorties.
* Linear (couche 5) : 64 entrées, 10 sorties.

nombre de paramètres : 267530

Resultat:
```
[1,  2000] loss: 2.085
[1,  4000] loss: 1.643
[1,  6000] loss: 1.446
Time taken for epoch 1: 29.92 seconds
Accuracy of the network on the 10000 test images: 47 %
[2,  2000] loss: 1.313
[2,  4000] loss: 1.229
[2,  6000] loss: 1.165
Time taken for epoch 2: 37.04 seconds
Accuracy of the network on the 10000 test images: 60 %
Accuracy for class: plane is 63.4 %
Accuracy for class: car   is 66.2 %
Accuracy for class: bird  is 44.8 %
Accuracy for class: cat   is 37.1 %
Accuracy for class: deer  is 56.5 %
Accuracy for class: dog   is 62.6 %
Accuracy for class: frog  is 76.5 %
Accuracy for class: horse is 70.2 %
Accuracy for class: ship  is 72.2 %
Accuracy for class: truck is 58.1 %
Finished Training
```

On peut remarquer une amélioration en terme de précision et une diminuation des valeurs de perte, mais aussi un temps d'entraînement plus important.


nb : on a gardé les tailles de batch train_batch = 8 et test_batch = 120

On a depassé les 256k paramètres donc on ne garde pas cette version.
#### **version2 :**
```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 36, 5)
        self.fc1 = nn.Linear(36 * 5 * 5, 72)
        self.fc2 = nn.Linear(72, 36)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
* Conv2d (couche 1) : 3 entrées, 18 sorties, taille du noyau 5x5.
* MaxPool2d (couche 1) : taille du noyau 2x2, stride 2.
* Conv2d (couche 2) : 18 entrées, 36 sorties, taille du noyau 5x5.
* Linear (couche 3) : 36x5x5 entrées, 72 sorties.
* Linear (couche 4) : 72 entrées, 36 sorties.
* Linear (couche 5) : 36 entrées, 10 sorties.

nombre de paramètres : 85474

Résultat:
```
[1,  2000] loss: 2.160
[1,  4000] loss: 1.719
[1,  6000] loss: 1.538
Time taken for epoch 1: 29.37 seconds
Accuracy of the network on the 10000 test images: 46 %
[2,  2000] loss: 1.415
[2,  4000] loss: 1.349
[2,  6000] loss: 1.264
Time taken for epoch 2: 27.40 seconds
Accuracy of the network on the 10000 test images: 57 %
Accuracy for class: plane is 65.2 %
Accuracy for class: car   is 71.6 %
Accuracy for class: bird  is 48.2 %
Accuracy for class: cat   is 20.6 %
Accuracy for class: deer  is 41.5 %
Accuracy for class: dog   is 56.7 %
Accuracy for class: frog  is 63.7 %
Accuracy for class: horse is 70.6 %
Accuracy for class: ship  is 70.6 %
Accuracy for class: truck is 65.5 %
Finished Training
```
cette version offre une meilleur performance tout en gardant un nombre de paramètres acceptable, on peut encore l'améliorer en ajoutant des couches supplementaires.
On garde cette version pour les prochaines ameliorations.
=======
>>>>>>> parent of fb6ada9 (cc)
