### Image Classification des Früchte-Datensatzes
- Implementation eines neuronalen Netzwerks zur Klassifizierung des Früchte-Datensatzes
- Der Datensatz enthält 224 x 224 Pixel große Bilder mit 5 Klassen und 854 Trainingsinstanzen

### VGG16 Implementierung
- Implementiation der Funktionen `__init__()` und `forward()`.
- Transfer Learning der Architektur mit einem vorab trainierten VGG16-Modell aus dem PyTorch "models"-Paket, wobei die Layer eingefroren wurden und der letzte Layer durch einen fully connected Layer ersetzt wurde

### Optimierer
- Implementation `configure_optimizers()`, mit dem Adam-Optimierer mit einer Lernrate von 0.01 und verwende den StepLR-Lernratenplaner

### Trainings-, Validierungs- und Test-Schritt
- Implementiation `training_step()`, `validation_step()` und `test_step()` für einzelne Iterationen der Trainings-, Validierungs- und Test-Schleifen mit Protokolierung der jeweiligen Verluste und Genauigkeiten

### Data_Loading
- Laden des Trainings- und Testdatensatz des Früchte-Datensatzes und Vorverarbeitung: Umwandlung in PyTorch-Tensoren und Skalierung auf (224,224)
- Einteilung des Trainingsdatensatz in einen Validierungs- und einen kleineren Trainingsdatensatz. Der Validierungssatz macht 10 % des Gesamttrainingsdatensatzes aus
  
### Training und Bewertung
- Implementation das Training und Bewertung mit einem PyTorch Lightning Trainer, der eine GPU, maximal 100 Epochen, einen Tensorboard-Logger, einen Protokollierungsschritt von 10 und einen Early-Stopping-Callback verwendet

### Tensorboard
- Implementation Tensorboard zur Visualisierung der Protokolle.
