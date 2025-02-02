from Enums.ColunsDatasetColuns import DatasetColuns
#Valores considerados como normais para cada medição
features_limits = { 
    DatasetColuns.creatinine_phosphokinase.value: (10, 20000),
    DatasetColuns.ejection_fraction.value: (5, 90),
    DatasetColuns.platelets.value: (5000, 1500000),
    DatasetColuns.serum_creatinine.value: (0.1, 15), 
    DatasetColuns.serum_sodium.value: (100, 180)
}