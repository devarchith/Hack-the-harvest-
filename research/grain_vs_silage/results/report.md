# Grain vs silage: full validation report

Inputs: S2 series_barc.parquet, Landsat series_barc.parquet, weather weather_barc.parquet

Sensor offsets (median S2 minus Landsat, same field and day, all fields; added to Landsat):
      pairs   ndvi   ndti   ndmi
year                            
2021  267.0 -0.021 -0.012 -0.026
2022  345.0 -0.027 -0.011 -0.030
2023  448.0 -0.027 -0.020 -0.056

Labeled field-years: 44 (Grain 32, Silage 12); blank labels are unlabeled and excluded.
label  Grain  Silage
year                
2021      14       4
2022       3       4
2023      15       4
Min usable readings Aug-Nov per labeled field-year: 13

## Univariate separation (AUC, Silage = positive; 0.5 = none, 1.0 or 0.0 = perfect)
           feature  n   auc  separation silage_is  auc_central_farm_only  auc_2021  auc_2022  auc_2023  grain_median  silage_median
       t_below_0.3 35 0.008       0.992     lower                  0.000     0.000     0.000     0.000       302.325        255.060
          regrowth 44 0.987       0.987    higher                  0.994     0.929     1.000     1.000        -0.013          0.398
    ndti_post_drop 44 0.023       0.977     lower                  0.026     0.054     0.000     0.033         0.195          0.119
        ndti_sep15 44 0.034       0.966     lower                  0.019     0.107     0.000     0.033         0.247          0.125
         drop_size 44 0.964       0.964    higher                  0.968     0.929     1.000     0.967         0.180          0.512
        ndvi_sep15 44 0.036       0.964     lower                  0.032     0.143     0.000     0.000         0.538          0.278
        ndti_oct01 44 0.047       0.953     lower                  0.013     0.125     0.000     0.067         0.198          0.113
       t_below_0.4 44 0.073       0.927     lower                  0.039     0.232     0.000     0.017       267.992        252.869
        ndmi_sep15 44 0.115       0.885     lower                  0.045     0.161     0.083     0.100         0.039         -0.167
       t_below_0.5 44 0.128       0.872     lower                  0.032     0.268     0.000     0.100       260.841        250.443
       season_days 44 0.151       0.849     lower                  0.143     0.214     0.083     0.133       102.083         87.681
     ndvi_pre_drop 44 0.849       0.849    higher                  0.779     0.768     0.750     0.933         0.682          0.819
        ndmi_nov01 44 0.833       0.833    higher                  0.799     0.964     0.667     0.817        -0.100          0.121
    ndvi_post_drop 44 0.188       0.812     lower                  0.136     0.286     0.000     0.183         0.461          0.315
        ndvi_oct01 44 0.211       0.789     lower                  0.123     0.286     0.083     0.100         0.358          0.261
       t_below_0.6 44 0.219       0.781     lower                  0.104     0.446     0.000     0.183       253.715        248.026
         drop_rate 44 0.779       0.779    higher                  0.799     0.857     1.000     0.950         0.030          0.083
        ndvi_nov01 44 0.773       0.773    higher                  0.766     0.946     0.667     0.817         0.358          0.616
        ndmi_oct15 44 0.771       0.771    higher                  0.643     0.875     0.583     0.700        -0.146         -0.082
gdd_greenup_to_0.4 44 0.237       0.763     lower                  0.292     0.214     0.667     0.167      1435.573       1275.050

## Leave-one-year-out: train on two years, test on the third

### Farm only (Silage if Central Farm): confound check, not a method
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[18, 14], [1, 11]]
Grain  precision 0.95  recall 0.56 | Silage precision 0.44  recall 0.92 | accuracy 0.66  balanced acc 0.74
per held-out year: 2021: 13/18 correct (4 silage); 2022: 3/7 correct (4 silage); 2023: 13/19 correct (4 silage)

### H1 timing: one cut on the date of the largest NDVI drop
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[17, 15], [0, 12]]
Grain  precision 1.00  recall 0.53 | Silage precision 0.44  recall 1.00 | accuracy 0.66  balanced acc 0.77
per held-out year: 2021: 11/18 correct (4 silage); 2022: 6/7 correct (4 silage); 2023: 12/19 correct (4 silage)
rule chosen on the training years: 2021: drop_doy < 254.500 (train bal-acc 0.78) | 2022: drop_doy < 253.250 (train bal-acc 0.79) | 2023: drop_doy < 253.250 (train bal-acc 0.82)

### H2 cut green: one cut on NDVI just before the largest drop
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[18, 14], [2, 10]]
Grain  precision 0.90  recall 0.56 | Silage precision 0.42  recall 0.83 | accuracy 0.64  balanced acc 0.70
per held-out year: 2021: 13/18 correct (4 silage); 2022: 5/7 correct (4 silage); 2023: 10/19 correct (4 silage)
rule chosen on the training years: 2021: ndvi_pre_drop > 0.797 (train bal-acc 0.89) | 2022: ndvi_pre_drop > 0.789 (train bal-acc 0.83) | 2023: ndvi_pre_drop > 0.670 (train bal-acc 0.79)

### Best single feature, chosen inside each training fold (nested; no peeking at the test year)
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[32, 0], [3, 9]]
Grain  precision 0.91  recall 1.00 | Silage precision 1.00  recall 0.75 | accuracy 0.93  balanced acc 0.88
per held-out year: 2021: 17/18 correct (4 silage); 2022: 7/7 correct (4 silage); 2023: 17/19 correct (4 silage)
rule chosen on the training years: 2021: regrowth > 0.258 (train bal-acc 1.00) | 2022: t_below_0.3 < 267.521 (train bal-acc 0.95) | 2023: t_below_0.3 < 260.292 (train bal-acc 1.00)

### Logistic regression on 5 hypothesis features (threshold 0.5)
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[31, 1], [1, 11]]
Grain  precision 0.97  recall 0.97 | Silage precision 0.92  recall 0.92 | accuracy 0.95  balanced acc 0.94
per held-out year: 2021: 17/18 correct (4 silage); 2022: 6/7 correct (4 silage); 2023: 19/19 correct (4 silage)
held-out ROC-AUC 0.964

### Random forest on all features (threshold 0.5)
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[32, 0], [1, 11]]
Grain  precision 0.97  recall 1.00 | Silage precision 1.00  recall 0.92 | accuracy 0.98  balanced acc 0.96
per held-out year: 2021: 17/18 correct (4 silage); 2022: 7/7 correct (4 silage); 2023: 19/19 correct (4 silage)
held-out ROC-AUC 0.945

### Logistic regression, strict: held-out year AND its fields removed from training
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[31, 1], [1, 11]]
Grain  precision 0.97  recall 0.97 | Silage precision 0.92  recall 0.92 | accuracy 0.95  balanced acc 0.94
per held-out year: 2021: 17/18 correct (4 silage); 2022: 6/7 correct (4 silage); 2023: 19/19 correct (4 silage)
held-out ROC-AUC 0.964
Silage recall (logistic, LOYO) 11/12, 95% Wilson interval 0.65-0.99
Grain recall (logistic, LOYO) 31/32, 95% Wilson interval 0.84-0.99

## Learned weights: logistic regression on all 44, standardized features (+ pushes toward Silage)
ndti_post_drop   -1.00
drop_size         0.82
ndvi_sep15       -0.72
ndvi_pre_drop     0.56
drop_doy         -0.11
scores_all_field_years.csv: p_silage for all 249 field-years from the model fit on the 44. Only meaningful where the field was corn; most unlabeled field-years are not corn.

## Field-years the logistic model gets wrong on the held-out year
 year         farm field  label  logreg_p_silage  drop_doy  ndvi_pre_drop  drop_size  ndti_post_drop
 2022 Central Farm 2-32B  Grain            0.546     248.0          0.826      0.368           0.219
 2021 Central Farm   4-7 Silage            0.121     252.5          0.678      0.201           0.175

## Harvest-date check: 38 farm-record dates in [np.int64(2021)], 16 matched to a BARC polygon
              idn  year record_label  label harvest_date  harvest_doy  drop_doy  err_drop  t_below_0.4  err_t04
           APU_30  2021        Grain  Grain   2021-09-20          263     254.0      -9.0        260.0     -3.0
            APU_1  2021        Grain  Grain   2021-09-21          264     204.0     -60.0        254.2     -9.8
    EastFarm_5-12  2021        Grain  Grain   2021-09-28          271     237.5     -33.5        257.9    -13.1
CentralFarm_1-14B  2021        Grain    NaN   2021-10-04          277     236.5     -40.5        268.1     -8.9
    EastFarm_5-21  2021        Grain  Grain   2021-10-13          286     237.5     -48.5        269.4    -16.6
    EastFarm_5-23  2021        Grain  Grain   2021-10-13          286     230.0     -56.0        268.9    -17.1
   EastFarm_5-45A  2021        Grain    NaN   2021-10-13          286     252.5     -33.5        267.9    -18.1
 CentralFarm_1-15  2021        Grain  Grain   2021-10-20          293     259.0     -34.0        267.6    -25.4
 CentralFarm_2-51  2021        Grain  Grain   2021-10-28          301     301.0       0.0        259.7    -41.3
 CentralFarm_2-66  2021        Grain  Grain   2021-10-28          301     262.0     -39.0        271.0    -30.0
 LinkageFarm_EB-3  2021        Grain  Grain   2021-11-03          307     236.5     -70.5        268.2    -38.8
 CentralFarm_1-34  2021        Grain    NaN   2021-11-05          309     311.0       2.0        260.4    -48.6
   NorthFarm_NE-3  2021        Grain  Grain   2021-11-15          319     316.5      -2.5        262.4    -56.6
CentralFarm_2-32C  2021       Silage Silage   2021-09-07          250     251.5       1.5        252.4      2.4
  CentralFarm_4-8  2021       Silage Silage   2021-09-08          251     252.5       1.5        253.3      2.3
  CentralFarm_4-7  2021       Silage Silage   2021-11-04          308     252.5     -55.5        270.0    -38.0
err_drop: n=16 median -33.8 d, median |error| 33.8 d, within 10 days 6/16
err_t04: n=16 median -17.6 d, median |error| 17.6 d, within 10 days 5/16

# Adding 2017-2020 (Kraatz farm records, Zenodo 8087301)
Sensor offsets 2017-2020 (median S2 minus Landsat, added to Landsat):
      pairs   ndvi   ndti   ndmi
year                            
2017  106.0 -0.014 -0.024 -0.053
2018   48.0 -0.043 -0.026 -0.042
2019  140.0 -0.009 -0.016 -0.036
2020  220.0 -0.024 -0.008 -0.041

2017-2020 labeled field-years: 83 (Grain 64, Silage 19)
label                 Grain  Silage
year label_source                  
2017 planting record     18       6
2018 planting record     13       4
2019 harvest record      16       2
     planting record      5       1
2020 harvest record       4       5
     planting record      8       1
Min usable readings Aug-Nov: 11; field-years with fewer than 8: 0

## Same features, 2017-2020 vs 2021-2023 (medians; AUC with Silage = positive)
       feature  auc_2017-20  grain_2017-20  silage_2017-20  auc_2021-23  grain_2021-23  silage_2021-23
      drop_doy        0.219        258.000         243.500        0.362        257.750         248.000
 ndvi_pre_drop        0.642          0.658           0.754        0.849          0.682           0.819
     drop_size        0.851          0.188           0.430        0.964          0.180           0.512
    ndvi_sep15        0.117          0.598           0.350        0.036          0.538           0.278
ndti_post_drop        0.123          0.200           0.153        0.023          0.195           0.119
      regrowth        0.943         -0.042           0.355        0.987         -0.013           0.398
   t_below_0.3        0.130        289.767         270.921        0.008        302.325         255.060

## Train on 2017-2020 only, test on all 44 labeled field-years of 2021-2023

### Logistic regression, 5 hypothesis features
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[28, 4], [1, 11]]
Grain  precision 0.97  recall 0.88 | Silage precision 0.73  recall 0.92 | accuracy 0.89  balanced acc 0.90
per held-out year: 2021: 17/18 correct (4 silage); 2022: 6/7 correct (4 silage); 2023: 16/19 correct (4 silage)
test ROC-AUC 0.982
weights learned from 2017-2020 alone: ndti_post_drop -1.28, drop_size +1.02, drop_doy -0.89, ndvi_sep15 -0.72, ndvi_pre_drop -0.31

### Random forest, all features
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[30, 2], [1, 11]]
Grain  precision 0.97  recall 0.94 | Silage precision 0.85  recall 0.92 | accuracy 0.93  balanced acc 0.93
per held-out year: 2021: 16/18 correct (4 silage); 2022: 7/7 correct (4 silage); 2023: 18/19 correct (4 silage)
test ROC-AUC 0.948

### Logistic regression, strict: trained only on the 53 2017-2020 field-years whose fields never appear in 2021-2023
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[29, 3], [1, 11]]
Grain  precision 0.97  recall 0.91 | Silage precision 0.79  recall 0.92 | accuracy 0.91  balanced acc 0.91
per held-out year: 2021: 17/18 correct (4 silage); 2022: 7/7 correct (4 silage); 2023: 16/19 correct (4 silage)
test ROC-AUC 0.982

### Single rule chosen on 2017-2020: ndti_sep15 < 0.225 (train bal-acc 0.91)
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[24, 8], [1, 11]]
Grain  precision 0.96  recall 0.75 | Silage precision 0.58  recall 0.92 | accuracy 0.80  balanced acc 0.83
per held-out year: 2021: 13/18 correct (4 silage); 2022: 7/7 correct (4 silage); 2023: 15/19 correct (4 silage)

## Leave one year out over all seven years (logistic regression, 5 features)

### Train on six years, test on the seventh
confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): [[88, 8], [6, 25]]
Grain  precision 0.94  recall 0.92 | Silage precision 0.76  recall 0.81 | accuracy 0.89  balanced acc 0.86
per held-out year: 2017: 23/24 correct (6 silage); 2018: 14/17 correct (4 silage); 2019: 19/24 correct (3 silage); 2020: 17/18 correct (6 silage); 2021: 17/18 correct (4 silage); 2022: 6/7 correct (4 silage); 2023: 17/19 correct (4 silage)
held-out ROC-AUC 0.941
accuracy by label source: evaluation gpkg: 40/44; harvest record: 23/27; planting record: 50/56

Field-years wrong when their year is held out:
 year         farm field  label    label_source  p_silage  drop_doy  ndvi_pre_drop  drop_size  ndvi_sep15  ndti_post_drop  n_obs_aug_nov
 2017  CentralFarm  1-15 Silage planting record     0.497     241.0          0.769      0.416       0.582           0.186             23
 2018  CentralFarm 1-14B Silage planting record     0.351     247.5          0.538      0.234       0.332           0.177             22
 2018  CentralFarm 1-14D Silage planting record     0.377     247.5          0.587      0.222       0.367           0.173             22
 2018  CentralFarm  4-25 Silage planting record     0.105     243.5          0.559      0.209       0.343           0.217             13
 2019  CentralFarm 1-14B  Grain  harvest record     0.610     242.5          0.585      0.225       0.333           0.199             26
 2019  CentralFarm 1-14D  Grain planting record     0.632     242.5          0.630      0.238       0.347           0.191             26
 2019  CentralFarm 2-32A  Grain  harvest record     0.639     242.5          0.646      0.194       0.284           0.189             26
 2019     EastFarm  5-2A  Grain  harvest record     0.513     233.5          0.631      0.146       0.393           0.182             13
 2019  LinkageFarm  EB-3  Grain  harvest record     0.663     242.5          0.601      0.183       0.334           0.174             28
 2020          APU  1-98 Silage planting record     0.057     270.0          0.535      0.187       0.680           0.162             26
 2021 Central Farm   4-7 Silage evaluation gpkg     0.301     252.5          0.678      0.201       0.486           0.175             14
 2022 Central Farm 2-32B  Grain evaluation gpkg     0.524     248.0          0.826      0.368       0.362           0.219             29
 2023    East Farm  5-21  Grain evaluation gpkg     0.595     247.5          0.542      0.177       0.432           0.154             20
 2023    East Farm  5-40  Grain evaluation gpkg     0.751     247.5          0.666      0.260       0.384           0.153             20