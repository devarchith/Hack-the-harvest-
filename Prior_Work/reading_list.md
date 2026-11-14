# Prior Work and Reading List

This is a starting reading list for the UC Davis Hackathon challenge. It does
**not** identify a published, public field-level silage-versus-grain corn label
dataset. The most relevant work instead supports pieces of the inference:
harvest-date detection, corn phenology, synthetic-aperture radar (SAR), and
post-harvest residue signals.

Use the papers to motivate and test a method, not to claim that a signal proven
for harvest timing or residue uniquely identifies silage or grain. Validate any
feature on held-out fields, years, and regions.

## Included Local Papers

These PDFs are included in this folder for student access. They are not direct
silage-versus-grain classification studies, but provide relevant methods for
crop phenology, time-series analysis, and detecting management events.

1. **Gao, F., et al. (2020). _A within-season approach for detecting early
   growth stages in corn and soybean using high temporal and spatial resolution
   imagery._ Remote Sensing of Environment, 242, 111752.**
   [Local PDF](crop_emergence.RSE_2020_Gao.pdf) |
   [DOI: 10.1016/j.rse.2020.111752](https://doi.org/10.1016/j.rse.2020.111752)

   Provides a corn-specific phenology and time-series framework using Landsat
   and Sentinel-2. Adapt its event-detection ideas to late-season senescence and
   harvest timing, then validate them against the challenge labels.

2. **Gao, F., et al. (2023). _Near real-time detection of winter cover crop
   termination using harmonized Landsat and Sentinel-2 (HLS) to support
   ecosystem assessment._ Science of Remote Sensing, 7, 100073.**
   [Local PDF](winetr_cover_crop_termination.SRS_2023_Gao.pdf) |
   [DOI: 10.1016/j.srs.2022.100073](https://doi.org/10.1016/j.srs.2022.100073)

   Harvest and cover-crop termination are both abrupt management events. Use
   the event-detection approach as a hypothesis, not a ready-made corn harvest
   classifier, because the crop system and target differ.

3. **Gao, F., et al. (2026). _Mapping winter cover crops using remote sensing
   of crop phenology._ Remote Sensing of Environment, 342, 115480.**
   [Local PDF](winter_cover_crop_presence.RSE_2026_Gao.pdf) |
   [DOI: 10.1016/j.rse.2026.115480](https://doi.org/10.1016/j.rse.2026.115480)

   Demonstrates phenology-based mapping with harmonized Landsat and Sentinel-2
   observations. It is useful for feature construction and validation when a
   class is defined by seasonal behavior.

## Most Directly Relevant

1. **Xu, J., Meng, J., and Quackenbush, L. J. (2019). _Use of remote sensing to
   predict the optimal harvest date of corn._ Field Crops Research, 236, 1-13.**
   [DOI: 10.1016/j.fcr.2019.03.003](https://doi.org/10.1016/j.fcr.2019.03.003)

   A corn-specific study connecting remotely sensed observations to harvest
   timing. It is useful for feature ideas involving crop maturity, chlorophyll,
   and time series, but its target is optimal harvest timing rather than harvest
   use (silage or grain).

2. **Liu, Y., Diao, C., Yang, Z., Mei, W., and Guo, T. (2025). _A novel
   Normalized Harvest Phenology Index (NHPI) for corn and soybean harvesting
   date detection using Landsat and Sentinel-2 imagery on Google Earth Engine._
   Remote Sensing of Environment, 331, 115016.**
   [DOI: 10.1016/j.rse.2025.115016](https://doi.org/10.1016/j.rse.2025.115016)

   A recent optical time-series approach to harvest-date detection. It is a
   strong reference for testing whether an estimated harvest timing can separate
   earlier silage-like behavior from later grain-like behavior. Teams should
   test transferability to their geography and year rather than assuming the
   index will generalize unchanged.

3. **Khabbazan, S., Hosseini, M., Becker-Reshef, I., McNairn, H., and Oliinyk,
   O. (2026). _Harvest date estimation over corn and sunflower fields using
   Sentinel-1 and RADARSAT constellation mission data and machine learning
   algorithms._ International Journal of Remote Sensing, 47(5), 1996-2040.**
   [DOI: 10.1080/01431161.2026.2612904](https://doi.org/10.1080/01431161.2026.2612904)

   A corn harvest-date study using radar. SAR can be valuable where optical
   imagery is cloud-limited and provides a useful alternative to an NDVI-only
   workflow.

## SAR and Harvest Detection Methods

4. **Amherdt, S., Di Leo, N. C., Balbarani, S., Pereira, A., Cornero, C., and
   Pacino, M. C. (2021). _Exploiting Sentinel-1 data time-series for crop
   classification and harvest date detection._ International Journal of Remote
   Sensing, 42(19), 7313-7331.**
   [DOI: 10.1080/01431161.2021.1957176](https://doi.org/10.1080/01431161.2021.1957176)

   Demonstrates a Sentinel-1 time-series route to crop classification and
   harvest-date detection. It is not corn-silage-vs-grain work, but helps teams
   explore radar features and complementary optical-plus-radar models.

5. **Shang, J., et al. (2020). _Detection of crop seeding and harvest through
   analysis of time-series Sentinel-1 interferometric SAR data._ Remote Sensing,
   12(10), 1551.**
   [DOI: 10.3390/rs12101551](https://doi.org/10.3390/rs12101551)

   Useful for teams considering interferometric coherence as a management-event
   signal. Check data availability, preprocessing requirements, and whether the
   target field geometry supports reliable coherence estimates.

## Residue and Post-Harvest Surface Signals

6. **Jin, X., et al. (2015). _Estimation of maize residue cover using Landsat-8
   OLI image spectral information and textural features._ Remote Sensing, 7(11),
   14559-14575.**
   [DOI: 10.3390/rs71114559](https://doi.org/10.3390/rs71114559)

   Directly relevant to the hypothesis that grain harvest can leave more visible
   residue than silage harvest. It estimates residue cover, not harvest use, so
   residue should be treated as one feature among several.

7. **Quemada, M., et al. (2018). _Improved crop residue cover estimates obtained
   by coupling spectral indices for residue and moisture._ Remote Sensing of
   Environment, 206, 33-44.**
   [DOI: 10.1016/j.rse.2017.12.012](https://doi.org/10.1016/j.rse.2017.12.012)

   Shows why residue and moisture can be confounded in spectral observations.
   This is particularly important for a fall-harvest classifier: a bare or wet
   field should not be assigned a harvest type from one image alone.

8. **Zheng, B., Myint, S. W., Thenkabail, P. S., and Aggarwal, R. M. (2012).
   _Remote sensing of crop residue cover using multi-temporal Landsat imagery._
   Remote Sensing of Environment, 117, 177-183.**
   [DOI: 10.1016/j.rse.2011.09.016](https://doi.org/10.1016/j.rse.2011.09.016)

   Supports multi-date rather than single-date residue analysis. The emphasis on
   repeated observations is aligned with the challenge's timing hypothesis.

## Corn Phenology and Field-Scale Time Series

9. **Sakamoto, T., et al. (2010). _A two-step filtering approach for detecting
   maize and soybean phenology with time-series MODIS data._ Remote Sensing of
   Environment, 114(10), 2146-2159.**
   [DOI: 10.1016/j.rse.2010.04.019](https://doi.org/10.1016/j.rse.2010.04.019)

   A foundational maize phenology reference. MODIS resolution is typically too
   coarse for individual fields, but the time-series thinking and event
   extraction methods are useful.

10. **Shen, Y., et al. (2022). _Mapping corn and soybean phenometrics at field
    scales over the United States Corn Belt by fusing time series of Landsat 8
    and Sentinel-2 data with VIIRS data._ ISPRS Journal of Photogrammetry and
    Remote Sensing, 186, 55-69.**
    [DOI: 10.1016/j.isprsjprs.2022.01.023](https://doi.org/10.1016/j.isprsjprs.2022.01.023)

    A U.S. field-scale phenology reference that is useful for constructing
    denser optical time series and deriving seasonal features.

## Recommendations for Using This List

- Start with the direct-harvest papers to define a timing baseline.
- Add residue or SAR features when they can be justified and validated.
- Cite the original paper for each adopted feature, and verify that its sensor,
  geography, crop system, target variable, and validation design apply to your
  use case.
- Separate **published evidence**, **your feature hypothesis**, and **your
  observed model performance** in your final presentation.

All links above resolve through DOI. Access to the publisher-hosted full text may
vary; use the DOI metadata, institutional access, author copies, or legitimate
open-access versions when available.
