# Epigenetic Aging Project: Step 5 – Age Acceleration and Nonlinear Dynamics

In this step, I move beyond simple age prediction to analyze Age Acceleration—the difference between a person's biological age and their actual chronological age. This allows for the identification of specific aging phenotypes and the study of how the pace/rate of aging changes across different life stages.

###  Age Acceleration and Aging Phenotypes
By calculating the residuals (Predicted Age - Chronological Age), I used K-Means clustering to categorize samples into three distinct groups:
* **Slow Agers:** Individuals whose biological age is lower than their chronological age.
* **Normal Agers:** Individuals whose biological age matches their chronological age.
* **Fast Agers:** Individuals showing signs of accelerated biological aging.

| Tissue | Mean Age Acceleration | Correlation (r) | Biological Interpretation |
| :--- | :--- | :--- | :--- |
| **Brain** | 4.20 ± 17.53 years | 0.981 | High precision across the lifespan. |
| **Blood** | 2.58 ± 17.54 years | 0.931 | Strong systemic aging signal. |

---

###  Nonlinear Aging Trajectories
Aging does not always happen at a constant speed. I used advanced modeling (LOWESS smoothing and Piecewise Regression) to map the rate of aging:
* **Inflection Points:** Identified key ages (e.g., 13.1 and 13.7 years in the brain) where the pace of epigenetic aging shifts.
* **Life-Stage Dynamics:** The analysis revealed that epigenetic aging is generally decelerated in later life stages, suggesting the "biological clock" ticks fastest during early development and maturation.
  
### Nonlinear Trajectory Performance
The LOWESS and piecewise models prioritize capturing localized biological shifts over global linear accuracy. This results in higher variance at the lifespan extremities, reflecting the inherent biological noise and rapid epigenetic transitions characteristic of early development and advanced age.




---

<!--  ###  Developmental Stratification
I analyzed how the clock performs across six different life stages. 
* **Peak Performance:** The Brain clock is most precise during the **Young Adult** stage (MAE = 6.07 years).
* **Developmental Insights:** Accuracy varies in the **Fetal/Neonatal** and **Older Adult** stages. This is a common finding in epigenetic research, as these periods involve rapid cellular changes (development) or increased biological noise (aging), which are more complex to model than stable adulthood. -->

### Key Visualizations 
### Age Acceleration Distributions
This plot shows the difference between predicted biological age and chronological age for all samples, illustrating the rate of biological aging across tissues in a single visualization.  
![Age Acceleration Distributions](figures/age_acceleration_distributions.jpeg)

### Nonlinear Aging Trajectories
This plot shows epigenetic age versus chronological age using LOWESS smoothing and piecewise regression. It captures the nonlinear dynamics of aging, showing faster biological aging during early development and more stable rates in adulthood, with subtle differences between brain and blood.

<table>
<tr>
<td align="center">
<img src="figures/brain_nonlinear_trajectories.jpeg" alt="Nonlinear Aging Trajectories: Brain" width="300">
</td>
<td align="center">
<img src="figures/blood_nonlinear_trajectories.jpeg" alt="Nonlinear Aging Trajectories: Blood" width="300">
</td>
</tr>
</table>



---

###  Key Outputs Created
* `brain_aging_patterns.csv` & `blood_aging_patterns.csv`: Classifications for Fast, Normal, and Slow agers.
* `brain_piecewise_regression.csv`: Statistical data on aging rate changes across life stages.
* `age_acceleration_distributions.png`: Visualization of biological age variance in the population.

### Future Goals
In the future, age acceleration from Step 5 could be correlated with available clinical metadata, such as gender. All current samples are healthy, and for GSE40279 656 samples (one of three blood datasets used as of now), gender information appears on the GSE page but could not be retrieved during data extraction, so this analysis is still a work in progress (WIP).


Since biological aging phenotypes has been identified, we then continue on to **Step 6: Genomic Mapping (CpG-to-Gene)**.
