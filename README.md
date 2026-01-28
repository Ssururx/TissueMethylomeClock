# TissueMethylome Clock (TMC): Multi-Aging Tissue Analysis

## Project Summary
This project develops and analyzes tissue-specific epigenetic clocks using DNA methylation data from 1,657 human brain and blood samples. We built high-accuracy age predictors that reveal fundamental differences in how aging manifests across tissues, identifying unique biomarkers not captured by established epigenetic clocks. For more insight and detail on steps, outcomes, visualizations, work-in-progress analyses, and related data, see the README files in each project subfolder.

## Project Folder Structure
The folder structure below shows scripts, figures, and documentation for each step of the project:

```plaintext
01_data_acquisition_and_preprocessing
├── README.md
├── extract_gse74193_controls.py
├── extract_gse74193_methylation.py
├── fetch_gse19711_data.py
├── fetch_gse40279_data.py
├── fetch_gse41037_data.py
├── inspect_gse19711_format.py
└── merge_blood_gse_datasets.py

02_data_loading_qc_exploration
├── README.md
├── figures
│   ├── age_distributions.jpeg
│   └── pca_analysis.jpeg
└── brain_blood.py

03_feature_characterization
├── README.md
├── figures
│   ├── Blood_top_cpgs_scatter.jpeg
│   ├── Brain_top_cpgs_scatter.jpeg
│   └── tissue_specificity_analysis.jpeg
└── feature_characterization_tsi.py

04_clock_training
├── README.md
├── figures
│   ├── blood_test_predictions.jpeg
│   ├── brain_feature_stability_analysis.jpeg
│   └── brain_test_predictions.jpeg
├── blood_clock.py
└── brain_clock.py

05_age_acceleration
├── README.md
├── figures
│   ├── age_acceleration_distributions.jpeg
│   ├── blood_nonlinear_trajectories.jpeg
│   └── brain_nonlinear_trajectories.jpeg
└── age_acceleration_analysis.py

06_cpg_to_gene_annotation
├── README.md
├── figures
│   └── gene_overlap_venn.jpeg
└── cpg_to_gene.py

07_pathway_enrichment
├── README.md
├── figures
│   └── pathway_enrichment_heatmap.jpeg
└── enrichment_analysis.py

08_statistical_validation
├── README.md
├── figures
│   ├── blood_bootstraps_distributions.jpeg
│   ├── blood_predictions.jpeg
│   ├── brain_bootstraps_distributions.jpeg
│   └── brain_predictions.jpeg
└── clock_validation.py

09_comprehensive_annotation
├── README.md
└── 09_comprehensive_annotation.py

10_dmr_comparative_analysis
├── README.md
├── figures
│   ├── aging_CpG_genomic_context_blood.jpeg
│   └── aging_CpG_genomic_context_brain.jpeg
└── dmr_comparison.py

11_latent_aging_profiles
├── README.md
├── figures
│   ├── blood-genomic-dashboard.jpg
│   └── brain-genomic-dashboard.jpg
└── latent_aging_profile_analysis.py

LICENSE
README.md
```

## Key Results at a Glance

**Model Performance:**

* **Brain Clock:** MAE = 2.51 years, R² = 0.971 (174 CpGs)
* **Blood Clock:** MAE = 5.46 years, R² = 0.859 (500 CpGs)

**Dataset Composition:**

* **Brain:** 335 samples, GSE74193, age range: prenatal to 75.6 years
* **Blood:** 1,322 samples (GSE19711, GSE40279, GSE41037 merged), age range: 16-101 years
* **Total:** 1,657 healthy control samples, Illumina 450K platform

## Analytical Pipeline Summary

### **Step 1: Data Acquisition & Preprocessing**

Downloaded and integrated four GEO datasets. Standardized age to decimal years, filtered for healthy controls only, and merged three blood studies into a unified cohort of 1,322 samples. *Key insight: Established a clean, well-annotated foundation from heterogeneous sources.*

### **Step 2: Quality Control & Exploration**

Implemented rigorous QC: removed samples with >5% missing values, verified beta-value ranges (0-1), and filtered 2 low-quality blood samples. PCA confirmed clear tissue separation. *Key insight: Data integrity validated with minimal missing values (0.09% blood, 0% brain).*

### **Step 3: Feature Discovery & Tissue Specificity**

Identified top age-associated CpGs: **38,707 significant in brain** (top r=0.936), **1,450 in blood** (top r=0.740). Developed Tissue Specificity Index (TSI) analyzing 250 shared CpGs: **60% tissue-locked**, **14% universal**, **26% intermediate**. *Key insight: Aging is predominantly tissue-specific, with minimal conserved signals.*

### **Step 4: Model Training & Optimization**

Built ElasticNet + Bagging ensemble models. Brain model uses **174 stable CpGs** achieving r=0.985; blood model uses **500 CpGs** achieving r=0.927. *Key insight: Brain's superior performance stems from stronger individual CpG-age correlations allowing more compact feature sets.*

### **Step 5: Age Acceleration & Nonlinear Dynamics**

Calculated epigenetic age acceleration (predicted - chronological age). Identified **slow, normal, and fast agers** via clustering. Nonlinear analysis revealed **accelerated epigenetic aging during development**, with inflection points around age 13. *Key insight: Epigenetic aging rate varies across lifespan, fastest during early development.*

### **Step 6: Genomic Annotation**

Mapped top 500 CpGs to genes: **363 unique brain genes**, **523 unique blood genes**, with only **13 genes shared**. **98.6% of brain genes and 96.6% of blood genes are unique** compared to Horvath/Hannum clocks. *Key insight: Minimal genetic overlap confirms tissue-specific aging programs.*

### **Step 7: Pathway Analysis**

Brain genes enriched for **synaptic signaling and neuronal development**. Blood genes enriched for **immune response and inflammatory pathways**. *Key insight: Aging engages tissue-core functions: neural communication in brain, immune regulation in blood.*

### **Step 8: Statistical Validation**

Passed rigorous bootstrap (500 iterations) and permutation testing (500 shuffles). Final correlations: **brain r=0.987, blood r=0.933** (p < 1e-100). *Key insight: Models capture genuine biological signal, not data artifacts.*

### **Step 9: Comprehensive Genomic Context**

Extracted detailed genomic features: chromosomal location, CpG island status, gene region. **Chromosomes 1 and 19** show highest density of aging markers. Created streamlined datasets for top 500 CpGs. *Key insight: Aging-associated CpGs distribute non-randomly across genome.*

### **Step 10: DMR & Comparative Analysis**

Identified **472 DMRs in blood, 350 in brain**. Found fundamental difference: **blood aging targets promoter regions** (24.5% in TSS1500), while **brain aging targets gene bodies** (32.6%). *Key insight: Divergent regulatory logic—transcriptional control in blood vs. splicing/elongation in brain.*

### **Step 11: Latent Cellular Architecture**

Applied NMF decomposition to extract latent cellular profiles: **5 profiles in blood** (4 age-associated), **6 profiles in brain** (3 age-associated). *Key insight: Bulk tissue aging signal decomposes into multiple cellular trajectories.* **Current WIP: Biological annotation of profiles to specific cell types.**

## Major Biological Insights

1. **Tissue Specificity Dominates:** Only 14% of aging signals are universal across tissues
2. **Unique Biomarker Identification:** >96% of identified genes are not in established clocks
3. **Divergent Regulatory Mechanisms:** Promoter-focused in blood vs. gene body-focused in brain
4. **Nonlinear Aging Trajectory:** Epigenetic change accelerates during development
5. **Functional Specialization:** Aging engages tissue-core pathways (neural in brain, immune in blood)

## Immediate Next Steps & Future Directions

### **Current Work in Progress:**

* **Biological annotation of Latent Cellular Profiles** using cell-type-specific methylation databases (PanglaoDB, Allen Brain Atlas)
* **Causal mediation analysis** to quantify clock signal attribution: cellular composition vs. intracellular drift
* **Gender analysis** pending metadata extraction from GSE40279

### **Planned Future Investigations:**

**3. Integrated Multi-Omics and Phenotypic Correlation:**

* **Longitudinal trajectory modeling** using available blood DNAm data to predict later-life brain epigenetic aging
* **Clinical correlation analysis** with Lothian Birth Cohort 1936 data: cognitive decline, lifestyle factors, inflammatory markers, neuropathology
* **Differential association testing** to identify factors with tissue-specific aging effects
* **Blood-brain barrier investigation** of age-related DMR genes for neuroinflammation links

**4. Causality and Predictive Utility:**

* **Causal inference modeling** using Mendelian randomization
* **Predictive modeling** for cognitive decline and neurodegenerative pathology
* **Utility demonstration** comparing predictive power against general epigenetic clocks

## Technical Implementation Highlights

* **Machine Learning:** ElasticNet regression with Bagging ensemble (30-50 estimators)
* **Statistical Validation:** Bootstrap resampling, permutation testing, 5-fold cross-validation
* **Bioinformatics:** g:Profiler pathway analysis, Illumina HM450 annotation, custom TSI framework
* **Visualization:** Comprehensive dashboards for LAP analysis, genomic context distributions

## Project Significance

This work advances epigenetic aging research by: (1) developing high-accuracy tissue-specific clocks, (2) revealing fundamental tissue differences in aging mechanisms, (3) identifying unique biomarkers beyond established clocks, and (4) providing a framework for deconstructing cellular contributions to aging. The findings establish a foundation for targeted interventions and precision aging assessment.

## References

1. GEO: [GSE74193](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74193) – Human brain methylation dataset
2. GEO: [GSE19711](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19711) – Human blood methylation dataset
3. GEO: [GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) – Human blood methylation dataset
4. GEO: [GSE41037](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41037) – Human blood methylation dataset
5. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*, 14, R115.
6. Hannum, G. et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell*, 49, 359–367
7. Lu, A. T., Quach, A., Wilson, J. G., Reiner, A. P., Aviv, A., Raj, K., ... & Horvath, S. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan. Aging (Albany NY), 11(2), 303–327. https://doi.org/10.18632/aging.101684.
8. Levine, M. E. et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. Aging (Albany NY), 10(4), 573–591.
9. Slieker, R. C. et al. (2018). Age-related accrual of methylomic variability is linked to fundamental biological pathways. Genome Biology, 19, 1–18.
10. Rakyan, V. K. et al. (2010). Human aging-associated DNA hypermethylation occurs preferentially at bivalent chromatin domains. Genome Research, 20(4), 434–439.
