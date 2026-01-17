# Epigenetic Aging Project: Step 1 – Data Acquisition & Preprocessing

This is the first step of my project to build an epigenetic clock. In this phase, I download, clean, and organize DNA methylation data from the Gene Expression Omnibus (GEO). 

###  Dataset Overview
I am using data from four different studies as of now to look at how the human brain and blood age over a lifetime.

| Tissue | Source (GEO ID) | Samples (N) | Age Range |
| :--- | :--- | :--- | :--- |
| **Brain** | GSE74193 | 335 | -0.5 to 75.6 years |
| **Blood** | GSE19711, GSE40279, GSE41037 | 1,322 | 16.0 to 101.0 years |
| **Total** | | **1,657** | |

---
### Script Explanations

The workflow is organized into separate scripts to keep the data clear and manageable, and especially because of the vast size of the files.
* **Fetching (`fetch_*.py`)**  
  * These scripts download the raw data and metadata from GEO and save them in a usable format.  
  * I also check the files to ensure everything is downloaded and structured correctly.

* **Extraction (`extract_*.py`)**  
  * These scripts filter the datasets to retain only the desired samples.  
  * For example, in the Brain dataset (GSE74193), I select only the healthy control samples.

* **Inspecting & Merging (`inspect_*.py` & `merge_*.py`)**  
  * **Inspect:** These scripts allow viewing the formatting of files without loading the entire dataset.  
  * **Merge:** For the Blood datasets, this script combines three studies into one unified dataset (1,322 samples) for analysis.

---



