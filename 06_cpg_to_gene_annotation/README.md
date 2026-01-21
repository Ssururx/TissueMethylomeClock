# Step 6 – Genomic Mapping and Novelty Analysis

In this step, I transitioned from mathematical coordinates (CpG probe IDs) to biological function. By mapping the top age-associated CpGs to their corresponding genes, I can identify the specific biological systems involved in aging for both the brain and blood.

---

###  CpG-to-Gene Mapping
Using the Illumina 450K annotation manifest, I mapped the top 500 CpGs from each tissue to official gene symbols. This step is essential for translating data points into a biological "story."

| Metric | Brain Analysis | Blood Analysis |
| :--- | :--- | :--- |
| **Mapped CpGs** | 390 / 500 (78.0%) | 494 / 500 (98.8%) |
| **Unique Genes** | 363 Genes | 523 Genes |
| **Tissue-Specific** | 350 Unique Genes | 510 Unique Genes |

**Shared Biological Signal:** Only **13 genes** were shared between the brain and blood datasets, further confirming that aging is a highly tissue-specific process at the genomic level.

---

###  Identification of Unique Aging Genes
A major goal of this project is to discover aging markers that existing models might have missed. I compared my gene lists against the two most famous epigenetic clocks: **Horvath (2013)** and **Hannum (2013)**.

* **Brain Unique Genes Identified:** 358 genes (**98.6%**) identified in my model are not found in the Horvath/Hannum clocks.
* **Blood Unique Genes Identified:** 505 genes (**96.6%**) are uniquemarkers not used in the original gold-standard clocks.
* **Significance:** These unique genes represent potential new targets for anti-aging research and suggest that my models are capturing novel biological signals overlooked by earlier research.

---

###  Key Visualization(s)
* **Gene Overlap Venn Diagram:** Highlights the minimal overlap (13 genes) between brain and blood, emphasizing tissue-specific aging.
<img src="figures/gene_overlap_venn.jpeg" alt="Gene Overlap Venn Diagram" width="500">



---

###  Key Outputs Created
* `novel_brain_genes.csv` & `novel_blood_genes.csv`: Lists of newly discovered genes for downstream analysis.
* `shared_genes.csv`: The 13 genes shared between brain and blood, representing a potential core framework of tissue-independent aging.
* `brain_cpg_gene_mapping.csv` & `blood_cpg_gene_mapping.csv`: Detailed tables connecting every CpG probe to its chromosome, position, and gene function.

> **Note:** Here, "novel" or "unique" refers to genes that are **not included in the Horvath (2013) or Hannum (2013) epigenetic clocks**. It does **not** imply that these genes were previously unknown to science—only that they represent aging signals **distinct from those captured by the existing clocks**.


Since the "biological translation" is complete, the project is then ready for **Step 7: Pathway Enrichment Analysis**.
