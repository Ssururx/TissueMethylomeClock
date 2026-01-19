# Step 7 – Pathway Enrichment Analysis

In this step, I performed functional enrichment analysis to discover the biological "meaning" behind the age-associated genes identified in Step 6. By mapping these genes to known biological pathways, I can see exactly which cellular processes are driving the aging signal in each tissue.

---

###  Biological Pathway Discovery
Using **g:Profiler**, I analyzed 6 distinct gene sets against major biological databases (GO, KEGG, Reactome). This analysis reveals the functional "fingerprint" of epigenetic aging.

| Metric | Brain Discovery | Blood Discovery |
| :--- | :--- | :--- |
| **Total Significant Pathways** | 37 Pathways | 128 Pathways |
| **Novel Signal Contribution** | ~98% driven by novel genes | ~90% driven by novel genes |
| **Primary Biological Theme** | Neural/Synaptic Signaling | Immune/Inflammatory Response |

---

### Tissue-Specific Aging Mechanisms

Analysis of tissue-specific CpGs reveals distinct biological processes driving aging in different tissues:

* **Brain:** Enrichment for **synaptic development**, **neuronal signaling**, and **cell junction communication** suggests that epigenetic aging in the brain primarily involves changes in neuronal connectivity and communication.
* **Blood:** Enrichment for **plasma membrane adhesion**, **calcium ion binding**, and **multicellular development** highlights shifts in immune function and inflammatory signaling characteristic of blood aging.



---

###  Key Visualization(s)
* **Enrichment Heatmap:** A comparative visualization showing the strength of association ($-log_{10}$ adjusted p-value) for top pathways across all gene sets.
![Pathway Enrichment Heatmap](figures/pathway_enrichment_heatmap.jpeg)


---
### Biological Interpretation of Novel Genes

Over **90%** of the genes in our models are not found in the Horvath or Hannum clocks. In the brain, these genes highlight synaptic regulators and neural signaling pathways, suggesting tissue-specific mechanisms of aging overlooked by previous epigenetic clocks.

---

###  Key Outputs Created
* `unique_brain_pathways.csv` & `unique_blood_pathways.csv`: The definitive list of biological processes specific to each tissue.
* `novel_brain_enrichment_top50.csv`: The top 50 functional insights discovered from previously overlooked brain genes.
* `pathway_enrichment_heatmap_data.csv`: A machine-readable matrix of all biological pathway strengths.

The biological drivers of aging now have been identified and validated, this allows us to move on to **Step 8: Statistical Validation**.
