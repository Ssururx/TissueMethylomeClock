# Step 3 – Feature Discovery

In this step, I identify the specific CpG sites on the DNA that are most strongly associated with aging. Rather than analyzing the entire genome, I narrow the focus to the top 500 markers for each tissue to ensure high model accuracy and biological relevance.

### Research Summary
I used correlation analysis and a custom Tissue Specificity Index (TSI) to evaluate how aging markers behave across different tissues.

* **Brain Analysis:** Identified 38,707 significant aging sites. The strongest site achieved a correlation of 0.936 with chronological age.
* **Blood Analysis:** Identified 1,450 significant aging sites. The strongest site achieved a correlation of -0.740 with chronological age.
* **Comparison with Established Clocks:** I compared my findings to the Horvath and Hannum clocks. I identified 972 unique CpGs, meaning these markers are not currently utilized in those established epigenetic clocks.

---

### Tissue Specificity Index (TSI) Analysis
The TSI is a mathematical tool I used to determine if an aging signal is "Universal" (happening everywhere) or "Tissue-Locked" (happening only in one tissue).

**The Formula:**
$$TSI = 1 - \frac{|r_{brain} - r_{blood}|}{\max(|r_{brain}|, |r_{blood}|)}$$

**How it works:**
The formula compares the correlation strength ($r$) of a specific CpG in the brain versus the blood. 
* If the correlations are almost identical, the numerator becomes small, and the TSI approaches 0 (Universal).
* If the CpG has a strong correlation in one tissue but almost zero in the other, the TSI approaches 1 (Tissue-Specific).

I applied this formula to the **250 CpGs** shared between the brain and blood datasets. Even though these 250 sites exist in both tissues, they do not always age the same way.

| Category | Count | Biological Meaning |
| :--- | :--- | :--- |
| **Universal** | 90 CpGs | Core aging signals conserved across both tissues (TSI < 0.3). |
| **Tissue-Locked** | 29 CpGs | Aging signals unique to the biology of one specific tissue (TSI > 0.8). |
| **Intermediate** | 131 CpGs | Signals that show varying degrees of tissue-specific modulation. |

---

### Shared vs. Unique Markers
While I analyzed 50,000 CpGs in the brain and 1,670 in the blood, only 250 sites were common to both datasets. 
* **Shared (250 CpGs):** These are the only sites that allow for a direct comparison of aging rates between brain and blood.
* **Unique (Not Shared):** The vast majority of markers are unique to one dataset. This is because different tissues and different laboratory platforms often capture different parts of the methylome. Focusing on these unique markers allows for a more specialized and precise "Tissue-Specific Clock."

---

### Key Visualizations
#### 1. Correlation Scatterplots
These plots demonstrate the relationship between DNA methylation levels and chronological age for the top-ranked markers in both tissues.
![Brain Top CpGs](figures/Brain_top_cpgs_scatter.jpeg)
![Blood Top CpGs](figures/Blood_top_cpgs_scatter.jpeg)


#### 2. TSI Distribution
This graph shows how the 250 shared CpGs are distributed across the specificity spectrum, highlighting the difference between systemic and localized aging.
![Tissue Specificity](figures/tissue_specificity_analysis.jpeg)

---


### Data Outputs
* **top_500_brain_cpgs.csv**: The highest-performing markers for brain age prediction.
* **top_500_blood_cpgs.csv**: The highest-performing markers for blood age prediction.
* **universal_cpgs.csv**: 90 markers identified as potential candidates for a pan-tissue aging model.

Now that feature discovery is complete, we can now move on to **Step 4: Epigenetic Clock Training**.
