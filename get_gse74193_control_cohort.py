"""
Goal: Extract a set of 335 healthy control samples from the GSE74193 series matrix.

Purpose: To identify control samples, applies quality filters, removes
duplicate brain donors, and outputs a final curated cohort for downstream
analysis.
"""

import gzip
import pandas as pd
import re

def main():
    input_file = "GSE74193_series_matrix.txt.gz"

    print("Reading compressed series matrix file...")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        file_lines = f.readlines()

    # Initialize containers for metadata
    sample_titles = []
    diagnosis_list = []
    brnum_list = []
    qc_status_list = []

    print("Parsing sample metadata headers...")
    for line in file_lines:
        stripped_line = line.strip()

        if stripped_line.startswith('!Sample_title'):
            # Extract all sample titles
            parts = stripped_line.split('\t')
            sample_titles = [p.strip('"') for p in parts[1:]]

        elif '!Sample_characteristics_ch1' in stripped_line and 'group:' in stripped_line:
            # Extract diagnosis: Control or Schizo
            parts = stripped_line.split('\t')
            for entry in parts[1:]:
                clean_entry = entry.strip('"')
                if 'Control' in clean_entry:
                    diagnosis_list.append('Control')
                elif 'Schizo' in clean_entry:
                    diagnosis_list.append('Schizo')
                else:
                    diagnosis_list.append('Unknown')

        elif '!Sample_characteristics_ch1' in stripped_line and 'brnum' in clean_entry.lower():
            # Extract brain donor identifier
            parts = stripped_line.split('\t')
            for entry in parts[1:]:
                donor_match = re.search(r'Br(\d+)', entry)
                if donor_match:
                    brnum_list.append(f"Br{donor_match.group(1)}")
                else:
                    brnum_list.append('')

        elif '!Sample_characteristics_ch1' in stripped_line and 'bestqc' in clean_entry.lower():
            # Extract QC flag
            parts = stripped_line.split('\t')
            for entry in parts[1:]:
                if 'TRUE' in entry.upper():
                    qc_status_list.append('TRUE')
                elif 'FALSE' in entry.upper():
                    qc_status_list.append('FALSE')
                else:
                    qc_status_list.append('')

    print(f"Found metadata for {len(sample_titles)} total samples")

    # Build initial dataframe
    sample_data = pd.DataFrame({
        'sample_title': sample_titles,
        'diagnosis': diagnosis_list[:len(sample_titles)],
        'brain_number': brnum_list[:len(sample_titles)],
        'qc_flag': qc_status_list[:len(sample_titles)]
    })

    # Extract numeric sample identifier for sorting
    sample_data['sample_id'] = sample_data['sample_title'].str.extract(r'Sample(\d+)_').astype(int)

    control_count = (sample_data['diagnosis'] == 'Control').sum()
    schizo_count = (sample_data['diagnosis'] == 'Schizo').sum()

    print(f"Diagnosis breakdown: {control_count} control, {schizo_count} schizophrenia")

    # Apply sequential filters
    print("\nApplying filters...")

    # Keep only control samples
    control_samples = sample_data[sample_data['diagnosis'] == 'Control'].copy()
    print(f"  control samples: {len(control_samples)}")

    # Keep only samples passing QC
    qc_passed_samples = control_samples[control_samples['qc_flag'] == 'TRUE'].copy()
    print(f"  samples with qc_flag = TRUE: {len(qc_passed_samples)}")

    # Remove duplicate brain donors (keep first occurrence)
    unique_donor_samples = qc_passed_samples.drop_duplicates(subset='brain_number', keep='first')
    print(f"  unique brain donors: {len(unique_donor_samples)}")

    # Sort by sample ID and select first 335 for final cohort
    final_cohort = unique_donor_samples.sort_values('sample_id').head(335)

    cohort_range = f"{final_cohort['sample_id'].min()} to {final_cohort['sample_id'].max()}"
    print(f"\nFinal cohort contains {len(final_cohort)} samples (IDs {cohort_range})")

    # Write output files
    print("\nWriting output files...")
    final_cohort.to_csv('GSE74193_EXACT_335_CONTROLS_FINAL.csv', index=False)
    unique_donor_samples.to_csv('GSE74193_ALL_UNIQUE_CONTROLS_FINAL.csv', index=False)

    print("Completed.")
    print(f"  Primary output: GSE74193_EXACT_335_CONTROLS_FINAL.csv")
    print(f"  Reference file: GSE74193_ALL_UNIQUE_CONTROLS_FINAL.csv")

if __name__ == '__main__':
    main()
