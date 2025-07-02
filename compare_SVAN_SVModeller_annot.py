import formats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.stats import spearmanr
from matplotlib import cm
import matplotlib.colors as mcolors

#load VCF object each and save to a variable to read each vcf file

VCF1=formats.VCF()
VCF1.read("/users/brodriguez/asanchez/SVModeller/output/Module2/30k_variants/VCF_Insertions_SVModeller.vcf")
VCF2=formats.VCF()
VCF2.read("/users/brodriguez/asanchez/SVAN/output/simulations/1000_per_variant/annot_INS/Simulated.vcf.INS.SVAN.vcf")

outDir="/users/brodriguez/asanchez/scripts/MEI_annotations/simulation_comp"
#function that extracts the chromosome and info from each vcf variant and saves it into a nested dictionary where variant ID is the key.
def vcf_2_dict(vcf):
    variant_info_dict = {}

    for variant in vcf.variants:
        variant_info_dict[variant.ID] = variant.info.copy()
        variant_info_dict[variant.ID]["CHROM"] = variant.chrom 

    return variant_info_dict       

## turn empty values into None
def normalize(value):
    if value in [None, "", ".", "NONE", "?", "NA"]:
        return None
    return value

## turn numerical values into integer, else turn into None
def safe_int(val):
    try:
        if val is None or str(val).strip().upper() == "NA":
            return None
        return int(float(val))
    except (ValueError, TypeError):
        return None
    
##Filter outliers based on the interquartile range

def filter_outliers(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)

    # Combine both lists for global mean and std
    combined = np.concatenate([list1, list2])
    global_mean = np.mean(combined)
    global_std = np.std(combined)

    # Compute shared bounds
    lower_bound = global_mean - 3 * global_std
    upper_bound = global_mean + 3 * global_std

    # Keep only values within bounds in both lists
    valid_indices = np.where(
        (list1 >= lower_bound) & (list1 <= upper_bound) &
        (list2 >= lower_bound) & (list2 <= upper_bound)
    )[0]

    filtered_list1 = list1[valid_indices]
    filtered_list2 = list2[valid_indices]

    return filtered_list1, filtered_list2

def compute_td_len_diff(val1_raw, val2_raw, treat_val2_na_as_none=False):
            """
            Computes the difference between two TD_LEN values with proper safety.

            Args:
                val1_raw: Raw value from dict1 (could be int, str, None)
                val2_raw: Raw value from dict2 (could be int, str, None, or 'NA')
                treat_val2_na_as_none: If True, treat 'NA' string as None (special case for 5PRIME)

            Returns:
                An integer difference, "NA" string, or None (for skipping).
            """
            val1 = safe_int(val1_raw)
            
            if treat_val2_na_as_none and val2_raw == "NA":
                val2 = None
            else:
                val2 = safe_int(val2_raw)

            if val1 is None and val2 == 0:
                return None  # skip this entry
            elif val1 is None or val2 is None:
                return "NA"
            else:
                return val1 - val2
def process_field(field_1_name, field_2_name, all_dict1, all_dict2, values_dict1, values_dict2,  dict1_entry, dict2_entry, combined, variant_id, family_counts):
    val1 = safe_int(dict1_entry.get(field_1_name))
    val2 = safe_int(dict2_entry.get(field_2_name))

    if val1 is not None and val1 != 0:
        all_dict1.append(val1)
    if val2 is not None and val2 != 0:
        all_dict2.append(val2)

    diff_key = f"{field_2_name}_DIFF"
    if val1 is not None and val2 is not None:
        combined[variant_id][diff_key] = val1 - val2
        values_dict1.append(val1)
        values_dict2.append(val2)
    else:
        combined[variant_id][diff_key] = "NA"
    family = dict1_entry.get("FAM_N")
    if family in {"Alu", "L1", "SVA"}:
        if family not in family_counts:
            family_counts[family] = {
                "val1_entry_count": 0,
                "val2_match_count": 0
            }
        if val1 is not None:
            family_counts[family]["val1_entry_count"] += 1
            if val2 is not None:
                family_counts[family]["val2_match_count"] += 1
##Plot length comparison of variant components after filtering outliers 
def plot_len_diff_no_outliers(list1, list2, title, label1="List 1", label2="List 2"):
    spearman_r, _ = spearmanr(list1, list2)
    filtered_list1, filtered_list2 = filter_outliers(list1, list2)
    min_value = min(np.min(filtered_list1), np.min(filtered_list2))
    max_value = max(np.max(filtered_list1), np.max(filtered_list2))
    plt.scatter(filtered_list1, filtered_list2)
    plt.xlim(0, max_value)
    plt.ylim(0,max_value)
    plt.title(f'{title}\nSpearman ρ = {spearman_r:.3f}')
    plt.xlabel(f'length values for {label1}')
    plt.ylabel(f'length values for {label2}')
    filename = title.replace(" ", "_").replace(":", "") + ".svg"
    filename_png = title.replace(" ", "_").replace(":", "") + ".png"
    plt.savefig(f'{outDir}/plots/filtered_{filename}', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/filtered_{filename_png}', bbox_inches='tight')
    plt.close()
    min_value = min(np.min(filtered_list1), np.min(filtered_list2))
    max_value = max(np.max(filtered_list1), np.max(filtered_list2))
    bins= np.linspace(min_value, max_value, 15)
    plt.hist([filtered_list1,filtered_list2], bins=bins, label=[label1, label2])
    plt.legend(loc='upper right')
    plt.title(f"{label1}/{label2} {title}")
    plt.savefig(f'{outDir}/plots/filtered_hist_{filename}', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/filtered_hist_{filename_png}', bbox_inches='tight')
    plt.close()

##Plot length comparison of variant components without filtering
def plot_len_diff(list1, list2, title, label1="List 1", label2="List 2"):
    axis_limit = max(max(list1), max(list2))
    spearman_r, _ = spearmanr(list1, list2)
    plt.scatter(list1, list2)
    plt.xlim(0, axis_limit)
    plt.ylim(0, axis_limit)
    plt.axis('square')
    plt.title(f'{title}\nSpearman ρ = {spearman_r:.3f}')
    plt.xlabel(f'length values for {label1}')
    plt.ylabel(f'length values for {label2}')
    filename = title.replace(" ", "_").replace(":", "") + ".svg"
    plt.savefig(f'{outDir}/plots/{filename}', bbox_inches='tight')
    plt.close()
    plt.hist([list1,list2], label=[label1, label2])
    plt.legend(loc='upper right')
    plt.title(f"{label1}/{label2} {title}")
    plt.savefig(f'{outDir}/plots/hist_{filename}', bbox_inches='tight')
    plt.close()

##Create horizontal bar plots for the top 10 conformations that takes the lists and title for the plot as input
def plot_top_conformations(conf_list, dataset_name="Dataset"):
    conf, count = zip(*conf_list)
    conf = list(conf)[::-1]
    count = list(count)[::-1]
    bars = plt.barh(conf, count)
    plt.title(f'{dataset_name}')
    plt.xlabel('Counts')
    plt.ylabel('Conformations')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(count) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}',
                 va='center')
    plt.savefig(f'{outDir}/plots/{dataset_name}.svg', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/{dataset_name}.png', bbox_inches='tight')
    plt.close()

def plot_sensitivity(true_total_dict, true_pos_dict, title):
    sensitivity_data = [ (entry, (true_pos_dict.get(entry, 0)/true_total_dict[entry]*100) if true_total_dict[entry] > 0 else 0) for entry in true_total_dict]
    sensitivity_data.sort(key=lambda x: x[1], reverse=True)
    sorted_categories, sorted_sensitivities = zip(*sensitivity_data)
    norm = plt.Normalize(min(sorted_sensitivities), max(sorted_sensitivities))
    colors = cm.viridis(norm(sorted_sensitivities))
    bars = plt.barh(sorted_categories, sorted_sensitivities, color=colors)
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width - 1,  # Position: inside the bar (left-aligned to the bar's right edge)
            bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}%',
            ha='right',  # Right alignment within the text box
            va='center',
            color='white' if width > 50 else 'black'  # Auto-contrast text color
        )
    plt.xlim(0, 100)
    plt.xlabel('Sensitivity (%)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.title(f'Sensitivity for {title}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{outDir}/plots/sensitivity_{title}.svg', bbox_inches='tight')
    plt.close()

def plot_precision(total_positive_dict, false_positive_dict, title):
    precision_data = {
        category: (1 - (false_positive_dict.get(category, 0) / total_positives)) * 100 
        if total_positives > 0 else 0
        for category, total_positives in total_positive_dict.items()
    }
    sorted_categories = sorted(precision_data.keys(), key=lambda x: precision_data[x])
    sorted_precision = [precision_data[cat] for cat in sorted_categories]
    bars = plt.barh(sorted_categories, sorted_precision)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 ha='left', va='center')
    plt.xlim(0, 100)
    plt.xlabel('Precision (%)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.title(f'Precision for {title}')
    plt.savefig(f'{outDir}/plots/precision_{title}.svg', bbox_inches='tight')
    plt.close()


def plot_metrics(true_total_dict, true_pos_dict, total_positive_dict, title):
    categories = set(true_total_dict.keys()) | set(total_positive_dict.keys())

    data = []
    for category in categories:
        tp = true_pos_dict.get(category, 0)
        total_pred = total_positive_dict.get(category, 0)
        total_true = true_total_dict.get(category, 0)

        sensitivity = (tp / total_true) * 100 if total_true > 0 else 0
        precision = (tp / total_pred) * 100 if total_pred > 0 else 0
        if precision + sensitivity > 0:
            f1 = (2 * (precision / 100) * (sensitivity / 100)) / ((precision / 100) + (sensitivity / 100))
        else:
            f1 = 0
        data.append((category, sensitivity, precision, f1))

    # Sort by F1 descending
    data.sort(key=lambda x: x[3], reverse=True)
    categories, sensitivities, precisions, f1_scores = zip(*data)
    y_positions = range(len(categories))
    bar_width = 0.35

    # === Plot 1: F1 Score ===
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    colors = cm.YlOrRd(norm(f1_scores))

    plt.figure(figsize=(10, 0.5 * len(categories) + 1))
    bars = plt.barh(y_positions, f1_scores, color=colors)

    for y, f in zip(y_positions, f1_scores):
        plt.text(f + 0.01, y, f'{f:.2f}', va='center', ha='left', fontsize=8)

    plt.yticks(y_positions, categories)
    plt.gca().invert_yaxis()  # Highest F1 score on top
    plt.xlim(0, 1)
    plt.xlabel('F1 Score')
    plt.title(f'F1 Score per Category: {title}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{outDir}/plots/f1score_{title}.svg', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/f1score_{title}.png', bbox_inches='tight')

    plt.close()

    # === Plot 2: Precision & Sensitivity (same order as F1) ===
    plt.figure(figsize=(12, 0.5 * len(categories) + 1))
    plt.barh([y + bar_width / 2 for y in y_positions], sensitivities, height=bar_width, label='Sensitivity (%)', color='#1f77b4')
    plt.barh([y - bar_width / 2 for y in y_positions], precisions, height=bar_width, label='Precision (%)', color='#ff7f0e')

    for y, s, p in zip(y_positions, sensitivities, precisions):
        plt.text(s + 1, y + bar_width / 2, f'{s:.1f}%', va='center', ha='left', fontsize=8)
        plt.text(p + 1, y - bar_width / 2, f'{p:.1f}%', va='center', ha='left', fontsize=8)

    plt.yticks(y_positions, categories)
    plt.gca().invert_yaxis()  # Same top-down order as F1
    plt.xlim(0, 100)
    plt.xlabel('Percentage (%)')
    plt.title(f'Precision & Sensitivity per Category: {title}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{outDir}/plots/precision_sensitivity_{title}.svg', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/precision_sensitivity_{title}.png', bbox_inches='tight')

    plt.close()


##Counts instances of canonical/noncanonical conformations for each ME in a dictionary and prints summaries
def check_var_conformations(var_dict, dataset_name="Dataset"):
    Alu_conformations_canon = []
    L1_conformations_canon = []
    SVA_conformations_canon = []
    Alu_conformations_noncanon = []
    L1_conformations_noncanon = []
    SVA_conformations_noncanon = []
    for key, entry in var_dict.items():
        fam = entry.get("FAM_N")
        conformation = entry.get("CONFORMATION")
        is_non_canon = entry.get("NOT_CANONICAL", False)
        if fam == "Alu":
            if is_non_canon:
                Alu_conformations_noncanon.append(conformation)
            else: Alu_conformations_canon.append(conformation)
        elif fam == "L1":
            if is_non_canon:
                L1_conformations_noncanon.append(conformation)
            else: L1_conformations_canon.append(conformation)
        elif fam == "SVA":
            if is_non_canon:
                SVA_conformations_noncanon.append(conformation)
            else: SVA_conformations_canon.append(conformation)

    # Count and get top 10 conformations for each family
    top_Alu_canon = Counter(Alu_conformations_canon).most_common(10)
    top_L1_canon = Counter(L1_conformations_canon).most_common(10)
    top_SVA_canon = Counter(SVA_conformations_canon).most_common(10)
    top_Alu_noncanon = Counter(Alu_conformations_noncanon).most_common(10)
    top_L1_noncanon = Counter(L1_conformations_noncanon).most_common(10)
    top_SVA_noncanon = Counter(SVA_conformations_noncanon).most_common(10)

    # Print results
    print("Top 10 Alu canonical conformations:")
    for conf, count in top_Alu_canon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_Alu_canon, f"{dataset_name}_Top 10 Alu Canonical Conformations")

    print("Top 10 Alu non-canonical Conformations:")
    for conf, count in top_Alu_noncanon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_Alu_noncanon, f"{dataset_name}_Top 10 Alu Non-Canonical Conformations")

    print("\nTop 10 L1 canonical conformations:")
    for conf, count in top_L1_canon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_L1_canon, f"{dataset_name}_Top 10 L1 Canonical Conformations")

    print("\nTop 10 L1 non-canonical conformations:")
    for conf, count in top_L1_noncanon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_L1_noncanon, f"{dataset_name}_Top 10 L1 Non-Canonical Conformations")

    print("\nTop 10 SVA canonical conformations:")
    for conf, count in top_SVA_canon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_SVA_canon, f"{dataset_name}_Top 10 SVA Canonical Conformations")
    print("\nTop 10 SVA non-canonical conformations:")
    for conf, count in top_SVA_noncanon:
        print(f"{conf}: {count}")
    plot_top_conformations(top_SVA_noncanon, f"{dataset_name}_Top 10 SVA Non-Canonical Conformations")

    return top_Alu_canon, top_L1_canon, top_SVA_canon, top_Alu_noncanon, top_L1_noncanon, top_SVA_noncanon



##Join two dictionaries based on matching variant IDs, keeping the original data from each dictionary. Print information regarding mismatched strand and ME family infromation. Plot comparisons of variant sequence componentlengths and add the length difference as a key/value pair to each variant.
def combined_dict(dict1, dict2):
    combined = {}
    strand_mismatch_dict={}
    fam_mismatch_dict={}
    No_dict1_strand={}
    No_dict1_fam={}
    No_dict2_strand={}
    No_dict2_fam={}
    missing_both_strands=0
    missing_both_fams=0
    sva_hexamer_dict1=[]
    sva_hexamer_dict2=[]
    sva_hexamer_family_counts = {}
    sva_vntr_dict1=[]
    sva_vntr_dict2=[]
    sva_vntr_family_counts = {}
    prime3_td_dict1=[]
    prime3_td_dict2=[]
    prime3_td_family_counts = {}
    prime5_td_dict1=[]
    prime5_td_dict2=[]
    prime5_td_family_counts = {}
    all_prime3_td_dict1=[]
    all_prime3_td_dict2=[]
    all_polya_td_dict1=[]
    all_polya_td_dict2=[]
    all_prime5_td_dict2=[]
    all_prime5_td_dict1=[]
    all_tsd_dict1=[]
    all_tsd_dict2=[]
    all_sva_hexamer_dict1=[]
    all_sva_hexamer_dict2=[]
    all_sva_vntr_dict1=[]
    all_sva_vntr_dict2=[]
    polya_dict1=[]
    polya_dict2=[]
    polya_family_counts = {}
    tsd_dict1=[]
    tsd_dict2=[]
    tsd_family_counts = {}
    true_total_itype_and_fam = {"VNTR": 0, "DUP": 0, "INV_DUP": 0, "NUMT": 0, "orphan": 0, "Alu": 0, "L1": 0, "SVA": 0}
    false_pos_itype_and_fam = {"VNTR": 0, "DUP": 0, "INV_DUP": 0, "NUMT": 0, "orphan": 0, "Alu": 0, "L1": 0, "SVA": 0}
    true_pos_itype_and_fam = {"VNTR": 0, "DUP": 0, "INV_DUP": 0, "NUMT": 0, "orphan": 0, "Alu": 0, "L1": 0, "SVA": 0}
    total_pos_itype_and_fam = {"VNTR": 0, "DUP": 0, "INV_DUP": 0, "NUMT": 0, "orphan": 0, "Alu": 0, "L1": 0, "SVA": 0}
    true_total_l1_conf = defaultdict(int)
    true_pos_l1_conf = defaultdict(int)
    total_pos_l1_conf = {"TRUN+FOR+POLYA": 0, "FOR+POLYA": 0, "TRUN+REV+DEL+FOR+POLYA": 0, "TRUN+REV+DUP+FOR+POLYA": 0, "TRUN+FOR+POLYA+TD+POLYA": 0, "FOR+POLYA+TD+POLYA": 0, "TRUN+REV+DEL+FOR+POLYA+TD+POLYA": 0, "TD+FOR+POLYA": 0, "TRUN+REV+BLUNT+FOR+POLYA": 0, "TRUN+REV+DUP+FOR+POLYA+TD+POLYA": 0, "REV+DEL+FOR+POLYA": 0, "TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA": 0}
    false_pos_l1_conf = {"TRUN+FOR+POLYA": 0, "FOR+POLYA": 0, "TRUN+REV+DEL+FOR+POLYA": 0, "TRUN+REV+DUP+FOR+POLYA": 0, "TRUN+FOR+POLYA+TD+POLYA": 0, "FOR+POLYA+TD+POLYA": 0, "TRUN+REV+DEL+FOR+POLYA+TD+POLYA": 0, "TD+FOR+POLYA": 0, "TRUN+REV+BLUNT+FOR+POLYA": 0, "TRUN+REV+DUP+FOR+POLYA+TD+POLYA": 0, "REV+DEL+FOR+POLYA": 0, "TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA": 0}
    true_total_alu_conf = defaultdict(int)
    true_pos_alu_conf = defaultdict(int)
    total_pos_alu_conf = {"FOR+POLYA": 0, "TRUN+FOR+POLYA": 0}
    false_pos_alu_conf = {"FOR+POLYA": 0, "TRUN+FOR+POLYA": 0}
    true_total_sva_conf = defaultdict(int)
    true_pos_sva_conf = defaultdict(int)
    total_pos_sva_conf = {"Hexamer+Alu-like+VNTR+SINE-R+POLYA": 0, "VNTR+SINE-R+POLYA": 0, "MAST2+VNTR+SINE-R+POLYA": 0, "Alu-like+VNTR+SINE-R+POLYA": 0, "Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA": 0, "TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA": 0, "TD+MAST2+VNTR+SINE-R+POLYA": 0, "SINE-R+POLYA": 0, "VNTR+SINE-R+POLYA+TD+POLYA": 0, "Alu-like+VNTR+SINE-R+POLYA+TD+POLYA": 0, "MAST2+VNTR+SINE-R+POLYA+TD+POLYA": 0, "SINE-R+POLYA+TD+POLYA": 0}
    false_pos_sva_conf = {"Hexamer+Alu-like+VNTR+SINE-R+POLYA": 0, "VNTR+SINE-R+POLYA": 0, "MAST2+VNTR+SINE-R+POLYA": 0, "Alu-like+VNTR+SINE-R+POLYA": 0, "Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA": 0, "TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA": 0, "TD+MAST2+VNTR+SINE-R+POLYA": 0, "SINE-R+POLYA": 0, "VNTR+SINE-R+POLYA+TD+POLYA": 0, "Alu-like+VNTR+SINE-R+POLYA+TD+POLYA": 0, "MAST2+VNTR+SINE-R+POLYA+TD+POLYA": 0, "SINE-R+POLYA+TD+POLYA": 0}
    mismatch_dict1_chimera_dict={}
    mismatch_dict1_solo_dict={}
    mismatch_dict1_partnered_dict={}
    for variant_id in dict1:
        if variant_id not in dict2:
            continue
        dict1_entry = dict1[variant_id]
        dict2_entry = dict2[variant_id]
        dict1_fam = normalize(dict1_entry.get("FAM_N"))
        dict2_fam = normalize(dict2_entry.get("FAM_N"))
        dict1_strand = normalize(dict1_entry.get("STRAND"))
        dict2_strand = normalize(dict2_entry.get("STRAND"))
        dict1_I_type = normalize(dict1_entry.get("ITYPE_N"))
        dict2_I_type = normalize(dict2_entry.get("ITYPE_N"))
        dict1_conf = normalize(dict1_entry.get("CONFORMATION"))
        dict2_conf = normalize(dict2_entry.get("CONFORMATION"))
        fam_mismatch=f"{dict1_fam}_{dict2_fam}"
        strand_mismatch=f"{dict1_strand}_{dict2_strand}"
        if dict1_strand is None and dict2_strand is None:
            strand_match = "UNK"
            missing_both_strands+=1
        elif dict1_strand is None and dict2_strand is not None:
            if variant_id in No_dict1_strand:
                strand_match ="NO_DICT1"
            else: 
                No_dict1_strand[variant_id] = dict1_entry
                strand_match ="NO_DICT1"
        elif dict1_strand is not None and dict2_strand is None:
            if variant_id in No_dict2_strand:
                strand_match ="NO_DICT2"
            else: 
                No_dict2_strand[variant_id] = dict2_entry
                strand_match = "NO_DICT2"
        elif dict1_strand == dict2_strand:
            strand_match = dict1_strand
        else:
            strand_match = strand_mismatch
        if dict1_fam is None and dict2_fam is None:
            fam_match = "UNK"
            missing_both_fams+=1
        elif dict1_fam is None and dict2_fam is not None:
            if variant_id in No_dict1_fam:
                fam_match ="NO_DICT1"
            else: 
                No_dict1_fam[variant_id] = dict1_entry
                fam_match="NO_DICT1"
        elif dict1_fam is not None and dict2_fam is None:
            if variant_id in No_dict2_fam:
                fam_match="NO_DICT2"
            else: 
                No_dict2_fam[variant_id] = dict2_entry
                fam_match="NO_DICT2"
        elif dict1_fam == dict2_fam:
            fam_match = dict1_fam
        else:
            fam_match = fam_mismatch
        combined[variant_id] = {
            'VCF': dict1_entry,
            'CSV': dict2_entry,
            'STRAND_MATCH': strand_match,
            'FAM_MATCH': fam_match
            }
        valid_itypes = ["VNTR", "DUP", "INV_DUP", "NUMT", "orphan"]
        if dict1_I_type in valid_itypes:
            true_total_itype_and_fam[dict1_I_type] += 1  # Count all ground truth instances
            if dict1_I_type == dict2_I_type:
                true_pos_itype_and_fam[dict1_I_type] += 1  # Correct prediction

        if dict2_I_type in valid_itypes:
            total_pos_itype_and_fam[dict2_I_type] += 1  # Count all predictions made
            if dict2_I_type != dict1_I_type:
                false_pos_itype_and_fam[dict2_I_type] += 1  # Incorrect prediction

        family_conf_dicts = {"L1": (true_total_l1_conf, true_pos_l1_conf), "Alu": (true_total_alu_conf, true_pos_alu_conf), "SVA": (true_total_sva_conf, true_pos_sva_conf)}

        family_precision_conf_dicts = {"L1": (total_pos_l1_conf, false_pos_l1_conf), "Alu": (total_pos_alu_conf, false_pos_alu_conf), "SVA": (total_pos_sva_conf, false_pos_sva_conf)}

        if dict1_fam in family_conf_dicts:
            true_total_itype_and_fam[dict1_fam] += 1
            total_dict, true_pos_dict = family_conf_dicts[dict1_fam]
            total_dict[dict1_conf] += 1
            if dict1_conf == dict2_conf:
                true_pos_dict[dict1_conf] += 1
            if dict1_fam == dict2_fam:
                true_pos_itype_and_fam[dict1_fam] +=1
        if dict2_fam in family_precision_conf_dicts:
            total_pos_itype_and_fam[dict2_fam] +=1
            if dict2_fam != dict1_fam:
                false_pos_itype_and_fam[dict2_fam] += 1
            total_pos_dict, false_pos_dict = family_precision_conf_dicts[dict2_fam]
            if dict2_conf in total_pos_dict:
                total_pos_dict[dict2_conf] += 1
                if dict2_conf != dict1_conf:
                    false_pos_dict[dict2_conf] += 1
            

        if combined[variant_id]['FAM_MATCH']==fam_mismatch:
            fam_mismatch_dict[variant_id]=combined[variant_id]
            if dict1_I_type == "solo":
                mismatch_dict1_solo_dict[variant_id]=combined[variant_id]
            elif dict1_I_type == "partnered":
                mismatch_dict1_partnered_dict[variant_id]= combined[variant_id]
            elif dict1_I_type == "chimera":
                mismatch_dict1_chimera_dict[variant_id] = combined[variant_id]
        if combined[variant_id]['STRAND_MATCH']==strand_mismatch:
            strand_mismatch_dict[variant_id]=combined[variant_id]
        process_field("3PRIME_TD_LEN", "3PRIME_TD_LEN", all_prime3_td_dict1, all_prime3_td_dict2, prime3_td_dict1, prime3_td_dict2, dict1_entry, dict2_entry, combined, variant_id, prime3_td_family_counts)
        process_field("5PRIME_TD_LEN", "5PRIME_TD_LEN", all_prime5_td_dict1, all_prime5_td_dict2, prime5_td_dict1, prime5_td_dict2, dict1_entry, dict2_entry, combined, variant_id, prime5_td_family_counts)
        process_field("POLYA_LEN", "POLYA_LEN", all_polya_td_dict1, all_polya_td_dict2, polya_dict1, polya_dict2, dict1_entry, dict2_entry, combined, variant_id, polya_family_counts)
        process_field("TSD_LEN", "TSD_LEN", all_tsd_dict1, all_tsd_dict2, tsd_dict1, tsd_dict2, dict1_entry, dict2_entry, combined, variant_id, tsd_family_counts)
        if fam_match=="SVA":
            process_field("HEXAMER_LEN", "HEXAMER_LEN", all_sva_hexamer_dict1, all_sva_hexamer_dict2, sva_hexamer_dict1, sva_hexamer_dict2, dict1_entry, dict2_entry, combined, variant_id, sva_hexamer_family_counts)
            process_field("SVA_VNTR_Length", "VNTR_LEN", all_sva_vntr_dict1, all_sva_vntr_dict2, sva_vntr_dict1, sva_vntr_dict2, dict1_entry, dict2_entry, combined, variant_id, sva_vntr_family_counts)
    strand_mismatch_df=pd.DataFrame.from_dict(strand_mismatch_dict, orient='index')
    fam_mismatch_df=pd.DataFrame.from_dict(fam_mismatch_dict, orient='index')
    combined_df=pd.DataFrame.from_dict(combined, orient='index')
    print(f"\nstrand mismatches: {len(strand_mismatch_dict)}")
    print(f"\nME family mismatches: {len(fam_mismatch_dict)}")
    print(f"'Solo' labeled mismatches: {len(mismatch_dict1_solo_dict)}")
    print(f"'Partnered' labeled mismatches: {len(mismatch_dict1_partnered_dict)}")
    print(f"'Chimera' labeled mismatches: {len(mismatch_dict1_chimera_dict)}")
    print(f"\nvariants with missing strand information in both dictionaries: {missing_both_strands}")
    print(f"variants with missing ME family information in both dictionaries: {missing_both_fams}")
    print(f"Variants with missing strand info from dictionary 1: {len(No_dict1_strand)} ")
    print(f"Variants with missing ME family info from dictionary 1: {len(No_dict1_fam)} ")
    print(f"Variants with missing strand info from dictionary 2: {len(No_dict2_strand)} ")
    print(f"Variants with missing ME family info from dictionary 2: {len(No_dict2_fam)} ")
    plot_len_diff_no_outliers(prime3_td_dict1, prime3_td_dict2, "3 Prime TD Length Comparison", label1="SVModeller", label2="SVAN")
    plot_len_diff_no_outliers(polya_dict1, polya_dict2, "PolyA Length Comparison", label1="SVModeller", label2="SVAN")
    plot_len_diff_no_outliers(tsd_dict1, tsd_dict2, "TSD Length Comparison", label1= "SVModeller", label2="SVAN")
    plot_len_diff_no_outliers(prime5_td_dict1, prime5_td_dict2, "5 Prime TD Length Comparison", label1="SVModeller", label2="SVAN")
    plot_len_diff_no_outliers(sva_hexamer_dict1, sva_hexamer_dict2, "SVA Hexamer Length Comparison", label1="SVModeller", label2="SVAN")
    plot_len_diff_no_outliers(sva_vntr_dict1, sva_vntr_dict2, "SVA VNTR Length Comparison", label1="SVModeller", label2="SVAN")
    print(f"5 prime transductions in dict1:{len(all_prime5_td_dict1)}")
    print(f"5 prime transductions in dict2:{len(all_prime5_td_dict2)}")
    print(f"5 prime transduction sensitivity: {len(prime5_td_dict2)/len(all_prime5_td_dict1)}")
    print(prime5_td_family_counts)
    print(f"3 prime transductions in dict1:{len(all_prime3_td_dict1)}")
    print(f"3 prime transductions in dict2:{len(all_prime3_td_dict2)}")
    print(f"3 prime transduction sensitivity: {len(prime3_td_dict2)/len(all_prime3_td_dict1)}")
    print(prime3_td_family_counts)
    print(f"Poly As measured in dict1:{len(all_polya_td_dict1)}")
    print(f"Poly As measured in dict2:{len(all_polya_td_dict2)}")
    print(f"PolyA tail sensitivity:{len(polya_dict2)/len(all_polya_td_dict1)}")
    print(polya_family_counts)
    print(f"TSDs measured in dict1:{len(all_tsd_dict1)}")
    print(f"TSDs measured in dict2:{len(all_tsd_dict2)}")
    print(f"TSDs sensitivity:{len(tsd_dict2)/len(all_tsd_dict1)}")
    print(tsd_family_counts)
    print(f"SVA Hexamers measured in dict 1:{len(all_sva_hexamer_dict1)}")
    print(f"SVA Hexamers measured in dict 2:{len(all_sva_hexamer_dict2)}")
    print(f"SVA Hexamer sensitivity:{len(sva_hexamer_dict2)/len(all_sva_hexamer_dict1)}")
    print(sva_hexamer_family_counts)
    print(f"SVA VNTRs measured in dict 1:{len(all_sva_vntr_dict1)}")
    print(f"SVA VNTRs measured in dict 2:{len(all_sva_vntr_dict2)}")
    print(f"SVA VNTRs sensitivity:{len(sva_vntr_dict2)/len(all_sva_vntr_dict1)}")
    print(sva_vntr_family_counts)
    print(true_total_itype_and_fam)
    print(true_pos_itype_and_fam)
    print("L1 Conformations")
    print(true_total_l1_conf)
    print("Alu Conformations")
    print(true_total_alu_conf)
    print("SVA Conformations")
    print(true_total_sva_conf)
    plot_sensitivity(true_total_itype_and_fam, true_pos_itype_and_fam, "Insertion Type and ME Family")
    plot_sensitivity(true_total_l1_conf, true_pos_l1_conf, "L1 Structural Conformation")
    plot_sensitivity(true_total_alu_conf, true_pos_alu_conf, "Alu Conformations")
    plot_sensitivity(true_total_sva_conf, true_pos_sva_conf, "SVA Conformation")
    plot_precision(total_pos_itype_and_fam, false_pos_itype_and_fam, "Insertion Type and ME Family")
    plot_precision(total_pos_alu_conf, false_pos_alu_conf, "Alu Conformations")
    plot_precision(total_pos_l1_conf, false_pos_l1_conf, "L1 Structural Conformation")
    plot_precision(total_pos_sva_conf, false_pos_sva_conf, "SVA Conformation")
    plot_metrics(true_total_itype_and_fam, true_pos_itype_and_fam, total_pos_itype_and_fam, "Insertion Type and ME Family")
    plot_metrics(true_total_alu_conf, true_pos_alu_conf, total_pos_alu_conf, "Alu Conformations")
    plot_metrics(true_total_l1_conf, true_pos_l1_conf, total_pos_l1_conf, "L1 Structural Conformation")
    plot_metrics(true_total_sva_conf, true_pos_sva_conf, total_pos_sva_conf, "SVA Conformation")
    combined_df.to_csv("combined_variants.csv", index=True)
    strand_mismatch_df.to_csv("strand_mismatches.csv", index=True)
    fam_mismatch_df.to_csv("fam_mismatches.csv", index=True)
    return combined

##Count amount of canonical/noncanonical variants found in each chromosome
def count_canon_by_chrom(variant_dict):
    canon_counts = defaultdict(int)
    noncanon_counts = defaultdict(int)

    for variant in variant_dict:
        variant_data = variant_dict[variant]
        
        chrom = variant_data['CHROM']
        # Check if the variant has the 'NOT_CANONICAL' flag to determine if it's non-canonical
        is_canon = 'NOT_CANONICAL' not in variant_data

        if is_canon:
            canon_counts[chrom] += 1
        else:
            noncanon_counts[chrom] += 1

    return dict(canon_counts), dict(noncanon_counts)

def plot_canon_dist(variant_dicts):
    for title, variant_dict in variant_dicts.items():
        canon_counts, noncanon_counts = count_canon_by_chrom(variant_dict)

        # Get all unique chromosomes from both canonical and non-canonical counts
        all_chroms = sorted(set(canon_counts.keys()) | set(noncanon_counts.keys()))

        # Remove 'chr' prefix and sort the chromosomes numerically and alphabetically
        all_chroms_sorted = sorted(all_chroms, key=lambda chrom: (int(chrom[3:]) if chrom[3:].isdigit() else float('inf'), chrom))

        print(all_chroms_sorted)

        # Use .get() to safely access missing chromosomes
        canon_vals = [canon_counts.get(chrom, 0) for chrom in all_chroms_sorted]
        noncanon_vals = [noncanon_counts.get(chrom, 0) for chrom in all_chroms_sorted]

        x = range(len(all_chroms_sorted))
        width = 0.4

        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], canon_vals, width=width, label='Canonical', color='green')
        plt.bar([i + width/2 for i in x], noncanon_vals, width=width, label='Non-Canonical', color='orange')

        plt.xlabel('Chromosome')
        plt.ylabel('Variant Count')
        plt.title(f'Canonical vs Non-Canonical Variants by Chromosome: {title}')
        plt.xticks(ticks=x, labels=all_chroms_sorted, rotation=45)
        plt.legend()
        plt.savefig(f'{outDir}/plots/{title}_chrom_dist.svg', bbox_inches='tight')
        plt.close()



combined_dict(vcf_2_dict(VCF1), vcf_2_dict(VCF2))