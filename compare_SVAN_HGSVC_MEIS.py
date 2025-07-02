
import formats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.stats import spearmanr

#load VCF object
VCF = formats.VCF()
#Load vcf and csv file
VCF.read("/users/brodriguez/asanchez/SVAN/output/Complex_gen_var_Logsdon_paper/annot_INS/Complex_gen_var_Logsdon.alt.vcf.unphased.INS.SVAN.vcf")
data_csv=pd.read_csv("/users/brodriguez/asanchez/SVAN/input/Complex_gen_var_paper_100gp_data/MEI_csvs/MEI_Callset_T2T-CHM13.ALL.20241211.csv")
outDir="/users/brodriguez/asanchez/scripts/MEI_annotations/"
#function that extracts the chromosome and info from each vcf variant and saves it into a nested dictionary where the variant ID is the key.
def vcf_2_dict(vcf):
    variant_info_dict = {}

    for variant in vcf.variants:
        variant_info_dict[variant.ID] = variant.info.copy()
        variant_info_dict[variant.ID]["CHROM"] = variant.chrom 

    return variant_info_dict       


# Function to create a dictionary from the csv, which saves chromosome information, TE family, and variant caller info into a nested dictionary where the variant ID is the key.
def mei_csv_2_dict(csv):
    csv_dict = {}
    for index, variant in csv.iterrows():
        variant_id = variant['ID']
        
        csv_dict[variant_id] = {
            "CHROM": variant["CHROM"]
        }

        TE = variant['TE_Designation']
        if TE == "SINE/Alu":
            csv_dict[variant_id]['FAM_N'] = "Alu"
        elif TE == "LINE/L1":
            csv_dict[variant_id]['FAM_N'] = "L1"
        elif TE == "Retroposon/SVA":
            csv_dict[variant_id]['FAM_N'] = "SVA"
        else:
            csv_dict[variant_id]['FAM_N'] = TE
        csv_info = variant['INFO'].split(";")
        for value in csv_info:
            if "=" in value:
                entry = value.split("=")
                key = entry[0]
                val = entry[1]
                if key == "SVLEN":
                    csv_dict[variant_id]["INS_LEN"] = val


        if variant['PALMER'] == 1:
            palmer_info = variant['PALMER_INFO'].split(";")
            for value in palmer_info:
                if "=" in value:
                    entry = value.split("=")
                    key = entry[0]
                    val = entry[1]
                    if key == "3TD_LEN":
                        csv_dict[variant_id]["3PRIME_TD_LEN"] = val
                    elif key == "5TD_LEN":
                        csv_dict[variant_id]["5PRIME_TD_LEN"] = val
                    elif key == "TSD_LEN":
                        length= val.split(",")
                        csv_dict[variant_id][key] = length[0]
                    else:
                        csv_dict[variant_id][key] = val
        else:
            l1me_info = variant['L1ME-AID_INFO'].split(";")
            for value in l1me_info:
                if ":" in value:
                    l1me_entry = value.split(":")
                    if len(l1me_entry) == 2:
                        key, val = l1me_entry
                    if key == "Orientation":
                        csv_dict[variant_id]["STRAND"] = val
                    else: csv_dict[variant_id][key] = val

    return csv_dict


#Function to join two directories based on variant ID, and create several comparisons per variant.
##Function to change all empty values into None 
def normalize(value):
    if value in [None, "", ".", "NONE", "?", "NA"]:
        return None
    return value

def safe_int(val):
    try:
        if val is None or str(val).strip().upper() == "NA":
            return None
        return int(val)
    except (ValueError, TypeError):
        return None
##Function to filter outliers larger than 3 stdevs for plotting 
def filter_outliers(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)

    mean_list1 = np.mean(list1)
    std_list1 = np.std(list1)
    lower_bound_list1 = mean_list1 - 3 * std_list1
    upper_bound_list1 = mean_list1 + 3 * std_list1

    mean_list2 = np.mean(list2)
    std_list2 = np.std(list2)
    lower_bound_list2 = mean_list2 - 3 * std_list2
    upper_bound_list2 = mean_list2 + 3 * std_list2

    valid_indices = np.where((list1 >= lower_bound_list1) & (list1 <= upper_bound_list1) & (list2 >= lower_bound_list2) & (list2 <= upper_bound_list2))[0]

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
            
##Function to create horizontal bar plots for the top 10 conformations that takes the lists and title for the plot as input
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

##Function to create plots of insertion type per family that takes lists of integers and a dataset title as input.
def plot_Itype_by_fam(list1, list2, list3, list4, dataset_name="dataset", outDir="."):
    category_order = ['PSD', 'chimera', 'orphan', 'partnered', 'solo']
    category_colors = {
        'PSD': '#2ca02c',       # Blue
        'chimera': '#ff7f0e',   # Orange
        'orphan': '#d62728',    # Green
        'partnered': '#9467bd', # Red
        'solo': '#1f77b4'       # Purple
    }

    # Convert input lists to value counts
    counts = {
        'Alus': pd.Series(list1).value_counts(),
        'L1s': pd.Series(list2).value_counts(),
        'SVAs': pd.Series(list3).value_counts(),
        'Other': pd.Series(list4).value_counts()
    }

    # Create DataFrame and reindex to include all categories in consistent order
    Itype_counts_df = pd.DataFrame(counts).fillna(0).astype(int).reindex(category_order).fillna(0)

    # Plot
    ax = Itype_counts_df.T.plot(
        kind='bar', 
        stacked=True, 
        color=[category_colors[cat] for cat in Itype_counts_df.index]
    )
    plt.ylabel("Count")
    plt.title(f"Insertion Type by Family: {dataset_name}")
    plt.legend(title="Insertion Type")
    plt.tight_layout()

    # Add annotations
    for bar_group in ax.containers:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                if height >= 5:
                    va = 'center'
                    y = bar.get_y() + height / 2
                else:
                    va = 'bottom'
                    y = bar.get_y() + height + 1
                ax.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    ha='center',
                    va=va,
                    fontsize=8,
                    color='black'
                )

    # Save
    plt.savefig(f'{outDir}/plots/Itype_by_fam_matches_{dataset_name}.svg', bbox_inches='tight')
    plt.savefig(f'{outDir}/plots/Itype_by_fam_matches_{dataset_name}.png', bbox_inches='tight')
    plt.close()

##Version of the function to plot length comparisons for variant components such as polyA tails and TSDs that filters values outsides the interquartile range.
def plot_len_diff_no_outliers(list1, list2, title, label1="List 1", label2="List 2"):
    filtered_list1, filtered_list2 = filter_outliers(list1, list2)
    spearman_r, _ = spearmanr(list1, list2)
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

##function to plot length comparisons without filtering outliers.
def plot_len_diff(list1, list2, title, label1="List 1", label2="List 2"):
    spearman_r, _ = spearmanr(list1, list2)
    min_value = min(np.min(list1), np.min(list2))
    max_value = max(np.max(list1), np.max(list2))
    plt.scatter(list1, list2)
    plt.xlim(0, max_value)
    plt.ylim(0,max_value)
    plt.title(f'{title}\nSpearman ρ = {spearman_r:.3f}')
    plt.xlabel(f'length values for {label1}')
    plt.ylabel(f'length values for {label2}')
    filename = title.replace(" ", "_").replace(":", "") + ".svg"
    plt.savefig(f'{outDir}/plots/{filename}', bbox_inches='tight')
    plt.close()
    min_value = min(np.min(list1), np.min(list2))
    max_value = max(np.max(list1), np.max(list2))
    bins= np.linspace(min_value, max_value, 20)
    plt.hist([list1,list2], bins=bins, label=[label1, label2])
    plt.legend(loc='upper right')
    plt.title(f"{label1}/{label2} {title}")
    plt.savefig(f'{outDir}/plots/hist_{filename}', bbox_inches='tight')
    plt.close()

###Function to separate variants from dictionaries into lists depending on canonical/noncanonical conformation and ME family.
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

    # Print results and create plots of top conformations
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


##Function to create a combined dictionary for variants with matching IDs from two different dictionaries
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
    prime3_td_dict1=[]
    prime3_td_dict2=[]
    prime5_td_dict1=[]
    prime5_td_dict2=[]
    all_prime3_td_dict1=[]
    all_prime3_td_dict2=[]
    all_polya_td_dict1=[]
    all_polya_td_dict2=[]
    all_prime5_td_dict2=[]
    all_prime5_td_dict1=[]
    all_tsd_dict1=[]
    all_tsd_dict2=[]
    polya_dict1=[]
    polya_dict2=[]
    tsd_dict1=[]
    tsd_dict2=[]
    Alu_I_types=[]
    L1_I_types =[]
    SVA_Itypes=[]
    other_Itypes=[]
    mismatch_dict1_chimera_dict={}
    mismatch_dict1_solo_dict={}
    mismatch_dict1_partnered_dict={}
    for variant_id in dict1:
        if variant_id not in dict2:
            continue
        dict1_entry = dict1[variant_id]
        dict2_entry = dict2[variant_id]
##Extract strand and ME family from each match, as well as insertion type from the SVAN vcf
        dict1_strand = normalize(dict1_entry.get("STRAND"))
        dict2_strand = normalize(dict2_entry.get("STRAND"))
        dict1_fam = normalize(dict1_entry.get("FAM_N"))
        dict2_fam = normalize(dict2_entry.get("FAM_N"))
        dict1_I_type = normalize(dict1_entry.get("ITYPE_N"))
        fam_mismatch=f"{dict1_fam}_{dict2_fam}"
        strand_mismatch=f"{dict1_strand}_{dict2_strand}"
##Separate entries into dictionaries based on matches/mismatches of strand orientation and ME family annotation.
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
##Add the information from both dictionarie into the combined dictionary, and add key/value pairs with information on insertion length difference, as well as strand and ME family matches/mismatches.
        combined[variant_id] = {
            'VCF': dict1_entry,
            'CSV': dict2_entry,
            'INS_LEN_DIFF': int(dict1_entry["INS_LEN"]) - int(dict2_entry["INS_LEN"]),
            'STRAND_MATCH': strand_match,
            'FAM_MATCH': fam_match
            }
        combined_entry = combined[variant_id]
        combined_fam = normalize(combined_entry.get("FAM_MATCH"))
##Create lists for each ME family match with the insertion types for each family.
        if dict1_I_type is not None:
            combined_entry["ITYPE_N"]=dict1_I_type
            if combined_fam=="Alu":
                Alu_I_types.append(dict1_I_type)
            elif combined_fam=="L1":
                L1_I_types.append(dict1_I_type)
            elif  combined_fam=="SVA":
                SVA_Itypes.append(dict1_I_type)
            else: other_Itypes.append(dict1_I_type)
        else: combined_entry["ITYPE_N"] = "NA"
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
##Create lists with the lengths of 3prime TDs, 5prime TDs, PolyAs, and TSDs found in each dictionary in order to calculate the lenght differences and add them as a key/value pair in each match. Only does this for variants that contain length information in both dictionaries.
        val1 = safe_int(dict1_entry.get("3PRIME_TD_LEN"))
        if val1 is not None: 
            if val1 != 0:
                all_prime3_td_dict1.append(val1)
        val2 = safe_int(dict2_entry.get("3PRIME_TD_LEN"))
        if val2 is not None:
            if val2 != 0:
                all_prime3_td_dict2.append(val2)
        if val1 is not None and val2 is not None:
            diff_3prime = val1 - val2
            combined[variant_id]['3PRIME_TD_LEN_DIFF'] = diff_3prime
            prime3_td_dict1.append(val1)
            prime3_td_dict2.append(val2)
        else:
            combined[variant_id]['3PRIME_TD_LEN_DIFF'] = "NA"


        val1 = safe_int(dict1_entry.get("5PRIME_TD_LEN"))
        if val1 is not None: 
            if val1 != 0:
                all_prime5_td_dict1.append(val1)
        val2 = safe_int(dict2_entry.get("5PRIME_TD_LEN"))
        if val2 is not None: 
            if val2 != 0:
                all_prime5_td_dict2.append(val2)
        if val1 is not None and val2 is not None:
            diff_5prime = val1 - val2
            combined[variant_id]['5PRIME_TD_LEN_DIFF'] = diff_5prime
            prime5_td_dict1.append(val1)
            prime5_td_dict2.append(val2)
        else:
            combined[variant_id]['5PRIME_TD_LEN_DIFF'] = "NA"

        val1 = safe_int(dict1_entry.get("POLYA_LEN"))
        if val1 is not None: 
            if val1 != 0:
                all_polya_td_dict1.append(val1)
        val2 = safe_int(dict2_entry.get("POLYA_LEN"))
        if val2 is not None:
            if val2 != 0:
                all_polya_td_dict2.append(val2)
        if val1 is not None and val2 is not None:
            diff_polya = val1 - val2
            combined[variant_id]['POLYA_LEN_DIFF'] = diff_polya
            polya_dict1.append(val1)
            polya_dict2.append(val2)
        else:
            combined[variant_id]['POLYA_LEN_DIFF'] = "NA"
        val1 = safe_int(dict1_entry.get("TSD_LEN"))
        if val1 is not None: 
            if val1 != 0:
                all_tsd_dict1.append(val1)
        val2 = safe_int(dict2_entry.get("TSD_LEN"))
        if val2 is not None:
            if val2 != 0:
                all_tsd_dict2.append(val2)
        if val1 is not None and val2 is not None:
            diff_tsd = val1 - val2
            combined[variant_id]['TSD_LEN_DIFF'] = diff_tsd
            tsd_dict1.append(val1)
            tsd_dict2.append(val2)
        else:
            combined[variant_id]['TSD_LEN_DIFF'] = "NA"
##Create dataframes for variants with mismatched strand and ME family information, as well as for the combined dictionary in order to convert the information into CSVs for manual inspection.
    strand_mismatch_df=pd.DataFrame.from_dict(strand_mismatch_dict, orient='index')
    fam_mismatch_df=pd.DataFrame.from_dict(fam_mismatch_dict, orient='index')
    combined_df=pd.DataFrame.from_dict(combined, orient='index')
    print(f"\nstrand mismatches: {len(strand_mismatch_dict)}")
    print(strand_mismatch_df)
    print(f"\nME family mismatches: {len(fam_mismatch_dict)}")
    print(f"'Solo' labeled mismatches: {len(mismatch_dict1_solo_dict)}")
    print(f"'Partnered' labeled mismatches: {len(mismatch_dict1_partnered_dict)}")
    print(f"'Chimera' labeled mismatches: {len(mismatch_dict1_chimera_dict)}")
    print(fam_mismatch_df)
    print(f"\nvariants with missing strand information in both dictionaries: {missing_both_strands}")
    print(f"variants with missing ME family information in both dictionaries: {missing_both_fams}")
    print(f"Variants with missing strand info from dictionary 1: {len(No_dict1_strand)} ")
    print(f"Variants with missing ME family info from dictionary 1: {len(No_dict1_fam)} ")
    print(f"Variants with missing strand info from dictionary 2: {len(No_dict2_strand)} ")
    print(f"Variants with missing ME family info from dictionary 2: {len(No_dict2_fam)} ")
    plot_len_diff_no_outliers(prime3_td_dict1, prime3_td_dict2, "3 Prime TD Length Comparison", label1="SVAN", label2="HGSVC3")
    plot_len_diff_no_outliers(polya_dict1, polya_dict2, "PolyA Length Comparison", label1="SVAN", label2="HGSVC3")
    plot_len_diff_no_outliers(tsd_dict1, tsd_dict2, "TSD Length Comparison", label1= "SVAN", label2="HGSVC3")
    print(prime5_td_dict1)
    print(prime5_td_dict2)
    print(f"5 prime transductions in dict1:{len(all_prime5_td_dict1)}")
    print(all_prime5_td_dict1)
    print(f"5 prime transductions in dict2:{len(all_prime5_td_dict2)}")
    print(f"3 prime transductions in dict1:{len(all_prime3_td_dict1)}")
    print(f"3 prime transductions in dict2:{len(all_prime3_td_dict2)}")
    print(f"Poly As measured in dict1:{len(all_polya_td_dict1)}")
    print(f"Poly As measured in dict2:{len(all_polya_td_dict2)}")
    print(f"TSDs measured in dict1:{len(all_tsd_dict1)}")
    print(f"TSDs measured in dict2:{len(all_tsd_dict2)}")
    print(f"TSDs measured in both dictionaries:{len(tsd_dict1)}")
    print(f"Poly As measured in both dictionaries:{len(polya_dict1)}")
    print(f"3 prime transductions in both dictionaries:{len(prime3_td_dict1)}")
    print(f"5 prime transductions in both dictionaries:{len(prime5_td_dict1)}")
    print(all_prime5_td_dict2)
    combined_df.to_csv("combined_variants.csv", index=True)
    strand_mismatch_df.to_csv("strand_mismatches.csv", index=True)
    fam_mismatch_df.to_csv("fam_mismatches.csv", index=True)
    return combined

##Function to separate variants in a dictionary into separate dictionaries based on ME family and canonical/noncanonical conformation. Print total values for each category at the end.
def group_and_count_MEs(var_dict):
    Alu_canon = {}
    Alu_noncanon = {}
    L1_canon = {}
    L1_noncanon = {}
    SVA_canon = {}
    SVA_noncanon = {}
    Alu_I_types_canon=[]
    L1_I_types_canon =[]
    SVA_I_types_canon=[]
    other_Itypes_canon=[]
    Alu_I_types_noncanon=[]
    L1_I_types_noncanon =[]
    SVA_Itypes_noncanon=[]
    other_Itypes_noncanon=[]
    MEI_Itypes=["PSD", "chimera", "orphan", "partnered", "solo"]

    for var_id, entry in var_dict.items():
        fam = entry.get("FAM_N")
        is_non_canon = entry.get("NOT_CANONICAL", False)
        Itype = entry.get("ITYPE_N", None)
        if Itype not in MEI_Itypes: continue
        if fam == "Alu":
            if is_non_canon:
                Alu_noncanon[var_id] = entry
                Alu_I_types_noncanon.append(Itype)
            else:
                Alu_canon[var_id] = entry
                Alu_I_types_canon.append(Itype)
        elif fam == "L1":
            if is_non_canon:
                L1_noncanon[var_id] = entry
                L1_I_types_noncanon.append(Itype)
            else:
                L1_canon[var_id] = entry
                L1_I_types_canon.append(Itype)
        elif fam == "SVA":
            if is_non_canon:
                SVA_noncanon[var_id] = entry
                SVA_Itypes_noncanon.append(Itype)
            else:
                SVA_canon[var_id] = entry
                SVA_I_types_canon.append(Itype)
        else:
            if is_non_canon:
                other_Itypes_noncanon.append(Itype)
            else:
                other_Itypes_canon.append(Itype)

    print(f"Total:{len(Alu_canon) + len(Alu_noncanon) + len(L1_canon) + len(L1_noncanon) + len(SVA_canon) + len(SVA_noncanon)}")
    print(f"Alus: {len(Alu_canon) + len(Alu_noncanon)} (Canonical: {len(Alu_canon)}, Non-canonical: {len(Alu_noncanon)})")
    print(f"L1s: {len(L1_canon) + len(L1_noncanon)} (Canonical: {len(L1_canon)}, Non-canonical: {len(L1_noncanon)})")
    print(f"SVAs: {len(SVA_canon) + len(SVA_noncanon)} (Canonical: {len(SVA_canon)}, Non-canonical: {len(SVA_noncanon)})\n")

    
    return (Alu_I_types_canon, L1_I_types_canon, SVA_I_types_canon, other_Itypes_canon,
    Alu_I_types_noncanon, L1_I_types_noncanon, SVA_Itypes_noncanon, other_Itypes_noncanon
)

def group_and_count_MEs_in_csv(var_dict):
    Alu = {}
    L1 = {}
    SVA = {}

    for var_id, entry in var_dict.items():
        fam = entry.get("FAM_N")
        if fam == "Alu":
            Alu[var_id] = entry
        elif fam == "L1":
            L1[var_id] = entry
        elif fam == "SVA":
                SVA[var_id] = entry

    print(f"Total:{len(Alu)+ len(L1)+ len(SVA)}")
    print(f"Alus: {len(Alu)} ")
    print(f"L1s: {len(L1)}")
    print(f"SVAs: {len(SVA)}\n")

    
    return (Alu, L1, SVA
)

def count_mei_features(entry: dict) -> tuple[int, int, int, int]:
    return (
        int(bool(safe_int(entry.get("3PRIME_TD_LEN")))),
        int(bool(safe_int(entry.get("5PRIME_TD_LEN")))),
        int(bool(safe_int(entry.get("TSD_LEN")))),
        int(bool(safe_int(entry.get("POLYA_LEN")))),
    )

#Function to group vcf and csv dictionary into matches and unique variants per dictionary.
def check_var_pres(vcf, csv):
    vcf_dict = vcf_2_dict(vcf)
    csv_dict = mei_csv_2_dict(csv)

    matches = {}
    not_in_csv = {}
    not_in_vcf = {}
    MEI_fams=["Alu","L1","SVA"]
    total_vcf_3prime_td=0
    total_vcf_5prime_td=0
    total_vcf_tsds=0
    total_vcf_polya=0
    total_csv_3prime_td=0
    total_csv_5prime_td=0
    total_csv_tsds=0
    total_csv_polya=0
    for variant_id in vcf_dict:
        vcf_entry = vcf_dict[variant_id]
        if vcf_entry.get("FAM_N") in MEI_fams:
            td3, td5, tsd, polya = count_mei_features(vcf_entry)
            total_vcf_3prime_td += td3
            total_vcf_5prime_td += td5
            total_vcf_tsds += tsd
            total_vcf_polya += polya
        if variant_id in csv_dict:
            matches[variant_id] = vcf_dict[variant_id]
        else:
            not_in_csv[variant_id] = vcf_dict[variant_id]

    for variant_id in csv_dict:
        csv_entry = csv_dict[variant_id]
        td3, td5, tsd, polya = count_mei_features(csv_entry)
        total_csv_3prime_td += td3
        total_csv_5prime_td += td5
        total_csv_tsds += tsd
        total_csv_polya += polya
        if variant_id not in vcf_dict:
            not_in_vcf[variant_id] = csv_dict[variant_id]
    print(f"Total ME 3prime TDs in SVAN VCF:{total_vcf_3prime_td}")
    print(f"Total ME 5prime TDs in SVAN VCF:{total_vcf_5prime_td}")
    print(f"Total TSDs in SVAN VCF:{total_vcf_tsds}")
    print(f"Total PolyA tails in SVAN VCF:{total_vcf_polya}")
    print(f"Total ME 3prime TDs in HGSVC CSV:{total_csv_3prime_td}")
    print(f"Total ME 5prime TDs in HGSVC CSV:{total_csv_5prime_td}")
    print(f"Total TSDs in HGSVC CSV:{total_csv_tsds}")
    print(f"Total PolyA tails in HGSVC CSV:{total_csv_polya}")
    print("== ME call matches ==")
    (
    alu_c, l1_c, sva_c, other_c,
    alu_nc, l1_nc, sva_nc, other_nc
) = group_and_count_MEs(matches)
    check_var_conformations(matches, dataset_name="Matches")
    plot_Itype_by_fam(alu_c, l1_c, sva_c, other_c, dataset_name="Canonical Matches")
    plot_Itype_by_fam(alu_nc, l1_nc, sva_nc, other_nc, dataset_name="NonCanonical Matches")

    print("\n== MEs called by SVAN but not in HGSVC ==")
    (
    alu_c, l1_c, sva_c, other_c,
    alu_nc, l1_nc, sva_nc, other_nc
) = group_and_count_MEs(not_in_csv)
    check_var_conformations(not_in_csv, dataset_name="Unique to SVAN")
    plot_Itype_by_fam(alu_c, l1_c, sva_c, other_c, dataset_name="Canonical Unique to SVAN")
    plot_Itype_by_fam(alu_nc, l1_nc, sva_nc, other_nc, dataset_name="NonCanonical Unique to SVAN")

    print("\n== MEs in HGSVC but not called by SVAN ==")
    group_and_count_MEs_in_csv(not_in_vcf)

    return matches, not_in_csv, not_in_vcf

##Function to count number of canonical/noncanonical variants per chromosome in a dictionary.
def count_canon_by_chrom(variant_dict):
    canon_counts = defaultdict(int)
    noncanon_counts = defaultdict(int)

    for variant in variant_dict:
        variant_data = variant_dict[variant]
        
        chrom = variant_data['CHROM']
        #Check if the variant has the 'NOT_CANONICAL' flag to determine if it's non-canonical
        is_canon = 'NOT_CANONICAL' not in variant_data

        if is_canon:
            canon_counts[chrom] += 1
        else:
            noncanon_counts[chrom] += 1

    return dict(canon_counts), dict(noncanon_counts)

##Function to plot distribution of canonical/noncanonical variants per chromosome in a dictionary.
def plot_canon_dist(variant_dicts):
    for title, variant_dict in variant_dicts.items():
        canon_counts, noncanon_counts = count_canon_by_chrom(variant_dict)

        #Get all unique chromosomes from both canonical and non-canonical counts
        all_chroms = sorted(set(canon_counts.keys()) | set(noncanon_counts.keys()))

        #Remove 'chr' prefix and sort the chromosomes numerically and alphabetically
        all_chroms_sorted = sorted(all_chroms, key=lambda chrom: (int(chrom[3:]) if chrom[3:].isdigit() else float('inf'), chrom))

        print(all_chroms_sorted)

        #Use .get() to safely access missing chromosomes
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
        plt.savefig(f'{outDir}/plots/{title}_chrom_dist.png', bbox_inches='tight')
        plt.close()


matches, not_in_csv, not_in_vcf = check_var_pres(VCF, data_csv)

combined_dict(vcf_2_dict(VCF), mei_csv_2_dict(data_csv))
